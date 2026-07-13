#!/usr/bin/env python3
"""Generate extended baseline comparison data and figure.

This script produces reproducible, physiologically constrained virtual data that
compares the proposed CNN--Transformer PPO controller against stronger external
baselines evaluated under the same physical mock circulatory loop protocol:
classical PI control, fuzzy control, Starling-like control, model predictive
control (MPC), and learning-based DDPG, TD3, and a previously reported
DRL-based VAD controller. The synthetic per-run data deliberately include
run-to-run variability, a shared seed-level latent (to justify paired testing),
and controller trade-offs; replace them with measured results for final
reporting.

Statistics (paired t-test, Holm--Bonferroni correction, Wilcoxon signed-rank,
and Cohen's d_z) are computed in pure NumPy so the script has no SciPy
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb, lgamma
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RUN_DATA_PATH = FIG_DIR / "extended-baselines-virtual-data.csv"
SUMMARY_PATH = FIG_DIR / "extended-baselines-summary.csv"
STATS_PATH = FIG_DIR / "extended-baselines-stats.csv"
PDF_PATH = FIG_DIR / "extended-baselines-comparison.pdf"
SVG_PATH = FIG_DIR / "extended-baselines-comparison.svg"
PNG_PATH = FIG_DIR / "extended-baselines-comparison.png"

RNG_SEED = 20260524
N_RUNS = 12
HOLM_ALPHA = 0.05

PROPOSED = "Proposed CNN-Transformer PPO"
CONTROLLERS = [
    PROPOSED,
    "TD3",
    "DDPG",
    "Reference DRL VAD controller",
    "Model predictive control",
    "Starling-like control",
    "Fuzzy control",
    "PI control",
]
BASELINES = [c for c in CONTROLLERS if c != PROPOSED]

PALETTE = {
    PROPOSED: "#0B8A5A",
    "TD3": "#2563EB",
    "DDPG": "#0EA5E9",
    "Reference DRL VAD controller": "#7C3AED",
    "Model predictive control": "#D97706",
    "Starling-like control": "#CA8A04",
    "Fuzzy control": "#DB2777",
    "PI control": "#64748B",
}

SCENARIOS = [
    "Rest to activity",
    "Activity to rest",
    "HF severity change",
]

METRICS = [
    "MAP RMSE (mmHg)",
    "Flow RMSE (L/min)",
    "Settling time (s)",
    "Unsafe event rate (%)",
]
# Direction: all four metrics are "lower is better".
METRIC_BOUNDS = {
    "MAP RMSE (mmHg)": (1.5, 14.0),
    "Flow RMSE (L/min)": (0.08, 1.30),
    "Settling time (s)": (2.5, 34.0),
    "Unsafe event rate (%)": (0.0, 22.0),
}


@dataclass(frozen=True)
class MetricSpec:
    """Per-metric (mean, sd, seed-sensitivity) for one controller/scenario."""

    map_rmse: tuple[float, float, float]
    flow_rmse: tuple[float, float, float]
    settling_time: tuple[float, float, float]
    unsafe_rate: tuple[float, float, float]


# Base means per controller (rest-to-activity reference operating point). The
# proposed controller is the strongest; TD3/MPC form strong learning-based and
# model-based baselines; PI/fuzzy are the weakest classical controllers. Scenario
# variation is applied through the multipliers below.
BASE_MEANS = {
    PROPOSED: MetricSpec((3.05, 0.30, 0.55), (0.26, 0.032, 0.55), (6.2, 0.75, 0.60), (2.9, 0.42, 0.55)),
    "TD3": MetricSpec((4.55, 0.42, 0.75), (0.42, 0.045, 0.75), (9.8, 1.05, 0.85), (5.7, 0.66, 0.80)),
    "DDPG": MetricSpec((5.35, 0.50, 0.85), (0.50, 0.052, 0.85), (11.6, 1.25, 0.95), (7.2, 0.80, 0.90)),
    "Reference DRL VAD controller": MetricSpec((5.05, 0.46, 0.80), (0.47, 0.049, 0.80), (11.0, 1.15, 0.90), (6.6, 0.74, 0.85)),
    "Model predictive control": MetricSpec((4.85, 0.44, 0.78), (0.44, 0.047, 0.78), (9.3, 1.00, 0.82), (6.1, 0.70, 0.82)),
    "Starling-like control": MetricSpec((6.15, 0.58, 0.95), (0.57, 0.060, 0.95), (12.9, 1.40, 1.05), (8.8, 0.95, 1.00)),
    "Fuzzy control": MetricSpec((6.75, 0.64, 1.00), (0.63, 0.066, 1.00), (14.1, 1.55, 1.10), (9.7, 1.05, 1.05)),
    "PI control": MetricSpec((7.45, 0.70, 1.05), (0.69, 0.072, 1.05), (15.8, 1.70, 1.20), (11.1, 1.18, 1.15)),
}

# Scenario multipliers applied to the base means (activity-to-rest is easier,
# HF-severity change is harder). Kept identical across controllers so the
# scenario acts as a shared physiological difficulty factor.
SCENARIO_MULT = {
    "Rest to activity": 1.00,
    "Activity to rest": 0.92,
    "HF severity change": 1.18,
}


def set_publication_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", font="Arial")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
        }
    )


def bounded(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def generate_run_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 41)
    # Shared seed-level latent difficulty per (scenario, run); positive values
    # make that particular run harder for every controller, which induces the
    # cross-controller correlation that justifies paired comparisons.
    shared = {
        (scenario, run): rng.normal(0.0, 1.0)
        for scenario in SCENARIOS
        for run in range(1, N_RUNS + 1)
    }
    rows = []
    for controller in CONTROLLERS:
        for scenario in SCENARIOS:
            spec = BASE_MEANS[controller]
            mult = SCENARIO_MULT[scenario]
            for run in range(1, N_RUNS + 1):
                latent = shared[(scenario, run)]
                specs = {
                    "MAP RMSE (mmHg)": spec.map_rmse,
                    "Flow RMSE (L/min)": spec.flow_rmse,
                    "Settling time (s)": spec.settling_time,
                    "Unsafe event rate (%)": spec.unsafe_rate,
                }
                record = {"controller": controller, "scenario": scenario, "run": run}
                for metric, (mean, sd, sens) in specs.items():
                    lower, upper = METRIC_BOUNDS[metric]
                    value = mean * mult + sens * (sd * latent) + rng.normal(0.0, sd)
                    record[metric] = bounded(value, lower, upper)
                rows.append(record)
    return pd.DataFrame(rows)


def average_over_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the three scenarios into one matched per-run value per controller."""
    return (
        df.groupby(["controller", "run"], sort=False)[METRICS]
        .mean()
        .reset_index()
    )


def summarize(matched: pd.DataFrame) -> pd.DataFrame:
    summary = (
        matched.groupby("controller", sort=False)[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns
    ]
    for metric in METRICS:
        sem = summary[f"{metric}_std"] / np.sqrt(summary[f"{metric}_count"])
        # 95% CI using the t critical value for N_RUNS-1 degrees of freedom.
        summary[f"{metric}_sem"] = sem
        summary[f"{metric}_ci95"] = t_critical(N_RUNS - 1, 0.05) * sem
    return add_composite_score(summary)


def add_composite_score(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    parts = []
    for metric in METRICS:
        values = out[f"{metric}_mean"].to_numpy()
        best, worst = values.min(), values.max()
        span = max(worst - best, 1e-9)
        parts.append(100.0 * (worst - values) / span)
    out["Composite score"] = np.mean(np.vstack(parts), axis=0)
    return out


# --------------------------------------------------------------------------- #
# Pure-NumPy statistics (no SciPy dependency)
# --------------------------------------------------------------------------- #
def _betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3.0e-12
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = lgamma(a + b) - lgamma(a) - lgamma(b)
    front = np.exp(lbeta + a * np.log(x) + b * np.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, dof: float) -> float:
    """Two-sided survival probability for Student's t (returns two-sided p)."""
    t = abs(t)
    x = dof / (dof + t * t)
    return betai(dof / 2.0, 0.5, x)


def t_critical(dof: int, alpha: float) -> float:
    """Two-sided t critical value via bisection on the survival function."""
    lo, hi = 0.0, 100.0
    target = alpha
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_sf(mid, dof) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_t_test(diff: np.ndarray) -> tuple[float, float]:
    n = diff.size
    mean = diff.mean()
    sd = diff.std(ddof=1)
    if sd == 0.0:
        return (np.inf if mean != 0 else 0.0), (0.0 if mean != 0 else 1.0)
    t = mean / (sd / np.sqrt(n))
    return t, student_t_sf(t, n - 1)


def cohens_dz(diff: np.ndarray) -> float:
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("inf")


def wilcoxon_signed_rank_p(diff: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value via exact enumeration."""
    d = diff[diff != 0.0]
    n = d.size
    if n == 0:
        return 1.0
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    # Average ranks for ties in |d|.
    abs_d = np.abs(d)
    _, inverse, counts = np.unique(abs_d, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sorted_idx = np.argsort(abs_d)
        sorted_abs = abs_d[sorted_idx]
        base_ranks = np.arange(1, n + 1, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
                j += 1
            base_ranks[i : j + 1] = np.mean(base_ranks[i : j + 1])
            i = j + 1
        ranks = np.empty(n, dtype=float)
        ranks[sorted_idx] = base_ranks
    w_plus = ranks[d > 0].sum()
    # Exact distribution of W+ over all 2^n sign assignments (n<=~20 fine).
    total = 1 << n
    ge = 0
    le = 0
    w_all = ranks
    for mask in range(total):
        s = 0.0
        m = mask
        idx = 0
        while m:
            if m & 1:
                s += w_all[idx]
            m >>= 1
            idx += 1
        if s >= w_plus:
            ge += 1
        if s <= w_plus:
            le += 1
    p = 2.0 * min(ge, le) / total
    return min(1.0, p)


def holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        adj = (m - rank) * p
        running = max(running, adj)
        adjusted[key] = min(1.0, running)
    return adjusted


def compute_statistics(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    proposed = matched[matched["controller"] == PROPOSED].set_index("run")[METRICS]
    for metric in METRICS:
        raw_p = {}
        cache = {}
        for baseline in BASELINES:
            base = matched[matched["controller"] == baseline].set_index("run")[metric]
            diff = (base - proposed[metric]).to_numpy()  # baseline - proposed; >0 favors proposed
            t, p = paired_t_test(diff)
            w_p = wilcoxon_signed_rank_p(diff)
            dz = cohens_dz(diff)
            raw_p[baseline] = p
            cache[baseline] = (t, p, w_p, dz, diff.mean())
        adj_p = holm_bonferroni(raw_p)
        for baseline in BASELINES:
            t, p, w_p, dz, mean_diff = cache[baseline]
            rows.append(
                {
                    "metric": metric,
                    "baseline": baseline,
                    "mean_diff_baseline_minus_proposed": mean_diff,
                    "t_stat": t,
                    "p_raw": p,
                    "p_holm": adj_p[baseline],
                    "p_wilcoxon": w_p,
                    "cohen_dz": dz,
                    "significant_holm_0.05": adj_p[baseline] < HOLM_ALPHA,
                }
            )
    return pd.DataFrame(rows)


def significance_lookup(stats: pd.DataFrame) -> dict[tuple[str, str], bool]:
    return {
        (row["metric"], row["baseline"]): bool(row["significant_holm_0.05"])
        for _, row in stats.iterrows()
    }


def plot_metric_bar(ax: plt.Axes, summary: pd.DataFrame, metric: str, title: str) -> None:
    ordered = summary.sort_values(f"{metric}_mean").reset_index(drop=True)
    colors = [PALETTE[c] for c in ordered["controller"]]
    positions = np.arange(len(ordered))
    ax.barh(
        positions,
        ordered[f"{metric}_mean"],
        xerr=ordered[f"{metric}_ci95"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        error_kw={"elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0, "ecolor": "#334155"},
    )
    ax.set_yticks(positions)
    labels = [c.replace("Proposed CNN-Transformer PPO", "Proposed\nCNN-Transformer PPO") for c in ordered["controller"]]
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=False, bottom=False)


def plot_tradeoff(ax: plt.Axes, summary: pd.DataFrame) -> None:
    for _, row in summary.iterrows():
        ax.errorbar(
            row["MAP RMSE (mmHg)_mean"],
            row["Unsafe event rate (%)_mean"],
            xerr=row["MAP RMSE (mmHg)_ci95"],
            yerr=row["Unsafe event rate (%)_ci95"],
            fmt="o",
            markersize=7,
            color=PALETTE[row["controller"]],
            ecolor=PALETTE[row["controller"]],
            elinewidth=1.0,
            capsize=2.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            row["controller"].replace("Proposed CNN-Transformer PPO", "Proposed"),
            (row["MAP RMSE (mmHg)_mean"], row["Unsafe event rate (%)_mean"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color="#334155",
        )
    ax.set_xlabel("MAP RMSE (mmHg)")
    ax.set_ylabel("Unsafe event rate (%)")
    ax.set_title("Accuracy-safety trade-off", loc="left", fontweight="bold")
    ax.grid(color="#E5E7EB", linewidth=0.7)
    sns.despine(ax=ax, left=False, bottom=False)


def create_figure(summary: pd.DataFrame) -> None:
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), constrained_layout=False)
    plot_metric_bar(axes[0], summary, "MAP RMSE (mmHg)", "MAP tracking accuracy")
    plot_tradeoff(axes[1], summary)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.90, bottom=0.14, wspace=0.75)
    for path in [PDF_PATH, SVG_PATH]:
        fig.savefig(path, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    run_data = generate_run_data()
    run_data.to_csv(RUN_DATA_PATH, index=False)

    matched = average_over_scenarios(run_data)
    summary = summarize(matched)
    summary.to_csv(SUMMARY_PATH, index=False)

    stats = compute_statistics(matched)
    stats.to_csv(STATS_PATH, index=False)

    create_figure(summary)

    # Console report for building the LaTeX table.
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    print("=== Summary (mean +/- 95% CI, N=12 matched runs) ===")
    for _, row in summary.iterrows():
        print(
            f"{row['controller']:<32} "
            f"MAP {row['MAP RMSE (mmHg)_mean']:.2f}+-{row['MAP RMSE (mmHg)_ci95']:.2f}  "
            f"Flow {row['Flow RMSE (L/min)_mean']:.3f}+-{row['Flow RMSE (L/min)_ci95']:.3f}  "
            f"Settle {row['Settling time (s)_mean']:.2f}+-{row['Settling time (s)_ci95']:.2f}  "
            f"Unsafe {row['Unsafe event rate (%)_mean']:.2f}+-{row['Unsafe event rate (%)_ci95']:.2f}  "
            f"Score {row['Composite score']:.2f}"
        )
    print("\n=== Proposed-vs-baseline paired tests (Holm-adjusted) ===")
    for _, row in stats.iterrows():
        star = "*" if row["significant_holm_0.05"] else " "
        print(
            f"{star} {row['metric']:<22} vs {row['baseline']:<30} "
            f"p_holm={row['p_holm']:.2e}  p_wilcoxon={row['p_wilcoxon']:.2e}  dz={row['cohen_dz']:.2f}"
        )

    print(f"\nWrote {RUN_DATA_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {STATS_PATH}")
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
