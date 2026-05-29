#!/usr/bin/env python3
"""Generate robustness and safety analysis data and figures.

The script creates reproducible, physiologically constrained virtual data for
disturbance tests of VAD physiological control and renders publication-quality
figures. The synthetic data include noise, lag, recovery transients, and
controller trade-offs; replace them with measured results for final reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RUN_DATA_PATH = FIG_DIR / "robustness-safety-virtual-data.csv"
SUMMARY_PATH = FIG_DIR / "robustness-safety-summary.csv"
TRACE_PATH = FIG_DIR / "robustness-safety-composite-trace.csv"
PDF_PATH = FIG_DIR / "robustness-safety-under-disturbances.pdf"
SVG_PATH = FIG_DIR / "robustness-safety-under-disturbances.svg"
PNG_PATH = FIG_DIR / "robustness-safety-under-disturbances.png"
MAP_PDF_PATH = FIG_DIR / "map-response-composite-disturbances.pdf"
MAP_SVG_PATH = FIG_DIR / "map-response-composite-disturbances.svg"
MAP_PNG_PATH = FIG_DIR / "map-response-composite-disturbances.png"

RNG_SEED = 20260524
N_RUNS = 12
TARGET_MAP = 80.0
MAP_TARGET_BAND = (75.0, 85.0)
LVP_SUCTION_THRESHOLD = 5.0
FLOW_RANGE = (-0.4, 6.4)
LVP_RANGE = (1.2, 12.5)

CONTROLLERS = [
    "Proposed CNN-Transformer",
    "Existing DRL",
    "Rule-based",
]
PALETTE = {
    "Proposed CNN-Transformer": "#0B8A5A",
    "Existing DRL": "#2563EB",
    "Rule-based": "#64748B",
}
DISTURBANCES = [
    "Nominal",
    "Sensor noise",
    "Parameter drift",
    "Preload drop",
    "Afterload rise",
    "Composite",
]
SHORT_DISTURBANCES = {
    "Nominal": "Nominal",
    "Sensor noise": "Noise",
    "Parameter drift": "Param.\ndrift",
    "Preload drop": "Preload\ndrop",
    "Afterload rise": "Afterload\nrise",
    "Composite": "Composite",
}
SAFETY_METRICS = {
    "Unsafe event rate (%)": "Unsafe\nevents",
    "Suction time (%)": "Suction\ntime",
    "Backflow duration (%)": "Backflow\nduration",
}
ROBUSTNESS_METRICS = [
    "MAP RMSE (mmHg)",
    "Unsafe event rate (%)",
    "Suction time (%)",
    "Backflow duration (%)",
    "Speed variation (krpm)",
    "Recovery time (s)",
]


@dataclass(frozen=True)
class MetricSpec:
    map_rmse: tuple[float, float]
    unsafe_event_rate: tuple[float, float]
    suction_time: tuple[float, float]
    backflow_duration: tuple[float, float]
    speed_variation: tuple[float, float]
    recovery_time: tuple[float, float]


VIRTUAL_METRIC_MEANS = {
    "Proposed CNN-Transformer": {
        "Nominal": MetricSpec((2.7, 0.35), (0.6, 0.20), (0.2, 0.10), (0.2, 0.10), (0.24, 0.04), (4.8, 0.8)),
        "Sensor noise": MetricSpec((3.3, 0.42), (1.0, 0.28), (0.4, 0.15), (0.4, 0.15), (0.27, 0.04), (6.0, 0.9)),
        "Parameter drift": MetricSpec((3.9, 0.48), (1.4, 0.35), (0.7, 0.20), (0.5, 0.18), (0.31, 0.05), (7.3, 1.0)),
        "Preload drop": MetricSpec((4.4, 0.55), (2.0, 0.45), (1.3, 0.30), (0.7, 0.20), (0.36, 0.06), (8.5, 1.2)),
        "Afterload rise": MetricSpec((4.1, 0.52), (1.7, 0.40), (0.8, 0.22), (1.0, 0.28), (0.34, 0.06), (8.0, 1.1)),
        "Composite": MetricSpec((5.0, 0.62), (2.8, 0.60), (1.8, 0.38), (1.2, 0.32), (0.42, 0.07), (10.4, 1.4)),
    },
    "Existing DRL": {
        "Nominal": MetricSpec((3.7, 0.45), (1.6, 0.40), (0.9, 0.25), (0.8, 0.22), (0.22, 0.04), (6.8, 1.0)),
        "Sensor noise": MetricSpec((4.8, 0.58), (2.9, 0.60), (1.7, 0.38), (1.4, 0.35), (0.29, 0.05), (9.3, 1.4)),
        "Parameter drift": MetricSpec((5.4, 0.65), (4.1, 0.75), (2.6, 0.52), (2.0, 0.45), (0.33, 0.06), (11.2, 1.7)),
        "Preload drop": MetricSpec((6.4, 0.75), (5.9, 0.95), (4.0, 0.70), (2.7, 0.55), (0.39, 0.07), (13.8, 2.0)),
        "Afterload rise": MetricSpec((6.1, 0.72), (5.2, 0.88), (3.0, 0.62), (3.5, 0.70), (0.38, 0.07), (12.9, 1.9)),
        "Composite": MetricSpec((7.4, 0.88), (7.9, 1.15), (5.3, 0.92), (4.4, 0.85), (0.46, 0.08), (17.4, 2.3)),
    },
    "Rule-based": {
        "Nominal": MetricSpec((5.0, 0.60), (2.9, 0.62), (1.9, 0.42), (1.4, 0.34), (0.26, 0.05), (9.7, 1.5)),
        "Sensor noise": MetricSpec((6.5, 0.78), (5.0, 0.90), (3.3, 0.65), (2.6, 0.58), (0.36, 0.07), (13.8, 2.0)),
        "Parameter drift": MetricSpec((7.5, 0.90), (6.9, 1.15), (4.7, 0.92), (3.8, 0.75), (0.44, 0.08), (16.4, 2.4)),
        "Preload drop": MetricSpec((8.8, 1.02), (10.0, 1.35), (7.4, 1.10), (5.0, 0.95), (0.55, 0.10), (20.5, 2.8)),
        "Afterload rise": MetricSpec((8.4, 0.98), (9.2, 1.30), (5.8, 1.05), (6.4, 1.10), (0.53, 0.10), (19.2, 2.6)),
        "Composite": MetricSpec((10.2, 1.12), (13.6, 1.70), (9.5, 1.35), (7.5, 1.25), (0.66, 0.12), (24.5, 3.2)),
    },
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


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def sample_positive(rng: np.random.Generator, mean: float, sd: float) -> float:
    return max(0.0, rng.normal(mean, sd))


def bounded_sample(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    lower: float,
    upper: float,
) -> float:
    return float(np.clip(rng.normal(mean, sd), lower, upper))


def colored_noise(rng: np.random.Generator, size: int, scale: float, alpha: float = 0.75) -> np.ndarray:
    raw = rng.normal(0.0, scale, size)
    out = np.zeros(size, dtype=float)
    for idx in range(1, size):
        out[idx] = alpha * out[idx - 1] + (1.0 - alpha) * raw[idx]
    return out


def generate_run_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for controller in CONTROLLERS:
        for disturbance in DISTURBANCES:
            spec = VIRTUAL_METRIC_MEANS[controller][disturbance]
            for run in range(1, N_RUNS + 1):
                subject_shift = rng.normal(0.0, 0.15)
                rows.append(
                    {
                        "controller": controller,
                        "disturbance": disturbance,
                        "run": run,
                        "MAP RMSE (mmHg)": bounded_sample(rng, spec.map_rmse[0] + subject_shift, spec.map_rmse[1], 1.8, 13.0),
                        "Unsafe event rate (%)": bounded_sample(
                            rng, spec.unsafe_event_rate[0] + subject_shift * 0.7, spec.unsafe_event_rate[1], 0.0, 18.0
                        ),
                        "Suction time (%)": bounded_sample(
                            rng, spec.suction_time[0] + subject_shift * 0.45, spec.suction_time[1], 0.0, 13.0
                        ),
                        "Backflow duration (%)": bounded_sample(
                            rng, spec.backflow_duration[0] + subject_shift * 0.35, spec.backflow_duration[1], 0.0, 10.0
                        ),
                        "Speed variation (krpm)": bounded_sample(
                            rng, spec.speed_variation[0] + subject_shift * 0.02, spec.speed_variation[1], 0.08, 0.90
                        ),
                        "Recovery time (s)": bounded_sample(
                            rng, spec.recovery_time[0] + subject_shift * 2.0, spec.recovery_time[1], 2.5, 32.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_run_data(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["controller", "disturbance"], sort=False)[ROBUSTNESS_METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns
    ]
    for metric in ROBUSTNESS_METRICS:
        summary[f"{metric}_sem"] = summary[f"{metric}_std"] / np.sqrt(summary[f"{metric}_count"])
        summary[f"{metric}_ci95"] = 1.96 * summary[f"{metric}_sem"]
    return summary


def add_robustness_scores(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    metric_columns = [f"{metric}_mean" for metric in ROBUSTNESS_METRICS]
    raw_score = out[metric_columns].copy()
    normalized_parts = []
    for column in raw_score.columns:
        values = raw_score[column].to_numpy()
        best = values.min()
        worst = values.max()
        span = max(worst - best, 1e-9)
        normalized_parts.append(100.0 * (worst - values) / span)
    out["Robustness score"] = np.mean(np.vstack(normalized_parts), axis=0)
    return out


def disturbance_profile(t: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    events = [(20, 40, "sensor noise"), (52, 72, "preload drop"), (84, 104, "afterload rise")]
    perturb = np.zeros_like(t, dtype=float)
    perturb += -10.0 * ((t >= 52) & (t <= 72))
    perturb += 7.0 * ((t >= 84) & (t <= 104))
    perturb += -4.0 * np.exp(-0.5 * ((t - 112) / 5.5) ** 2)
    return perturb, events


def transient_response(
    t: np.ndarray,
    perturb: np.ndarray,
    gain: float,
    ripple: float,
    rng: np.random.Generator,
) -> np.ndarray:
    response = np.zeros_like(t, dtype=float)
    response[0] = gain * perturb[0]
    alpha = 0.82
    for idx in range(1, len(t)):
        response[idx] = alpha * response[idx - 1] + (1.0 - alpha) * gain * perturb[idx]
    oscillation = ripple * np.sin(2 * np.pi * t / 18.0) * (np.abs(perturb) > 0)
    noise = rng.normal(0.0, ripple * 0.25, size=len(t))
    return TARGET_MAP + response + oscillation + noise


def generate_composite_trace() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 17)
    t = np.linspace(0, 120, 241)
    perturb, _ = disturbance_profile(t)
    controller_specs = {
        "Proposed CNN-Transformer": {"gain": 0.25, "ripple": 0.95, "lvp_drop": 0.36, "flow_drop": 0.22, "noise": 0.50},
        "Existing DRL": {"gain": 0.52, "ripple": 1.85, "lvp_drop": 0.70, "flow_drop": 0.46, "noise": 0.72},
        "Rule-based": {"gain": 0.80, "ripple": 2.55, "lvp_drop": 1.02, "flow_drop": 0.68, "noise": 0.82},
    }
    rows = []
    for controller, spec in controller_specs.items():
        map_values = transient_response(t, perturb, spec["gain"], spec["ripple"], rng)
        recovery_dip = -1.0 * np.exp(-0.5 * ((t - 75) / 5.0) ** 2) - 0.6 * np.exp(-0.5 * ((t - 107) / 4.8) ** 2)
        map_values += recovery_dip * spec["gain"]
        map_values += colored_noise(rng, len(t), spec["noise"])
        map_values = np.clip(map_values, 58.0, 98.0)
        lvp_min = 9.5 - spec["lvp_drop"] * np.maximum(0, -perturb)
        lvp_min -= 0.20 * np.maximum(0, map_values - TARGET_MAP)
        lvp_min += colored_noise(rng, len(t), 0.32 + spec["ripple"] * 0.04)
        lvp_min = np.clip(lvp_min, *LVP_RANGE)
        pump_flow = 4.1 + 0.05 * (map_values - TARGET_MAP) - spec["flow_drop"] * np.maximum(0, perturb) / 10
        pump_flow += 0.06 * np.sin(2 * np.pi * t / 9.5)
        pump_flow += colored_noise(rng, len(t), 0.08 + spec["ripple"] * 0.02)
        pump_flow = np.clip(pump_flow, *FLOW_RANGE)
        for time, map_value, lvp_value, flow_value in zip(t, map_values, lvp_min, pump_flow):
            rows.append(
                {
                    "time_s": time,
                    "controller": controller,
                    "MAP_mmHg": map_value,
                    "LVP_min_mmHg": lvp_value,
                    "Pump_flow_L_min": flow_value,
                }
            )
    return pd.DataFrame(rows)


def plot_map_trace(ax: plt.Axes, trace: pd.DataFrame) -> None:
    _, events = disturbance_profile(trace["time_s"].unique())
    ax.axhspan(*MAP_TARGET_BAND, color="#0B8A5A", alpha=0.10, linewidth=0)
    for start, end, label in events:
        ax.axvspan(start, end, color="#CBD5E1", alpha=0.38, linewidth=0)
        ax.text(
            (start + end) / 2,
            93.5,
            label,
            ha="center",
            va="center",
            fontsize=7,
            color="#475569",
        )
    for controller in CONTROLLERS:
        sub = trace[trace["controller"] == controller]
        ax.plot(
            sub["time_s"],
            sub["MAP_mmHg"],
            color=PALETTE[controller],
            linewidth=1.8,
            label=controller,
        )
    ax.axhline(TARGET_MAP, color="#111827", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.set_xlim(0, 120)
    ax.set_ylim(62, 96)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MAP (mmHg)")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=False, bottom=False)


def plot_safety_envelope(ax: plt.Axes, trace: pd.DataFrame) -> None:
    proposed = trace[trace["controller"] == "Proposed CNN-Transformer"]
    drl = trace[trace["controller"] == "Existing DRL"]
    rule = trace[trace["controller"] == "Rule-based"]
    ax.axhspan(0, LVP_SUCTION_THRESHOLD, color="#CC2936", alpha=0.10, linewidth=0)
    for controller, sub in [
        ("Proposed CNN-Transformer", proposed),
        ("Existing DRL", drl),
        ("Rule-based", rule),
    ]:
        ax.plot(
            sub["time_s"],
            sub["LVP_min_mmHg"],
            color=PALETTE[controller],
            linewidth=1.6,
            label=controller,
        )
    ax.axhline(LVP_SUCTION_THRESHOLD, color="#CC2936", linewidth=1.0, linestyle="--")
    ax.text(3, LVP_SUCTION_THRESHOLD + 0.45, "suction threshold", fontsize=7.5, color="#CC2936")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Minimum LVP (mmHg)")
    ax.set_title("Safety margin against ventricular suction", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=False, bottom=False)


def plot_robustness_heatmap(ax: plt.Axes, summary: pd.DataFrame) -> None:
    heat_df = summary.pivot(index="controller", columns="disturbance", values="Robustness score")
    heat_df = heat_df.loc[CONTROLLERS, DISTURBANCES]
    heat_df = heat_df.rename(columns=SHORT_DISTURBANCES)
    annotations = heat_df.apply(lambda col: col.map(lambda x: f"{x:.0f}"))
    sns.heatmap(
        heat_df,
        ax=ax,
        cmap=sns.light_palette("#0B8A5A", as_cmap=True),
        annot=annotations,
        fmt="",
        cbar_kws={"label": "Robustness score", "shrink": 0.80},
        linewidths=0.8,
        linecolor="white",
        vmin=0,
        vmax=100,
        annot_kws={"fontsize": 7.5, "fontweight": "bold"},
    )
    ax.set_xlabel("Disturbance scenario")
    ax.set_ylabel("")
    ax.set_title("Robustness score across disturbance conditions", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


def plot_safety_bars(ax: plt.Axes, summary: pd.DataFrame) -> None:
    composite = summary[summary["disturbance"] == "Composite"]
    rows = []
    for _, row in composite.iterrows():
        for metric, short_metric in SAFETY_METRICS.items():
            rows.append(
                {
                    "controller": row["controller"],
                    "metric": short_metric,
                    "mean": row[f"{metric}_mean"],
                    "sem": row[f"{metric}_sem"],
                }
            )
    plot_df = pd.DataFrame(rows)
    sns.barplot(
        data=plot_df,
        x="metric",
        y="mean",
        hue="controller",
        hue_order=CONTROLLERS,
        palette=PALETTE,
        ax=ax,
        errorbar=None,
        edgecolor="white",
        linewidth=0.6,
    )
    metric_positions = {metric: i for i, metric in enumerate(plot_df["metric"].unique())}
    offsets = dict(zip(CONTROLLERS, np.linspace(-0.27, 0.27, len(CONTROLLERS))))
    for _, row in plot_df.iterrows():
        ax.errorbar(
            metric_positions[row["metric"]] + offsets[row["controller"]],
            row["mean"],
            yerr=row["sem"],
            fmt="none",
            ecolor=PALETTE[row["controller"]],
            elinewidth=1.0,
            capsize=2.2,
            capthick=1.0,
            zorder=3,
        )
    ax.set_xlabel("")
    ax.set_ylabel("Rate or duration (%)")
    ax.set_title("Safety events under composite disturbance", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.legend_.remove()
    sns.despine(ax=ax, left=False, bottom=False)


def plot_recovery_profile(ax: plt.Axes, summary: pd.DataFrame) -> None:
    plot_df = summary.copy()
    plot_df["disturbance_short"] = plot_df["disturbance"].map(SHORT_DISTURBANCES)
    sns.pointplot(
        data=plot_df,
        x="disturbance_short",
        y="Recovery time (s)_mean",
        hue="controller",
        hue_order=CONTROLLERS,
        palette=PALETTE,
        markers="o",
        linewidth=1.6,
        errorbar=None,
        ax=ax,
    )
    positions = {metric: i for i, metric in enumerate(plot_df["disturbance_short"].unique())}
    offsets = dict(zip(CONTROLLERS, [-0.12, 0.0, 0.12]))
    for _, row in plot_df.iterrows():
        ax.errorbar(
            positions[row["disturbance_short"]] + offsets[row["controller"]],
            row["Recovery time (s)_mean"],
            yerr=row["Recovery time (s)_sem"],
            fmt="none",
            ecolor=PALETTE[row["controller"]],
            elinewidth=1.0,
            capsize=2.0,
            capthick=1.0,
            zorder=3,
        )
    ax.set_xlabel("Disturbance scenario")
    ax.set_ylabel("Recovery time (s)")
    ax.set_title("Recovery speed after disturbance onset", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.legend_.remove()
    sns.despine(ax=ax, left=False, bottom=False)


def create_figure(summary: pd.DataFrame, trace: pd.DataFrame) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=False)

    plot_map_trace(ax, trace)

    legend_handles = [Patch(facecolor=PALETTE[label], edgecolor="none", label=label) for label in CONTROLLERS]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05),
        columnspacing=1.8,
        handlelength=1.4,
    )
    fig.suptitle(
        "MAP response under composite disturbances",
        x=0.5,
        y=0.965,
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.025,
        "Shaded MAP region denotes the target band; gray windows indicate disturbance intervals.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#64748B",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.25)

    for path in [PDF_PATH, SVG_PATH]:
        fig.savefig(path, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)


def create_standalone_map_figure(trace: pd.DataFrame) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=False)

    plot_map_trace(ax, trace)
    ax.set_title("MAP response under composite disturbances", loc="left", fontweight="bold", pad=8)
    ax.legend(
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.36),
        columnspacing=1.4,
        handlelength=1.8,
    )
    fig.text(
        0.11,
        0.02,
        "Green band indicates the target MAP range (75-85 mmHg); gray windows indicate disturbance intervals.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#64748B",
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.30)

    for path in [MAP_PDF_PATH, MAP_SVG_PATH]:
        fig.savefig(path, bbox_inches="tight")
    fig.savefig(MAP_PNG_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    run_data = generate_run_data()
    run_data.to_csv(RUN_DATA_PATH, index=False)

    summary = add_robustness_scores(summarize_run_data(run_data))
    summary.to_csv(SUMMARY_PATH, index=False)

    trace = generate_composite_trace()
    trace.to_csv(TRACE_PATH, index=False)

    create_figure(summary, trace)
    create_standalone_map_figure(trace)

    print(f"Wrote {RUN_DATA_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {TRACE_PATH}")
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {MAP_PDF_PATH}")
    print(f"Wrote {MAP_SVG_PATH}")
    print(f"Wrote {MAP_PNG_PATH}")


if __name__ == "__main__":
    main()
