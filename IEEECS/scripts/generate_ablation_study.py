#!/usr/bin/env python3
"""Generate synthetic ablation data and publication-quality figures.

This script creates reproducible, physiologically constrained virtual data for
the CNN-Transformer ablation study and renders multi-panel scientific figures.
The synthetic results preserve realistic trade-offs: the full model improves
adaptation and safety, but simpler variants may have lower control effort.
Replace the synthetic data with measured experimental results for final use.
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
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_PATH = FIG_DIR / "ablation-study-virtual-data.csv"
SUMMARY_PATH = FIG_DIR / "ablation-study-summary.csv"
PDF_PATH = FIG_DIR / "ablation-study-cnn-transformer-results.pdf"
SVG_PATH = FIG_DIR / "ablation-study-cnn-transformer-results.svg"
PNG_PATH = FIG_DIR / "ablation-study-cnn-transformer-results.png"
LOSS_PDF_PATH = FIG_DIR / "ablation-performance-loss-heatmap.pdf"
LOSS_SVG_PATH = FIG_DIR / "ablation-performance-loss-heatmap.svg"
LOSS_PNG_PATH = FIG_DIR / "ablation-performance-loss-heatmap.png"

RNG_SEED = 20260524
N_RUNS = 12

METRICS = [
    "MAP RMSE (mmHg)",
    "Flow RMSE (L/min)",
    "Unsafe Event Rate (%)",
    "Speed Variation (krpm)",
    "Return",
]
SHORT_METRICS = {
    "MAP RMSE (mmHg)": "MAP\nRMSE",
    "Flow RMSE (L/min)": "Flow\nRMSE",
    "Unsafe Event Rate (%)": "Unsafe\nEvents",
    "Speed Variation (krpm)": "Speed\nVariation",
    "Return": "Return",
}
LOWER_IS_BETTER = {
    "MAP RMSE (mmHg)",
    "Flow RMSE (L/min)",
    "Unsafe Event Rate (%)",
    "Speed Variation (krpm)",
}

PALETTE = {
    "Full CNN-Transformer": "#0B8A5A",
    "MLP Actor-Critic": "#64748B",
    "Transformer-only Actor-Critic": "#D97706",
    "CNN-only Actor-Critic": "#CC2936",
    "Reward ablation baseline": "#2563EB",
    "Fixed-speed baseline": "#9333EA",
}
VARIANT_ORDER = list(PALETTE.keys())


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    metrics: dict[str, tuple[float, float]]


VARIANTS = [
    VariantSpec(
        key="full",
        label="Full CNN-Transformer",
        metrics={
            "MAP RMSE (mmHg)": (3.4, 0.42),
            "Flow RMSE (L/min)": (0.33, 0.05),
            "Unsafe Event Rate (%)": (2.1, 0.55),
            "Speed Variation (krpm)": (0.37, 0.06),
            "Return": (89.0, 2.8),
        },
    ),
    VariantSpec(
        key="mlp_ac",
        label="MLP Actor-Critic",
        metrics={
            "MAP RMSE (mmHg)": (6.8, 0.78),
            "Flow RMSE (L/min)": (0.68, 0.09),
            "Unsafe Event Rate (%)": (10.5, 1.35),
            "Speed Variation (krpm)": (0.20, 0.04),
            "Return": (61.0, 4.2),
        },
    ),
    VariantSpec(
        key="transformer_only",
        label="Transformer-only Actor-Critic",
        metrics={
            "MAP RMSE (mmHg)": (4.9, 0.56),
            "Flow RMSE (L/min)": (0.49, 0.07),
            "Unsafe Event Rate (%)": (5.4, 0.95),
            "Speed Variation (krpm)": (0.29, 0.05),
            "Return": (76.0, 3.5),
        },
    ),
    VariantSpec(
        key="cnn_only",
        label="CNN-only Actor-Critic",
        metrics={
            "MAP RMSE (mmHg)": (5.6, 0.66),
            "Flow RMSE (L/min)": (0.56, 0.08),
            "Unsafe Event Rate (%)": (6.8, 1.15),
            "Speed Variation (krpm)": (0.25, 0.05),
            "Return": (71.0, 3.8),
        },
    ),
    VariantSpec(
        key="reward_ablation",
        label="Reward ablation baseline",
        metrics={
            "MAP RMSE (mmHg)": (4.1, 0.52),
            "Flow RMSE (L/min)": (0.40, 0.06),
            "Unsafe Event Rate (%)": (9.2, 1.25),
            "Speed Variation (krpm)": (0.62, 0.09),
            "Return": (73.0, 3.7),
        },
    ),
    VariantSpec(
        key="fixed_speed",
        label="Fixed-speed baseline",
        metrics={
            "MAP RMSE (mmHg)": (8.6, 0.95),
            "Flow RMSE (L/min)": (0.82, 0.11),
            "Unsafe Event Rate (%)": (11.8, 1.50),
            "Speed Variation (krpm)": (0.10, 0.02),
            "Return": (52.0, 4.5),
        },
    ),
]


def bounded_sample(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    lower: float,
    upper: float,
) -> float:
    return float(np.clip(rng.normal(mean, sd), lower, upper))


def generate_virtual_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    bounds = {
        "MAP RMSE (mmHg)": (1.8, 10.0),
        "Flow RMSE (L/min)": (0.10, 1.10),
        "Unsafe Event Rate (%)": (0.0, 16.0),
        "Speed Variation (krpm)": (0.08, 0.75),
        "Return": (45.0, 98.0),
    }
    rows = []
    for variant in VARIANTS:
        for run in range(1, N_RUNS + 1):
            row = {"variant": variant.key, "label": variant.label, "run": run}
            run_shift = rng.normal(0.0, 0.16)
            for metric, (mean, sd) in variant.metrics.items():
                lower, upper = bounds[metric]
                metric_shift = run_shift
                if metric == "Flow RMSE (L/min)":
                    metric_shift *= 0.05
                elif metric == "Unsafe Event Rate (%)":
                    metric_shift *= 0.80
                elif metric == "Speed Variation (krpm)":
                    metric_shift *= 0.025
                elif metric == "Return":
                    metric_shift *= -3.0
                row[metric] = bounded_sample(rng, mean + metric_shift, sd, lower, upper)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=["variant", "label", "run"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    summary = (
        long_df.groupby(["variant", "label", "metric"], sort=False)
        .agg(mean=("value", "mean"), std=("value", "std"), n=("value", "count"))
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["sem"]
    return summary


def add_normalized_scores(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    scores = []
    for metric in METRICS:
        sub = out[out["metric"] == metric]
        values = sub["mean"].to_numpy()
        best = values.min() if metric in LOWER_IS_BETTER else values.max()
        worst = values.max() if metric in LOWER_IS_BETTER else values.min()
        span = max(abs(worst - best), 1e-9)
        for value in values:
            if metric in LOWER_IS_BETTER:
                score = 100.0 * (worst - value) / span
            else:
                score = 100.0 * (value - worst) / span
            scores.append(np.clip(score, 0.0, 100.0))
    out["normalized_score"] = scores
    return out


def compute_degradation(summary: pd.DataFrame) -> pd.DataFrame:
    means = summary.pivot(index="label", columns="metric", values="mean")
    full = means.loc["Full CNN-Transformer"]
    rows = []
    for label in VARIANT_ORDER[1:]:
        row = {"label": label}
        for metric in METRICS:
            value = means.loc[label, metric]
            if metric in LOWER_IS_BETTER:
                degradation = (value - full[metric]) / full[metric] * 100.0
            else:
                degradation = (full[metric] - value) / full[metric] * 100.0
            row[metric] = degradation
        rows.append(row)
    return pd.DataFrame(rows).set_index("label")


def compute_radar_profiles(summary: pd.DataFrame) -> pd.DataFrame:
    score = summary.pivot(index="label", columns="metric", values="normalized_score")
    profiles = pd.DataFrame(
        {
            "Local morphology": score["Flow RMSE (L/min)"],
            "Temporal adaptation": (score["MAP RMSE (mmHg)"] + score["Return"]) / 2,
            "Safety": score["Unsafe Event Rate (%)"],
            "Smoothness": score["Speed Variation (krpm)"],
            "Overall return": score["Return"],
        }
    )
    return profiles.loc[["Full CNN-Transformer", "Transformer-only Actor-Critic", "CNN-only Actor-Critic", "Reward ablation baseline"]]


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


def plot_normalized_scores(ax: plt.Axes, summary: pd.DataFrame) -> None:
    plot_df = summary.copy()
    plot_df["metric_short"] = plot_df["metric"].map(SHORT_METRICS)
    sns.barplot(
        data=plot_df,
        x="metric_short",
        y="normalized_score",
        hue="label",
        hue_order=VARIANT_ORDER,
        palette=PALETTE,
        ax=ax,
        errorbar=None,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_ylim(0, 108)
    ax.set_xlabel("")
    ax.set_ylabel("Normalized score (higher is better)")
    ax.set_title("Normalized control quality across ablations", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.legend_.remove()
    sns.despine(ax=ax, left=False, bottom=False)


def plot_degradation_heatmap(ax: plt.Axes, degradation: pd.DataFrame) -> None:
    heat_df = degradation.rename(columns=SHORT_METRICS)
    annotations = heat_df.apply(lambda col: col.map(lambda x: f"{x:+.0f}%"))
    sns.heatmap(
        heat_df,
        ax=ax,
        cmap="vlag",
        center=0,
        annot=annotations,
        fmt="",
        cbar_kws={"label": "Relative change vs. full model (%)", "shrink": 0.80},
        linewidths=0.8,
        linecolor="white",
        vmin=min(heat_df.min().min(), -60),
        vmax=max(heat_df.max().max(), 120),
        annot_kws={"fontsize": 7.5, "fontweight": "bold"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Performance trade-off relative to full model", loc="center", fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


def plot_radar(ax: plt.Axes, profiles: pd.DataFrame) -> None:
    categories = list(profiles.columns)
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], color="#64748B", fontsize=7)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7.5)
    ax.grid(color="#E5E7EB", linewidth=0.7)
    ax.spines["polar"].set_color("#CBD5E1")

    for label in profiles.index:
        values = profiles.loc[label].to_list()
        values += values[:1]
        ax.plot(angles, values, color=PALETTE[label], linewidth=1.7, label=label)
        ax.fill(angles, values, color=PALETTE[label], alpha=0.10)
    ax.set_title("Module-level contribution profile", loc="left", pad=18, fontweight="bold")


def plot_raw_metric_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    key_metrics = ["MAP RMSE (mmHg)", "Unsafe Event Rate (%)", "Return"]
    subset = summary[summary["metric"].isin(key_metrics)].copy()
    subset["metric_short"] = subset["metric"].map(
        {
            "MAP RMSE (mmHg)": "MAP RMSE\n(mmHg)",
            "Unsafe Event Rate (%)": "Unsafe event\nrate (%)",
            "Return": "Return",
        }
    )
    sns.pointplot(
        data=subset,
        x="metric_short",
        y="mean",
        hue="label",
        hue_order=VARIANT_ORDER,
        palette=PALETTE,
        dodge=0.45,
        markers="o",
        linestyles="none",
        errorbar=None,
        ax=ax,
    )

    metric_positions = {metric: i for i, metric in enumerate(subset["metric_short"].unique())}
    offsets = dict(zip(VARIANT_ORDER, np.linspace(-0.27, 0.27, len(VARIANT_ORDER))))
    for _, row in subset.iterrows():
        x = metric_positions[row["metric_short"]] + offsets[row["label"]]
        ax.errorbar(
            x,
            row["mean"],
            yerr=row["sem"],
            fmt="none",
            ecolor=PALETTE[row["label"]],
            elinewidth=1.0,
            capsize=2.2,
            capthick=1.0,
            zorder=3,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Mean ± SEM")
    ax.set_title("Representative raw metrics", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.legend_.remove()
    sns.despine(ax=ax, left=False, bottom=False)


def create_figure(summary: pd.DataFrame, degradation: pd.DataFrame, profiles: pd.DataFrame) -> None:
    set_publication_style()
    fig = plt.figure(figsize=(10.8, 6.8), constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.55, 1.10, 1.05], height_ratios=[1.05, 1.0])

    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[0, 2], projection="polar")
    ax_d = fig.add_subplot(gs[1, 1:])

    plot_normalized_scores(ax_a, summary)
    plot_degradation_heatmap(ax_b, degradation)
    plot_radar(ax_c, profiles)
    plot_raw_metric_panel(ax_d, summary)

    panel_label(ax_a, "A")
    panel_label(ax_b, "B")
    ax_c.text(-0.20, 1.12, "C", transform=ax_c.transAxes, fontsize=11, fontweight="bold")
    panel_label(ax_d, "D")

    legend_handles = [Patch(facecolor=PALETTE[label], edgecolor="none", label=label) for label in VARIANT_ORDER]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        columnspacing=1.6,
        handlelength=1.4,
    )
    fig.suptitle(
        "Ablation study of representation modules and safety reward terms",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.13, wspace=0.42, hspace=0.46)

    for path in [PDF_PATH, SVG_PATH]:
        fig.savefig(path, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)


def create_standalone_degradation_figure(degradation: pd.DataFrame) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=False)

    plot_degradation_heatmap(ax, degradation)
    ax.set_xlabel("Evaluation metric")
    ax.set_ylabel("Ablated variant")
    fig.subplots_adjust(left=0.16, right=0.96, top=0.88, bottom=0.26)
    fig.text(
        0.02,
        0.07,
        "Values denote relative change compared with the full model; negative values indicate lower control effort.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#64748B",
    )

    for path in [LOSS_PDF_PATH, LOSS_SVG_PATH]:
        fig.savefig(path, bbox_inches="tight")
    fig.savefig(LOSS_PNG_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = generate_virtual_data()
    data.to_csv(DATA_PATH, index=False)

    summary = add_normalized_scores(summarize(data))
    summary.to_csv(SUMMARY_PATH, index=False)

    degradation = compute_degradation(summary)
    profiles = compute_radar_profiles(summary)
    create_figure(summary, degradation, profiles)
    create_standalone_degradation_figure(degradation)

    print(f"Wrote {DATA_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {LOSS_PDF_PATH}")
    print(f"Wrote {LOSS_SVG_PATH}")
    print(f"Wrote {LOSS_PNG_PATH}")


if __name__ == "__main__":
    main()
