#!/usr/bin/env python3
"""Generate comparative control performance data and figure.

The script creates reproducible, physiologically constrained virtual data for
VAD transition-control writing practice and renders publication-quality figures.
The synthetic traces deliberately include noise, lag, mild oscillation, and
trade-offs; replace them with measured experimental data for final reporting.
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
TRACE_PATH = FIG_DIR / "comparative-control-transition-trace.csv"
RUN_DATA_PATH = FIG_DIR / "comparative-control-performance-virtual-data.csv"
SUMMARY_PATH = FIG_DIR / "comparative-control-performance-summary.csv"
PDF_PATH = FIG_DIR / "comparative-control-performance-transitions.pdf"
SVG_PATH = FIG_DIR / "comparative-control-performance-transitions.svg"
PNG_PATH = FIG_DIR / "comparative-control-performance-transitions.png"

RNG_SEED = 20260524
N_RUNS = 12
TARGET_MAP = 80.0
MAP_TARGET_BAND = (75.0, 85.0)
FLOW_RANGE = (3.0, 6.2)
SPEED_RANGE = (7.6, 9.8)

CONTROLLERS = [
    "Proposed CNN-Transformer",
    "Existing DRL",
    "Rule-based",
    "Fixed-speed",
]
PALETTE = {
    "Proposed CNN-Transformer": "#0B8A5A",
    "Existing DRL": "#2563EB",
    "Rule-based": "#64748B",
    "Fixed-speed": "#9333EA",
}
SCENARIOS = [
    "Rest to activity",
    "Activity to rest",
    "HF severity change",
]
SHORT_SCENARIOS = {
    "Rest to activity": "Rest to\nactivity",
    "Activity to rest": "Activity to\nrest",
    "HF severity change": "HF severity\nchange",
}
METRICS = [
    "MAP RMSE (mmHg)",
    "Flow RMSE (L/min)",
    "Settling time (s)",
    "Speed variation (krpm)",
]
SHORT_METRICS = {
    "MAP RMSE (mmHg)": "MAP\nRMSE",
    "Flow RMSE (L/min)": "Flow\nRMSE",
    "Settling time (s)": "Settling\ntime",
    "Speed variation (krpm)": "Speed\nvariation",
}


@dataclass(frozen=True)
class MetricSpec:
    map_rmse: tuple[float, float]
    flow_rmse: tuple[float, float]
    settling_time: tuple[float, float]
    speed_variation: tuple[float, float]


VIRTUAL_METRIC_MEANS = {
    "Proposed CNN-Transformer": {
        "Rest to activity": MetricSpec((2.8, 0.38), (0.24, 0.04), (5.8, 0.9), (0.31, 0.05)),
        "Activity to rest": MetricSpec((2.6, 0.34), (0.23, 0.04), (5.2, 0.8), (0.28, 0.04)),
        "HF severity change": MetricSpec((3.5, 0.45), (0.31, 0.05), (7.4, 1.1), (0.36, 0.06)),
    },
    "Existing DRL": {
        "Rest to activity": MetricSpec((4.1, 0.55), (0.35, 0.06), (8.8, 1.3), (0.27, 0.04)),
        "Activity to rest": MetricSpec((3.8, 0.50), (0.33, 0.05), (8.0, 1.2), (0.25, 0.04)),
        "HF severity change": MetricSpec((5.2, 0.68), (0.47, 0.07), (11.2, 1.7), (0.33, 0.05)),
    },
    "Rule-based": {
        "Rest to activity": MetricSpec((5.6, 0.72), (0.52, 0.08), (12.5, 1.9), (0.22, 0.04)),
        "Activity to rest": MetricSpec((5.0, 0.64), (0.46, 0.07), (11.1, 1.7), (0.20, 0.03)),
        "HF severity change": MetricSpec((6.9, 0.86), (0.64, 0.09), (15.8, 2.2), (0.29, 0.05)),
    },
    "Fixed-speed": {
        "Rest to activity": MetricSpec((7.4, 0.90), (0.72, 0.10), (21.5, 2.6), (0.04, 0.01)),
        "Activity to rest": MetricSpec((6.6, 0.78), (0.65, 0.09), (19.4, 2.4), (0.04, 0.01)),
        "HF severity change": MetricSpec((9.1, 1.05), (0.84, 0.12), (25.2, 3.0), (0.04, 0.01)),
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


def colored_noise(rng: np.random.Generator, size: int, scale: float, alpha: float = 0.72) -> np.ndarray:
    raw = rng.normal(0.0, scale, size)
    out = np.zeros(size, dtype=float)
    for idx in range(1, size):
        out[idx] = alpha * out[idx - 1] + (1.0 - alpha) * raw[idx]
    return out


def demand_profile(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, str]]]:
    events = [(30, 70, "activity"), (90, 120, "recovery")]
    demand = np.zeros_like(t, dtype=float)
    demand += 1.0 * (t >= 30)
    demand -= 0.65 * (t >= 90)
    demand += 0.25 * np.exp(-0.5 * ((t - 118) / 9.0) ** 2)
    target_flow = 4.2 + 1.0 * (t >= 30) - 0.55 * (t >= 90)
    return demand, target_flow, events


def first_order_response(
    t: np.ndarray,
    signal: np.ndarray,
    tau: float,
    gain: float,
    baseline: float = 0.0,
) -> np.ndarray:
    out = np.zeros_like(t, dtype=float)
    out[0] = baseline + gain * signal[0]
    dt = float(t[1] - t[0])
    alpha = np.exp(-dt / tau)
    for idx in range(1, len(t)):
        target = baseline + gain * signal[idx]
        out[idx] = alpha * out[idx - 1] + (1.0 - alpha) * target
    return out


def generate_transition_trace() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 11)
    t = np.linspace(0, 140, 281)
    demand, target_flow, _ = demand_profile(t)
    controller_specs = {
        "Proposed CNN-Transformer": {
            "tau": 5.4,
            "map_err": 2.5,
            "flow_lag": 0.13,
            "speed_gain": 1.16,
            "ripple": 0.26,
            "overshoot": 1.15,
            "noise": 0.36,
        },
        "Existing DRL": {
            "tau": 8.8,
            "map_err": 4.1,
            "flow_lag": 0.26,
            "speed_gain": 0.94,
            "ripple": 0.42,
            "overshoot": 1.55,
            "noise": 0.50,
        },
        "Rule-based": {
            "tau": 13.5,
            "map_err": 5.9,
            "flow_lag": 0.44,
            "speed_gain": 0.70,
            "ripple": 0.38,
            "overshoot": 0.85,
            "noise": 0.42,
        },
        "Fixed-speed": {
            "tau": 36.0,
            "map_err": 8.3,
            "flow_lag": 0.82,
            "speed_gain": 0.00,
            "ripple": 0.10,
            "overshoot": 0.20,
            "noise": 0.28,
        },
    }
    rows = []
    for controller, spec in controller_specs.items():
        adaptive = first_order_response(t, demand, spec["tau"], 1.0)
        residual = demand - adaptive
        map_values = TARGET_MAP - spec["map_err"] * residual
        map_values += spec["overshoot"] * np.exp(-0.5 * ((t - 34) / 4.5) ** 2)
        map_values -= 0.65 * spec["overshoot"] * np.exp(-0.5 * ((t - 93) / 5.2) ** 2)
        map_values += spec["ripple"] * np.sin(2 * np.pi * t / 18.0)
        map_values += 0.35 * spec["ripple"] * np.sin(2 * np.pi * t / 7.0 + 0.4)
        map_values += colored_noise(rng, len(t), spec["noise"])
        map_values = np.clip(map_values, 65.0, 96.0)

        pump_flow = target_flow - spec["flow_lag"] * residual
        pump_flow += 0.04 * np.sin(2 * np.pi * t / 11.0)
        pump_flow += colored_noise(rng, len(t), 0.06 + spec["ripple"] * 0.03)
        pump_flow = np.clip(pump_flow, *FLOW_RANGE)

        pump_speed = 8.2 + spec["speed_gain"] * adaptive
        pump_speed += 0.08 * np.sin(2 * np.pi * t / 32.0) * (controller != "Fixed-speed")
        pump_speed += colored_noise(rng, len(t), 0.035 + spec["ripple"] * 0.025)
        pump_speed = np.clip(pump_speed, *SPEED_RANGE)

        for time, demand_value, target_flow_value, map_value, flow_value, speed_value in zip(
            t, demand, target_flow, map_values, pump_flow, pump_speed
        ):
            rows.append(
                {
                    "time_s": time,
                    "controller": controller,
                    "relative_demand": demand_value,
                    "target_flow_L_min": target_flow_value,
                    "MAP_mmHg": map_value,
                    "Pump_flow_L_min": flow_value,
                    "Pump_speed_krpm": speed_value,
                }
            )
    return pd.DataFrame(rows)


def generate_run_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for controller in CONTROLLERS:
        for scenario in SCENARIOS:
            spec = VIRTUAL_METRIC_MEANS[controller][scenario]
            for run in range(1, N_RUNS + 1):
                run_shift = rng.normal(0.0, 0.12)
                rows.append(
                    {
                        "controller": controller,
                        "scenario": scenario,
                        "run": run,
                        "MAP RMSE (mmHg)": bounded_sample(rng, spec.map_rmse[0] + run_shift, spec.map_rmse[1], 1.5, 12.5),
                        "Flow RMSE (L/min)": bounded_sample(rng, spec.flow_rmse[0] + run_shift * 0.04, spec.flow_rmse[1], 0.08, 1.20),
                        "Settling time (s)": bounded_sample(
                            rng, spec.settling_time[0] + run_shift * 1.8, spec.settling_time[1], 2.5, 32.0
                        ),
                        "Speed variation (krpm)": bounded_sample(
                            rng, spec.speed_variation[0] + run_shift * 0.015, spec.speed_variation[1], 0.02, 0.80
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_run_data(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["controller", "scenario"], sort=False)[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns
    ]
    for metric in METRICS:
        summary[f"{metric}_sem"] = summary[f"{metric}_std"] / np.sqrt(summary[f"{metric}_count"])
        summary[f"{metric}_ci95"] = 1.96 * summary[f"{metric}_sem"]
    return summary


def add_transition_scores(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    metric_columns = [f"{metric}_mean" for metric in METRICS]
    normalized_parts = []
    for column in metric_columns:
        values = out[column].to_numpy()
        best = values.min()
        worst = values.max()
        span = max(worst - best, 1e-9)
        normalized_parts.append(100.0 * (worst - values) / span)
    out["Transition score"] = np.mean(np.vstack(normalized_parts), axis=0)
    return out


def mark_transition_windows(ax: plt.Axes, ylim_top: float) -> None:
    for start, end, label in [(30, 70, "activity"), (90, 120, "recovery")]:
        ax.axvspan(start, end, color="#CBD5E1", alpha=0.34, linewidth=0)
        ax.text(
            (start + end) / 2,
            ylim_top,
            label,
            ha="center",
            va="center",
            fontsize=7,
            color="#475569",
        )


def plot_map_response(ax: plt.Axes, trace: pd.DataFrame) -> None:
    ax.axhspan(*MAP_TARGET_BAND, color="#0B8A5A", alpha=0.10, linewidth=0)
    mark_transition_windows(ax, 92.8)
    for controller in CONTROLLERS:
        sub = trace[trace["controller"] == controller]
        ax.plot(sub["time_s"], sub["MAP_mmHg"], color=PALETTE[controller], linewidth=1.7)
    ax.axhline(TARGET_MAP, color="#111827", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.set_xlim(0, 140)
    ax.set_ylim(68, 94)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MAP (mmHg)")
    ax.set_title(
        "MAP regulation during rest-activity-recovery transition",
        loc="center",
        fontsize=10,
        fontweight="bold",
        pad=16,
    )
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=False, bottom=False)


def plot_flow_response(ax: plt.Axes, trace: pd.DataFrame) -> None:
    mark_transition_windows(ax, 5.75)
    target = trace[trace["controller"] == CONTROLLERS[0]]
    ax.plot(
        target["time_s"],
        target["target_flow_L_min"],
        color="#111827",
        linewidth=1.0,
        linestyle="--",
        alpha=0.75,
        label="Target flow",
    )
    for controller in CONTROLLERS:
        sub = trace[trace["controller"] == controller]
        ax.plot(sub["time_s"], sub["Pump_flow_L_min"], color=PALETTE[controller], linewidth=1.6)
    ax.set_xlim(0, 140)
    ax.set_ylim(3.6, 5.95)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pump flow (L/min)")
    ax.set_title("Flow matching under changing demand", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=False, bottom=False)


def plot_speed_response(ax: plt.Axes, trace: pd.DataFrame) -> None:
    mark_transition_windows(ax, 9.35)
    for controller in CONTROLLERS:
        sub = trace[trace["controller"] == controller]
        ax.plot(sub["time_s"], sub["Pump_speed_krpm"], color=PALETTE[controller], linewidth=1.6)
    ax.set_xlim(0, 140)
    ax.set_ylim(7.95, 9.55)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pump speed (krpm)")
    ax.set_title("Adaptive pump speed command", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=False, bottom=False)


def plot_metric_bars(ax: plt.Axes, summary: pd.DataFrame) -> None:
    rest_activity = summary[summary["scenario"] == "Rest to activity"]
    rows = []
    for _, row in rest_activity.iterrows():
        for metric in METRICS:
            rows.append(
                {
                    "controller": row["controller"],
                    "metric": SHORT_METRICS[metric],
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
    offsets = dict(zip(CONTROLLERS, np.linspace(-0.30, 0.30, len(CONTROLLERS))))
    for _, row in plot_df.iterrows():
        ax.errorbar(
            metric_positions[row["metric"]] + offsets[row["controller"]],
            row["mean"],
            yerr=row["sem"],
            fmt="none",
            ecolor=PALETTE[row["controller"]],
            elinewidth=1.0,
            capsize=2.0,
            capthick=1.0,
            zorder=3,
        )
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.set_title("Performance metrics for rest-to-activity transition", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.legend_.remove()
    sns.despine(ax=ax, left=False, bottom=False)


def plot_transition_score(ax: plt.Axes, summary: pd.DataFrame) -> None:
    plot_df = summary.copy()
    plot_df["scenario_short"] = plot_df["scenario"].map(SHORT_SCENARIOS)
    sns.pointplot(
        data=plot_df,
        x="scenario_short",
        y="Transition score",
        hue="controller",
        hue_order=CONTROLLERS,
        palette=PALETTE,
        markers="o",
        linewidth=1.6,
        errorbar=None,
        ax=ax,
    )
    ax.set_xlabel("Physiological transition")
    ax.set_ylabel("Transition score")
    ax.set_ylim(0, 105)
    ax.set_title("Overall control quality across transition scenarios", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.legend_.remove()
    sns.despine(ax=ax, left=False, bottom=False)


def create_figure(summary: pd.DataFrame, trace: pd.DataFrame) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=False)

    plot_map_response(ax, trace)

    legend_handles = [Patch(facecolor=PALETTE[label], edgecolor="none", label=label) for label in CONTROLLERS]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05),
        columnspacing=1.6,
        handlelength=1.4,
    )
    fig.text(
        0.02,
        0.025,
        "Green band denotes target MAP range; gray windows denote physiological transition intervals.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#64748B",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.25)

    for path in [PDF_PATH, SVG_PATH]:
        fig.savefig(path, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    trace = generate_transition_trace()
    trace.to_csv(TRACE_PATH, index=False)

    run_data = generate_run_data()
    run_data.to_csv(RUN_DATA_PATH, index=False)

    summary = add_transition_scores(summarize_run_data(run_data))
    summary.to_csv(SUMMARY_PATH, index=False)

    create_figure(summary, trace)

    print(f"Wrote {TRACE_PATH}")
    print(f"Wrote {RUN_DATA_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
