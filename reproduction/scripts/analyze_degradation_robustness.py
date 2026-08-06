"""Create curves and robustness summaries from a completed degradation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


DEGRADATION_LABELS = {
    "gaussian_noise": "Gaussian noise",
    "gaussian_blur": "Gaussian blur",
    "motion_blur": "Motion blur",
    "jpeg_compression": "JPEG compression",
}
MODEL_ORDER = [
    "AlexNet random",
    "AlexNet pretrained",
    "ResNet18 random",
    "ResNet18 pretrained",
    "ResNet50 random",
    "ResNet50 pretrained",
]
MODEL_STYLE = {
    "AlexNet random": {"color": "#E76F51", "linestyle": "--", "marker": "o"},
    "AlexNet pretrained": {"color": "#E76F51", "linestyle": "-", "marker": "o"},
    "ResNet18 random": {"color": "#3A86FF", "linestyle": "--", "marker": "s"},
    "ResNet18 pretrained": {"color": "#3A86FF", "linestyle": "-", "marker": "s"},
    "ResNet50 random": {"color": "#2A9D8F", "linestyle": "--", "marker": "^"},
    "ResNet50 pretrained": {"color": "#2A9D8F", "linestyle": "-", "marker": "^"},
}
BACKGROUND = "#DCE9F7"
GRID = "#FFFFFF"
TEXT = "#263238"


def add_clean_rows(frame: pd.DataFrame, degradation: str) -> pd.DataFrame:
    clean = frame[frame["degradation"] == "clean"].copy()
    clean["degradation"] = degradation
    clean["severity"] = 0
    clean["parameter_name"] = "clean"
    clean["parameter_value"] = 0.0
    corrupted = frame[frame["degradation"] == degradation].copy()
    return pd.concat([clean, corrupted], ignore_index=True)


def severity_labels(subset: pd.DataFrame) -> list[str]:
    parameter_name = subset[subset["severity"] > 0]["parameter_name"].iloc[0]
    parameter_values = (
        subset[subset["model"] == MODEL_ORDER[0]]
        .sort_values("severity")["parameter_value"]
        .tolist()
    )
    friendly_name = {
        "sigma": r"$\sigma$",
        "kernel_size": "kernel",
        "quality": "quality",
    }[parameter_name]
    return ["Clean"] + [f"S{i}\n{friendly_name}={value:g}" for i, value in enumerate(parameter_values[1:], 1)]


def style_axis(axis, metric: str, panel: str) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.grid(True, color=GRID, linewidth=1.25, alpha=0.95)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(colors=TEXT, labelsize=9, length=0)
    axis.set_xlabel("Degradation severity", color=TEXT, labelpad=9)
    axis.set_ylabel(
        "Top-1 accuracy" if metric == "top1_accuracy" else "Macro-F1",
        color=TEXT,
        labelpad=9,
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_ylim(0, 0.86)
    axis.text(
        0.02,
        0.96,
        panel,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=TEXT,
    )


def draw_metric(axis, subset: pd.DataFrame, metric: str) -> None:
    for model in MODEL_ORDER:
        rows = subset[subset["model"] == model].sort_values("severity")
        style = MODEL_STYLE[model]
        axis.plot(
            rows["severity"],
            rows[metric],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.5 if "pretrained" in model else 1.65,
            markersize=6 if "pretrained" in model else 5,
            markerfacecolor="white" if "random" in model else style["color"],
            markeredgewidth=1.25,
            label=model,
        )


def plot_degradation(frame: pd.DataFrame, degradation: str, output: Path, panel_index: int):
    subset = add_clean_rows(frame, degradation)
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), facecolor="white")
    panels = (f"({chr(97 + panel_index * 2)})", f"({chr(98 + panel_index * 2)})")
    for axis, metric, panel in zip(axes, ("top1_accuracy", "macro_f1"), panels):
        draw_metric(axis, subset, metric)
        style_axis(axis, metric, panel)
        axis.set_xticks(range(5), severity_labels(subset))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        f"Robustness under {DEGRADATION_LABELS[degradation]}",
        fontsize=16,
        fontweight="bold",
        color=TEXT,
        y=0.99,
    )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.22, wspace=0.20)
    figure.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    metadata = json.loads((input_dir / "run_metadata.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(input_dir / "robustness_metrics.csv")
    expected = len(metadata["models"]) * len(metadata["conditions"])
    if metadata.get("status") != "completed" or len(frame) != expected:
        raise RuntimeError(
            f"Robustness run is incomplete: status={metadata.get('status')}, "
            f"rows={len(frame)}/{expected}"
        )
    output_dir.mkdir(parents=True)

    clean = (
        frame[frame["degradation"] == "clean"]
        .set_index("model")[["top1_accuracy", "macro_f1"]]
        .rename(columns=lambda name: f"clean_{name}")
    )
    corrupted = frame[frame["degradation"] != "clean"].copy()
    corrupted = corrupted.join(clean, on="model")
    corrupted["top1_drop"] = corrupted["clean_top1_accuracy"] - corrupted["top1_accuracy"]
    corrupted["macro_f1_drop"] = corrupted["clean_macro_f1"] - corrupted["macro_f1"]
    corrupted["relative_top1_retention"] = np.where(
        corrupted["clean_top1_accuracy"] > 0,
        corrupted["top1_accuracy"] / corrupted["clean_top1_accuracy"],
        0,
    )
    corrupted["relative_f1_retention"] = np.where(
        corrupted["clean_macro_f1"] > 0,
        corrupted["macro_f1"] / corrupted["clean_macro_f1"],
        0,
    )
    corrupted.to_csv(output_dir / "metrics_with_drops.csv", index=False)

    summary = (
        corrupted.groupby("model", sort=False)
        .agg(
            clean_top1_accuracy=("clean_top1_accuracy", "first"),
            clean_macro_f1=("clean_macro_f1", "first"),
            mean_corrupted_top1=("top1_accuracy", "mean"),
            mean_corrupted_macro_f1=("macro_f1", "mean"),
            mean_top1_drop=("top1_drop", "mean"),
            mean_macro_f1_drop=("macro_f1_drop", "mean"),
            mean_relative_top1_retention=("relative_top1_retention", "mean"),
            mean_relative_f1_retention=("relative_f1_retention", "mean"),
        )
        .reset_index()
    )
    summary["absolute_robustness_rank"] = summary["mean_corrupted_top1"].rank(
        ascending=False, method="min"
    ).astype(int)
    summary["relative_robustness_rank"] = summary["mean_relative_top1_retention"].rank(
        ascending=False, method="min"
    ).astype(int)
    summary.sort_values("absolute_robustness_rank").to_csv(
        output_dir / "model_robustness_summary.csv", index=False
    )

    by_type = (
        corrupted.groupby(["model", "degradation"], sort=False)
        .agg(
            mean_top1=("top1_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_top1_drop=("top1_drop", "mean"),
            mean_macro_f1_drop=("macro_f1_drop", "mean"),
        )
        .reset_index()
    )
    by_type.to_csv(output_dir / "robustness_by_degradation.csv", index=False)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir()
    for panel_index, degradation in enumerate(DEGRADATION_LABELS):
        plot_degradation(
            frame,
            degradation,
            plots_dir / f"{degradation}_robustness.png",
            panel_index,
        )
    print(summary.sort_values("absolute_robustness_rank").to_string(index=False))


if __name__ == "__main__":
    main()
