#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontcache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "slide_reproduction_conditional_rmst3_q90"
MODEL_ORDER = [
    "Final Conditional Oracle",
    "Final Conditional",
    "Revised Conditional",
]
MODEL_COLORS = {
    "Final Conditional Oracle": "#1f77b4",
    "Final Conditional": "#ff7f0e",
    "Revised Conditional": "#2ca02c",
}
TARGETS = ["RMST", "survival.probability"]
METRICS = ["rmse", "mae"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grouped RMSE/MAE boxplots for RMST and survival probability."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Slide reproduction output directory containing benchmark/results_full_three_models.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the four PNGs. Defaults to <run-dir>/presentation_boxplots.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(20.0, 9.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches.",
    )
    return parser.parse_args()


def _setting_sort_key(setting_id: str) -> tuple[int, str]:
    text = str(setting_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return int(digits), text
    return 10**9, text


def _plot_grouped_boxplot(
    data: pd.DataFrame,
    *,
    target: str,
    metric: str,
    output_path: Path,
    figsize: tuple[float, float],
) -> None:
    target_df = data.loc[data["target"].astype(str) == target].copy()
    settings = sorted(target_df["setting_id"].astype(str).unique().tolist(), key=_setting_sort_key)

    group_gap = 1.4
    box_width = 0.72
    n_models = len(MODEL_ORDER)

    box_data: list[list[float]] = []
    box_positions: list[float] = []

    for i, setting_id in enumerate(settings):
        base = i * (n_models + group_gap)
        for j, model_name in enumerate(MODEL_ORDER):
            values = (
                target_df.loc[
                    (target_df["setting_id"].astype(str) == setting_id)
                    & (target_df["name"].astype(str) == model_name),
                    metric,
                ]
                .astype(float)
                .tolist()
            )
            box_data.append(values)
            box_positions.append(base + j)

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=box_width,
        patch_artist=True,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 3.2,
            "markerfacecolor": "#1f1f1f",
            "markeredgecolor": "#1f1f1f",
            "alpha": 0.9,
        },
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )

    for idx, patch in enumerate(bp["boxes"]):
        model_name = MODEL_ORDER[idx % n_models]
        patch.set_facecolor(MODEL_COLORS[model_name])
        patch.set_alpha(0.72)
        patch.set_edgecolor("#444444")

    tick_positions = [i * (n_models + group_gap) + (n_models - 1) / 2 for i in range(len(settings))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(settings)
    ax.set_xlabel("Setting ID", fontsize=18)
    ax.set_ylabel(metric.upper(), fontsize=18)

    target_label = "Survival Probability" if target == "survival.probability" else "RMST"
    ax.set_title(f"Grouped {metric.upper()} Boxplots ({target_label})", fontsize=24)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    legend_handles = [
        Patch(facecolor=MODEL_COLORS[name], edgecolor="#444444", alpha=0.72, label=name)
        for name in MODEL_ORDER
    ]
    ax.legend(handles=legend_handles, title="Method", loc="upper right", frameon=True, fontsize=14, title_fontsize=14)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "presentation_boxplots")).resolve()
    results_path = run_dir / "benchmark" / "results_full_three_models.csv"

    df = pd.read_csv(results_path)
    required = {"name", "setting_id", "target", "rmse", "mae"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {results_path}: {missing}")

    df = df.loc[df["name"].isin(MODEL_ORDER)].copy()

    for target in TARGETS:
        for metric in METRICS:
            slug = target.replace(".", "_")
            output_path = output_dir / f"grouped_{metric}_boxplot_{slug}.png"
            _plot_grouped_boxplot(
                df,
                target=target,
                metric=metric,
                output_path=output_path,
                figsize=(float(args.figsize[0]), float(args.figsize[1])),
            )
            print(f"Saved plot: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
