#!/usr/bin/env python3
"""Create grouped boxplots by setting and model from benchmark results.

Default behavior:
- reads outputs/single_file_censored_models_5model/results_full.csv
- plots target=RMST and metric=rmse
- writes output image to outputs/single_file_censored_models_5model/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PREFERRED_MODEL_ORDER = [
    "Final Conditional Oracle",
    "Final Conditional",
    "Revised Conditional",
]


def _setting_sort_key(setting_id: str) -> tuple[int, str]:
    text = str(setting_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return (int(digits), text)
    return (10**9, text)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_output_dir = project_root / "outputs" / "single_file_censored_models_5model_conditional_suite"

    parser = argparse.ArgumentParser(description="Plot grouped boxplots by setting and model.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=default_output_dir / "results_full.csv",
        help="Path to results_full.csv",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Output image path (default is auto-generated under outputs/single_file_censored_models_5model)",
    )
    parser.add_argument(
        "--target",
        choices=["RMST", "survival.probability"],
        default="RMST",
        help="Target subset to plot",
    )
    parser.add_argument(
        "--metric",
        choices=["rmse", "mae"],
        default="rmse",
        help="Metric column to visualize",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(18.0, 8.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size",
    )
    parser.add_argument(
        "--hide-fliers",
        action="store_true",
        help="Hide outlier points in the boxplots",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset/order of model names to plot.",
    )
    return parser.parse_args()


def _resolve_output_png(args: argparse.Namespace) -> Path:
    if args.output_png is not None:
        return args.output_png

    input_path = args.input_csv.resolve()
    output_dir = input_path.parent
    target_slug = args.target.replace(".", "_")
    return output_dir / f"grouped_{args.metric}_boxplot_{target_slug}.png"


def main() -> int:
    args = parse_args()
    output_png = _resolve_output_png(args)

    df = pd.read_csv(args.input_csv)
    required_cols = {"setting_id", "name", "target", args.metric}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {args.input_csv}: {missing}")

    data = df.loc[df["target"] == args.target].copy()
    if data.empty:
        raise ValueError(f"No rows found for target={args.target}")

    settings = sorted(data["setting_id"].astype(str).unique().tolist(), key=_setting_sort_key)

    present_models = data["name"].astype(str).unique().tolist()
    requested_order = args.models if args.models else PREFERRED_MODEL_ORDER
    model_order = [m for m in requested_order if m in present_models]

    if not model_order:
        raise ValueError(
            "No requested models are present in the input file. "
            f"Available models: {sorted(present_models)}"
        )

    box_data: list[np.ndarray] = []
    box_positions: list[float] = []

    n_models = len(model_order)
    group_gap = 1.4
    box_width = 0.7

    for i, setting_id in enumerate(settings):
        base = i * (n_models + group_gap)
        for j, model_name in enumerate(model_order):
            values = data.loc[
                (data["setting_id"].astype(str) == setting_id) & (data["name"].astype(str) == model_name),
                args.metric,
            ].to_numpy(dtype=float)

            if values.size == 0:
                values = np.array([np.nan], dtype=float)

            box_data.append(values)
            box_positions.append(base + j)

    preferred_colors = {
        "Final Conditional Oracle": "#6aaed6",
        "Final Conditional": "#f28e2b",
        "Revised Conditional": "#59a14f",
    }
    cmap = plt.get_cmap("tab10")
    model_colors = {
        model_name: preferred_colors.get(model_name, cmap(i % 10))
        for i, model_name in enumerate(model_order)
    }

    fig, ax = plt.subplots(figsize=(args.figsize[0], args.figsize[1]))
    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=box_width,
        patch_artist=True,
        showfliers=not args.hide_fliers,
        flierprops={
            "marker": "o",
            "markersize": 4.0,
            "markerfacecolor": "#1f1f1f",
            "markeredgecolor": "#ffffff",
            "markeredgewidth": 0.5,
            "alpha": 0.9,
        },
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )

    for k, patch in enumerate(bp["boxes"]):
        model_idx = k % n_models
        model_name = model_order[model_idx]
        patch.set_facecolor(model_colors[model_name])
        patch.set_alpha(0.55)
        patch.set_edgecolor("#444444")

    tick_positions = [i * (n_models + group_gap) + (n_models - 1) / 2 for i in range(len(settings))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(settings)

    ax.set_xlabel("Setting ID", fontsize=13)
    ax.set_ylabel(args.metric.upper(), fontsize=13)
    ax.set_title(
        f"Grouped {args.metric.upper()} Boxplots by Setting and Method ({args.target})",
        fontsize=17,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    legend_handles = [
        Patch(facecolor=model_colors[m], edgecolor="#444444", alpha=0.55, label=m)
        for m in model_order
    ]
    ax.legend(handles=legend_handles, title="Method", loc="upper right", frameon=True)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    print(f"Saved plot: {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
