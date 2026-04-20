#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODELS = [
    "Strict Baseline",
    "Final Model",
    "Final Model Oracle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create grouped boxplots of average absolute subgroup bias by setting and method. "
            "For each setting/case/method, it averages |bias| across subgroups, then draws "
            "boxplots over cases within each setting."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "subgroup_bias_detailed.csv",
        help="Input detailed subgroup bias CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "grouped_subgroup_abs_bias_boxplots.png",
        help="Output plot PNG path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=SCRIPT_DIR / "grouped_subgroup_abs_bias_per_case.csv",
        help="Output CSV with per-setting/per-method/per-case averaged absolute subgroup bias.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Subset of models to include.",
    )
    parser.add_argument(
        "--subgroup-families",
        nargs="*",
        default=None,
        help="Optional subset of subgroup families (e.g., x0 x1 tau x0x1).",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional subset of targets (e.g., RMST survival.probability).",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=16.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=7.0,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--show-fliers",
        action="store_true",
        default=True,
        help="Show outlier points in boxplots.",
    )
    return parser.parse_args()


def _setting_sort_key(value: str) -> int:
    value = str(value)
    try:
        if value.startswith("S"):
            return int(value[1:])
    except Exception:
        pass
    return 10**9


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"setting_id", "case_id", "model", "subgroup_family"}
    if "subgroup_abs_bias" not in df.columns and "subgroup_bias" not in df.columns:
        raise ValueError("Input must contain either 'subgroup_abs_bias' or 'subgroup_bias'.")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    if "subgroup_abs_bias" in df.columns:
        df = df.copy()
        df["abs_bias"] = pd.to_numeric(df["subgroup_abs_bias"], errors="coerce")
    else:
        df = df.copy()
        df["abs_bias"] = pd.to_numeric(df["subgroup_bias"], errors="coerce").abs()
    cols = ["abs_bias", "setting_id", "case_id", "model", "subgroup_family"]
    if "target" in df.columns:
        cols.append("target")
    df = df.dropna(subset=cols)
    df["setting_id"] = df["setting_id"].astype(str)
    if "target" in df.columns:
        df["target"] = df["target"].astype(str)
    return df


def _build_plot_frame(df: pd.DataFrame) -> pd.DataFrame:
    # One value per (setting, model, case): mean absolute subgroup bias over subgroups.
    return (
        df.groupby(["setting_id", "model", "case_id"], as_index=False)
        .agg(mean_abs_subgroup_bias=("abs_bias", "mean"))
        .sort_values(
            ["setting_id", "model", "case_id"],
            key=lambda col: col.map(_setting_sort_key) if col.name == "setting_id" else col,
        )
        .reset_index(drop=True)
    )


def _plot_grouped_boxplots(
    plot_df: pd.DataFrame,
    output_path: Path,
    *,
    fig_width: float,
    fig_height: float,
    show_fliers: bool,
    title: str,
) -> None:
    settings = sorted(plot_df["setting_id"].unique().tolist(), key=_setting_sort_key)
    models = [m for m in DEFAULT_MODELS if m in set(plot_df["model"].astype(str).unique())]

    if not settings or not models:
        raise ValueError("No settings or models available to plot after filtering.")

    model_to_color = {
        "Strict Baseline": "#1f77b4",   # blue
        "Final Model": "#ff7f0e",       # orange
        "Final Model Oracle": "#2ca02c",  # green
    }

    n_models = len(models)
    box_width = min(0.7 / max(n_models, 1), 0.22)

    all_data = []
    positions = []
    colors = []

    for s_idx, setting_id in enumerate(settings):
        center = float(s_idx)
        for m_idx, model_name in enumerate(models):
            offset = (m_idx - (n_models - 1) / 2.0) * (box_width * 1.3)
            pos = center + offset
            vals = plot_df.loc[
                (plot_df["setting_id"] == setting_id) & (plot_df["model"] == model_name),
                "mean_abs_subgroup_bias",
            ].to_numpy()
            if len(vals) == 0:
                continue
            all_data.append(vals)
            positions.append(pos)
            colors.append(model_to_color[model_name])

    if not all_data:
        raise ValueError("No data points found for grouped boxplots.")

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bp = ax.boxplot(
        all_data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=show_fliers,
        flierprops={
            "marker": "o",
            "markerfacecolor": "#111111",
            "markeredgecolor": "#111111",
            "markersize": 1.7,
            "linestyle": "none",
            "alpha": 1.0,
        },
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#333333")

    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.4)

    ax.set_xticks(list(range(len(settings))))
    ax.set_xticklabels(settings, rotation=0)
    ax.set_xlabel("Setting ID")
    ax.set_ylabel("Mean |Subgroup Bias| per Case")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    legend_handles = [Patch(facecolor=model_to_color[m], edgecolor="#333333", alpha=0.55, label=m) for m in models]
    ax.legend(handles=legend_handles, title="Method", loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.input)
    _validate_columns(df)
    df = _prepare_data(df)

    if args.models:
        model_set = set(args.models)
        df = df[df["model"].isin(model_set)].copy()

    if args.subgroup_families:
        family_set = set(args.subgroup_families)
        df = df[df["subgroup_family"].isin(family_set)].copy()

    if args.targets:
        if "target" not in df.columns:
            raise ValueError("--targets was provided but no 'target' column exists in input.")
        target_set = set(args.targets)
        df = df[df["target"].isin(target_set)].copy()

    if df.empty:
        raise ValueError("No rows remain after applying filters.")

    families = sorted(df["subgroup_family"].astype(str).unique().tolist())
    if not families:
        raise ValueError("No subgroup families available after filtering.")

    target_values = sorted(df["target"].astype(str).unique().tolist()) if "target" in df.columns else [None]

    saved_plot_paths: list[Path] = []
    saved_csv_paths: list[Path] = []

    for target in target_values:
        target_df = df if target is None else df[df["target"] == target].copy()
        if target_df.empty:
            continue

        multi_family = len(families) > 1

        for family in families:
            family_df = target_df[target_df["subgroup_family"] == family].copy()
            plot_df = _build_plot_frame(family_df)
            if plot_df.empty:
                continue

            stem_suffix_parts = []
            if target is not None:
                stem_suffix_parts.append(target.replace(".", "_"))
            if multi_family:
                stem_suffix_parts.append(family)
            stem_suffix = "_".join(stem_suffix_parts)

            if stem_suffix:
                plot_path = args.output.with_name(f"{args.output.stem}_{stem_suffix}{args.output.suffix}")
                csv_path = args.csv_output.with_name(f"{args.csv_output.stem}_{stem_suffix}{args.csv_output.suffix}")
            else:
                plot_path = args.output
                csv_path = args.csv_output

            csv_path.parent.mkdir(parents=True, exist_ok=True)
            plot_df.to_csv(csv_path, index=False)

            if target is None:
                title = f"Subgroup Bias by Setting ({family})"
            else:
                title = f"Subgroup Bias by Setting ({family}, {target})"

            _plot_grouped_boxplots(
                plot_df,
                plot_path,
                fig_width=float(args.fig_width),
                fig_height=float(args.fig_height),
                show_fliers=bool(args.show_fliers),
                title=title,
            )

            saved_plot_paths.append(plot_path)
            saved_csv_paths.append(csv_path)

            print(f"Saved plot ({family}, target={target}): {plot_path.resolve()}")
            print(f"Saved per-case summary CSV ({family}, target={target}): {csv_path.resolve()}")
            print(f"Rows in per-case summary ({family}, target={target}): {len(plot_df)}")
            print(f"Settings ({family}, target={target}): {plot_df['setting_id'].nunique()} | Models: {plot_df['model'].nunique()}")

    if not saved_plot_paths:
        raise ValueError("No outputs were produced.")

    print(f"Generated {len(saved_plot_paths)} plot(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
