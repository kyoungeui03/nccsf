#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = [
    "Final Conditional Oracle",
    "Final Conditional",
    "Revised Conditional",
]
MODEL_COLORS = {
    "Final Conditional Oracle": "#dceef7",
    "Final Conditional": "#e8f0fb",
    "Revised Conditional": "#f6ecd8",
}
RMST_SPECS = [
    ("RMST-H1", "RMST-H1 (h = 1.5)", 1.5),
    ("RMST-H2", "RMST-H2 (h = 2.0)", 2.0),
    ("RMST-H3", "RMST-H3 (h = 15.0)", 15.0),
    ("RMST-H4", "RMST-H4 (h = 3.0)", 3.0),
]
SURV_SPECS = [
    ("SURV-Q60", "SURV-Q60"),
    ("SURV-Q70", "SURV-Q70"),
    ("SURV-Q80", "SURV-Q80"),
    ("SURV-Q90", "SURV-Q90"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render eight slide-ready PNG summary tables for the conditional horizon benchmark: "
            "RMST-H1..H4 and SURV-Q60..Q90."
        )
    )
    parser.add_argument(
        "--q60-results",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "five_models_output" / "results_full.csv",
        help="results_full.csv containing the q60 survival benchmark.",
    )
    parser.add_argument(
        "--q90-results",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "paper_style_horizon_conditional_triplet" / "results_full.csv",
        help="results_full.csv containing RMST-H1..H4 and SURV-Q90.",
    )
    parser.add_argument(
        "--q70q80-results",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "paper_style_horizon_conditional_triplet_q70_q80" / "results_full.csv",
        help="results_full.csv containing SURV-Q70 and SURV-Q80.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/kyoungeuihong/Desktop/paper_style_horizon_conditional_pngs"),
        help="Directory where the eight PNG files will be written.",
    )
    return parser.parse_args()


def _filter_models(df: pd.DataFrame, model_col: str = "name") -> pd.DataFrame:
    out = df.loc[df[model_col].isin(MODEL_ORDER)].copy()
    out[model_col] = pd.Categorical(out[model_col], MODEL_ORDER, ordered=True)
    return out.sort_values(model_col).reset_index(drop=True)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("name", as_index=False, observed=False)
        .agg(
            avg_horizon=("horizon", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_bias=("bias", "mean"),
            avg_abs_bias=("bias", lambda values: float(np.mean(np.abs(values)))),
            avg_pearson=("pearson_correlation", "mean"),
            avg_sign=("sign_precision", "mean"),
            case_count=("name", "size"),
        )
        .rename(columns={"name": "Model"})
    )
    summary["Model"] = pd.Categorical(summary["Model"], MODEL_ORDER, ordered=True)
    summary = summary.sort_values("Model").reset_index(drop=True)
    return summary


def _format_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _render_table(df: pd.DataFrame, output_path: Path, *, title: str, subtitle: str) -> None:
    display_df = pd.DataFrame(
        {
            "Model": df["Model"].astype(str),
            "RMSE": df["avg_rmse"].map(lambda v: _format_float(v, 3)),
            "MAE": df["avg_mae"].map(lambda v: _format_float(v, 3)),
            "Signed Bias": df["avg_bias"].map(lambda v: f"{float(v):+.3f}"),
            "|Bias|": df["avg_abs_bias"].map(lambda v: _format_float(v, 3)),
            "Pearson": df["avg_pearson"].map(lambda v: _format_float(v, 3)),
            "Sign %": df["avg_sign"].map(lambda v: f"{100.0 * float(v):.1f}"),
        }
    )

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.axis("off")
    ax.text(0.02, 0.95, title, fontsize=20, ha="left", va="top", transform=ax.transAxes, color="#1f2937")
    ax.text(0.02, 0.89, subtitle, fontsize=11.5, ha="left", va="top", transform=ax.transAxes, color="#4b5563")

    table = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=display_df.columns.tolist(),
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.06, 0.96, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#b9c3d0")
        if row == 0:
            cell.set_facecolor("#233b5d")
            cell.get_text().set_color("white")
        else:
            model_name = display_df.iloc[row - 1, 0]
            cell.set_facecolor(MODEL_COLORS.get(model_name, "#f8fafc"))
            if col == 0:
                cell.get_text().set_ha("left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    q60_df = _filter_models(pd.read_csv(args.q60_results))
    q90_df = _filter_models(pd.read_csv(args.q90_results))
    q70q80_df = _filter_models(pd.read_csv(args.q70q80_results))

    # RMST-H1..H4 come from the q90 paper-style conditional run.
    for label, title, fixed_h in RMST_SPECS:
        subset = q90_df.loc[(q90_df["target"] == "RMST") & (q90_df["horizon_label"] == label)].copy()
        summary = _summarize(subset)
        subtitle = (
            f"Average over 22 settings x 12 cases = {int(summary['case_count'].max())} cells per model. "
            f"Conditional triplet only. Fixed horizon h = {fixed_h:.1f}."
        )
        _render_table(
            summary,
            output_dir / f"{label.lower().replace('-', '_')}_summary.png",
            title=title,
            subtitle=subtitle,
        )

    # SURV-Q60 comes from the existing q60 benchmark without horizon labels.
    surv_q60 = q60_df.loc[q60_df["target"] == "survival.probability"].copy()
    q60_summary = _summarize(surv_q60)
    q60_avg_h = float(q60_summary["avg_horizon"].mean())
    _render_table(
        q60_summary,
        output_dir / "surv_q60_summary.png",
        title="SURV-Q60",
        subtitle=(
            f"Average over 22 settings x 12 cases = {int(q60_summary['case_count'].max())} cells per model. "
            f"Conditional triplet only. Mean realized horizon = {q60_avg_h:.3f}."
        ),
    )

    for label, title in SURV_SPECS[1:3]:
        subset = q70q80_df.loc[
            (q70q80_df["target"] == "survival.probability") & (q70q80_df["horizon_label"] == label)
        ].copy()
        summary = _summarize(subset)
        avg_h = float(summary["avg_horizon"].mean())
        subtitle = (
            f"Average over 22 settings x 12 cases = {int(summary['case_count'].max())} cells per model. "
            f"Conditional triplet only. Mean realized horizon = {avg_h:.3f}."
        )
        _render_table(
            summary,
            output_dir / f"{label.lower().replace('-', '_')}_summary.png",
            title=title,
            subtitle=subtitle,
        )

    surv_q90 = q90_df.loc[
        (q90_df["target"] == "survival.probability") & (q90_df["horizon_label"] == "SURV-Q90")
    ].copy()
    q90_summary = _summarize(surv_q90)
    q90_avg_h = float(q90_summary["avg_horizon"].mean())
    _render_table(
        q90_summary,
        output_dir / "surv_q90_summary.png",
        title="SURV-Q90",
        subtitle=(
            f"Average over 22 settings x 12 cases = {int(q90_summary['case_count'].max())} cells per model. "
            f"Conditional triplet only. Mean realized horizon = {q90_avg_h:.3f}."
        ),
    )

    print(f"Saved 8 PNG files under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
