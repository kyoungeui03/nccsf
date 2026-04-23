#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontcache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "slide_reproduction_conditional_rmst3_q90"


def _load_summary(run_dir: Path, target_slug: str) -> pd.DataFrame:
    path = run_dir / "benchmark" / f"basic12_conditional_suite_{target_slug}_summary.csv"
    df = pd.read_csv(path).copy()
    return df.loc[
        :,
        [
            "rank",
            "name",
            "avg_rmse",
            "avg_mae",
            "avg_pearson_correlation",
            "avg_bias",
            "avg_sign_precision",
            "avg_total_time",
        ],
    ]


def _format_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rank"] = out["rank"].astype(int).astype(str)
    out["name"] = out["name"].astype(str)
    out["avg_rmse"] = out["avg_rmse"].map(lambda x: f"{float(x):.4f}")
    out["avg_mae"] = out["avg_mae"].map(lambda x: f"{float(x):.4f}")
    out["avg_pearson_correlation"] = out["avg_pearson_correlation"].map(lambda x: f"{float(x):.4f}")
    out["avg_bias"] = out["avg_bias"].map(lambda x: f"{float(x):+.4f}")
    out["avg_sign_precision"] = out["avg_sign_precision"].map(lambda x: f"{100.0 * float(x):.1f}%")
    out["avg_total_time"] = out["avg_total_time"].map(lambda x: f"{float(x):.1f}s")
    out.columns = [
        "Rank",
        "Variant",
        "Avg RMSE",
        "Avg MAE",
        "Avg Pearson Corr",
        "Avg Bias",
        "Avg Sign Precision",
        "Avg Time",
    ]
    return out


def _draw_table(ax, df: pd.DataFrame, *, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=28, pad=18, fontfamily="serif")

    table = ax.table(
        cellText=df.values.tolist(),
        colLabels=df.columns.tolist(),
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.025, 0.02, 0.95, 0.88],
        colWidths=[0.08, 0.34, 0.11, 0.11, 0.14, 0.09, 0.13, 0.1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.0, 1.65)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C7D0DB")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#E9EEF5")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#1F2937")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F8FAFC")
            cell.get_text().set_color("#111827")
            if col == 1:
                cell.get_text().set_ha("left")


def _render_single_table(df: pd.DataFrame, *, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 5.8))
    fig.patch.set_facecolor("white")
    _draw_table(ax, df, title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_combined_table(rmst_df: pd.DataFrame, surv_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.patch.set_facecolor("white")
    _draw_table(axes[0], rmst_df, title="RMST")
    _draw_table(axes[1], surv_df, title="Survival Probability")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(h_pad=2.2)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    run_dir = DEFAULT_RUN_DIR.resolve()
    output_dir = run_dir / "presentation_tables"

    rmst_df = _format_summary(_load_summary(run_dir, "RMST"))
    surv_df = _format_summary(_load_summary(run_dir, "survival_probability"))

    _render_single_table(
        rmst_df,
        title="RMST",
        output_path=output_dir / "rmst_summary_table.png",
    )
    _render_single_table(
        surv_df,
        title="Survival Probability",
        output_path=output_dir / "survival_probability_summary_table.png",
    )
    _render_combined_table(
        rmst_df,
        surv_df,
        output_path=output_dir / "combined_summary_tables.png",
    )

    print(f"Saved table: {output_dir / 'rmst_summary_table.png'}")
    print(f"Saved table: {output_dir / 'survival_probability_summary_table.png'}")
    print(f"Saved table: {output_dir / 'combined_summary_tables.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
