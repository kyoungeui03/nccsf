#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = Path("/Users/kyoungeuihong/Desktop/horizon_final_vs_revised_conditional_summary.png")


RMST_ROWS = [
    ("RMST-H1", "Setting 1\n(h = 1.5)"),
    ("RMST-H2", "Setting 2\n(h = 2.0)"),
    ("RMST-H3", "Setting 3\n(h = 15.0)"),
    ("RMST-H4", "Setting 4\n(h = 3.0)"),
]
SURV_ROWS = [
    ("SURV-Q60", "q60"),
    ("SURV-Q70", "q70"),
    ("SURV-Q80", "q80"),
    ("SURV-Q90", "q90"),
]


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = PROJECT_ROOT / "outputs"
    q90 = pd.read_csv(base / "paper_style_horizon_conditional_triplet" / "results_full.csv")
    q70q80 = pd.read_csv(base / "paper_style_horizon_conditional_triplet_q70_q80" / "results_full.csv")
    q60 = pd.read_csv(base / "five_models_output" / "results_full.csv")
    return q90, q70q80, q60


def _summary_from_subset(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("name", as_index=False)
        .agg(
            rmse=("rmse", "mean"),
            abs_bias=("bias", lambda values: float(values.abs().mean())),
            horizon=("horizon", "mean"),
        )
        .set_index("name")
    )
    return out


def _build_panel_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    q90, q70q80, q60 = _load_inputs()
    models = ["Final Conditional", "Revised Conditional"]

    rmst_records: list[dict[str, object]] = []
    for label, display in RMST_ROWS:
        subset = q90.loc[
            (q90["target"] == "RMST")
            & (q90["horizon_label"] == label)
            & (q90["name"].isin(models))
        ].copy()
        summary = _summary_from_subset(subset)
        final_rmse = float(summary.loc["Final Conditional", "rmse"])
        revised_rmse = float(summary.loc["Revised Conditional", "rmse"])
        final_abs_bias = float(summary.loc["Final Conditional", "abs_bias"])
        revised_abs_bias = float(summary.loc["Revised Conditional", "abs_bias"])
        rmst_records.append(
            {
                "Horizon": display,
                "Final RMSE": final_rmse,
                "Revised RMSE": revised_rmse,
                "RMSE Gain": 100.0 * (revised_rmse - final_rmse) / revised_rmse,
                "Final |Bias|": final_abs_bias,
                "Revised |Bias|": revised_abs_bias,
                "|Bias| Gain": 100.0 * (revised_abs_bias - final_abs_bias) / revised_abs_bias,
            }
        )

    surv_sources = {
        "SURV-Q60": q60.loc[(q60["target"] == "survival.probability") & (q60["name"].isin(models))].copy(),
        "SURV-Q70": q70q80.loc[
            (q70q80["target"] == "survival.probability")
            & (q70q80["horizon_label"] == "SURV-Q70")
            & (q70q80["name"].isin(models))
        ].copy(),
        "SURV-Q80": q70q80.loc[
            (q70q80["target"] == "survival.probability")
            & (q70q80["horizon_label"] == "SURV-Q80")
            & (q70q80["name"].isin(models))
        ].copy(),
        "SURV-Q90": q90.loc[
            (q90["target"] == "survival.probability")
            & (q90["horizon_label"] == "SURV-Q90")
            & (q90["name"].isin(models))
        ].copy(),
    }

    surv_records: list[dict[str, object]] = []
    for label, display in SURV_ROWS:
        summary = _summary_from_subset(surv_sources[label])
        final_rmse = float(summary.loc["Final Conditional", "rmse"])
        revised_rmse = float(summary.loc["Revised Conditional", "rmse"])
        final_abs_bias = float(summary.loc["Final Conditional", "abs_bias"])
        revised_abs_bias = float(summary.loc["Revised Conditional", "abs_bias"])
        mean_h = float(summary["horizon"].mean())
        surv_records.append(
            {
                "Horizon": display,
                "Mean\nHorizon": mean_h,
                "Final\nRMSE": final_rmse,
                "Revised\nRMSE": revised_rmse,
                "RMSE\nGain": 100.0 * (revised_rmse - final_rmse) / revised_rmse,
                "Final\n|Bias|": final_abs_bias,
                "Revised\n|Bias|": revised_abs_bias,
                "|Bias|\nGain": 100.0 * (revised_abs_bias - final_abs_bias) / revised_abs_bias,
            }
        )

    rmst_df = pd.DataFrame(rmst_records)
    surv_df = pd.DataFrame(surv_records)
    return rmst_df, surv_df


def _format_panel_a(df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                str(row["Horizon"]),
                f'{row["Final RMSE"]:.3f}',
                f'{row["Revised RMSE"]:.3f}',
                f'{row["RMSE Gain"]:.1f}%',
                f'{row["Final |Bias|"]:.3f}',
                f'{row["Revised |Bias|"]:.3f}',
                f'{row["|Bias| Gain"]:.1f}%',
            ]
        )
    return rows


def _format_panel_b(df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                str(row["Horizon"]),
                f'{row["Mean\nHorizon"]:.3f}',
                f'{row["Final\nRMSE"]:.3f}',
                f'{row["Revised\nRMSE"]:.3f}',
                f'{row["RMSE\nGain"]:.1f}%',
                f'{row["Final\n|Bias|"]:.3f}',
                f'{row["Revised\n|Bias|"]:.3f}',
                f'{row["|Bias|\nGain"]:.1f}%',
            ]
        )
    return rows


def _style_table(table, row_colors: list[str]) -> None:
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#c3cfde")
        cell.set_linewidth(0.9)
        if r == 0:
            cell.set_facecolor("#1f446b")
            cell.get_text().set_color("white")
            cell.get_text().set_fontsize(10.5)
        else:
            cell.set_facecolor(row_colors[r - 1])
            cell.get_text().set_fontsize(10.5)
            if c == 0:
                cell.get_text().set_ha("left")


def main() -> int:
    rmst_df, surv_df = _build_panel_data()

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.03,
        0.955,
        "Paper-Style Horizon View: Final vs Revised (Conditional)",
        fontsize=24,
        color="#314d70",
        ha="left",
        va="top",
    )
    ax.text(
        0.03,
        0.918,
        "Averages are taken over 22 settings x 12 cases for each horizon row separately. Lower is better for RMSE and |Bias|.",
        fontsize=12.5,
        color="#6b7c93",
        ha="left",
        va="top",
    )

    panel_a = FancyBboxPatch(
        (0.03, 0.545),
        0.94,
        0.315,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor="#edf3fb",
        edgecolor="#c4d2e5",
        linewidth=1.2,
    )
    ax.add_patch(panel_a)
    ax.text(0.05, 0.83, "Panel A. RMST (paper fixed horizons)", fontsize=20, color="#36597f", ha="left", va="top")
    ax.text(
        0.05,
        0.801,
        "Each row uses the corresponding fixed paper horizon for that setting-matched view.",
        fontsize=11.5,
        color="#7a8ca5",
        ha="left",
        va="top",
    )

    ax_a = fig.add_axes([0.05, 0.58, 0.90, 0.19])
    ax_a.axis("off")
    table_a = ax_a.table(
        cellText=_format_panel_a(rmst_df),
        colLabels=["Horizon", "Final\nRMSE", "Revised\nRMSE", "RMSE\nGain", "Final\n|Bias|", "Revised\n|Bias|", "|Bias|\nGain"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.24, 0.12, 0.125, 0.12, 0.12, 0.125, 0.12],
        bbox=[0, 0, 1, 1],
    )
    table_a.auto_set_font_size(False)
    table_a.scale(1.0, 1.55)
    _style_table(table_a, ["#dfeeee", "#dde6f8", "#f9eed7", "#e9dff7"])

    panel_b = FancyBboxPatch(
        (0.03, 0.235),
        0.94,
        0.25,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor="#eef8f4",
        edgecolor="#c7ded6",
        linewidth=1.2,
    )
    ax.add_patch(panel_b)
    ax.text(0.05, 0.463, "Panel B. Survival Probability (q60, q70, q80, q90)", fontsize=20, color="#36597f", ha="left", va="top")
    ax.text(
        0.05,
        0.435,
        "The survival side uses four quantile-based horizons in increasing order: q60, q70, q80, and q90. The second column shows the mean realized horizon.",
        fontsize=11.5,
        color="#7a8ca5",
        ha="left",
        va="top",
    )

    ax_b = fig.add_axes([0.045, 0.265, 0.91, 0.14])
    ax_b.axis("off")
    table_b = ax_b.table(
        cellText=_format_panel_b(surv_df),
        colLabels=["Horizon", "Mean\nHorizon", "Final\nRMSE", "Revised\nRMSE", "RMSE\nGain", "Final\n|Bias|", "Revised\n|Bias|", "|Bias|\nGain"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.19, 0.12, 0.11, 0.12, 0.11, 0.11, 0.12, 0.12],
        bbox=[0, 0, 1, 1],
    )
    table_b.auto_set_font_size(False)
    table_b.scale(1.0, 1.55)
    _style_table(table_b, ["#dfeeee", "#dde6f8", "#f9eed7", "#e9dff7"])

    comment_box = FancyBboxPatch(
        (0.03, 0.085),
        0.94,
        0.095,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor="#fbf2de",
        edgecolor="#e3d1ab",
        linewidth=1.1,
    )
    ax.add_patch(comment_box)
    comment = (
        "Key message. Final Conditional outperforms Revised Conditional in all eight horizon views on both RMSE and |Bias|. "
        "The advantage is modest at early survival horizons (q60, q70) but becomes clearer at later survival horizons (q80, q90). "
        "On the RMST side, the gap is especially large at h = 15.0, where Final remains stable while Revised degrades sharply."
    )
    ax.text(0.045, 0.131, comment, fontsize=12.5, color="#5f6b72", ha="left", va="center", wrap=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary PNG to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
