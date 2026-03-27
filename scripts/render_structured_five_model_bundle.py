from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path("/Users/kyoungeuihong/Desktop/csf_grf_new")
DESKTOP = Path("/Users/kyoungeuihong/Desktop")

NC_ROOTS = [
    ROOT / "non_censored/outputs/benchmark_structured_14settings_extended_full",
    ROOT / "non_censored/outputs/benchmark_structured_priority_S05_S07",
]
SURV_ROOTS = [
    ROOT / "outputs/benchmark_structured_14settings_extended_full",
    ROOT / "outputs/benchmark_structured_priority_S05_S07",
]

MODEL_MAP = {
    "non_censored": {
        "EconML Baseline": "EconML Baseline",
        "C3  BestCurve NC-CSF (all estimated q/h)": "New C3",
        "D2  BestCurve no-PCI baseline": "D2",
        "Old C3  Legacy NC-CSF": "Old C3",
        "Old E2  Legacy no-PCI baseline": "Old E2",
    },
    "censored": {
        "R-CSF Baseline": "R-CSF Baseline",
        "C3  BestCurve NC-CSF (all estimated)": "New C3",
        "D2  BestCurve no-PCI baseline": "D2",
        "Old C3  Legacy NC-CSF": "Old C3",
        "Old E2  Legacy no-PCI baseline": "Old E2",
    },
}

OUTPUT_PDF = DESKTOP / "csf_grf_new_5model_structured_summary.pdf"
OUTPUT_NC_PNG = DESKTOP / "csf_grf_new_5model_structured_non_censored.png"
OUTPUT_SURV_PNG = DESKTOP / "csf_grf_new_5model_structured_censored.png"
OUTPUT_NC_CSV = DESKTOP / "csf_grf_new_5model_structured_non_censored.csv"
OUTPUT_SURV_CSV = DESKTOP / "csf_grf_new_5model_structured_censored.csv"


def _collect_setting_results(roots: list[Path]) -> dict[str, pd.DataFrame]:
    setting_frames: dict[str, pd.DataFrame] = {}
    for root in roots:
        if not root.exists():
            continue
        for result_path in sorted(root.glob("S*/results.csv")):
            df = pd.read_csv(result_path)
            if "setting_id" not in df.columns or df.empty:
                continue
            setting_id = str(df["setting_id"].iloc[0])
            setting_frames[setting_id] = df
    return setting_frames


def _build_domain_table(domain: str, frames_by_setting: dict[str, pd.DataFrame], setting_ids: list[str]) -> pd.DataFrame:
    wanted = MODEL_MAP[domain]
    frames = [frames_by_setting[sid] for sid in setting_ids]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["name"].isin(wanted)].copy()
    df["variant"] = df["name"].map(wanted)

    time_col = "time_sec" if "time_sec" in df.columns else "total_time"
    grouped = (
        df.groupby("variant", as_index=False)
        .agg(
            avg_pred_cate=("mean_pred", "mean"),
            avg_true_cate=("mean_true_cate", "mean"),
            avg_acc=("sign_acc", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
            avg_time=(time_col, "mean"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    return grouped


def _format_table(df: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    columns = [
        "Rank",
        "Variant",
        "Avg Pred CATE",
        "Avg True CATE",
        "Avg Acc",
        "Avg RMSE",
        "Avg MAE",
        "Avg Pearson",
        "Avg Bias",
    ]
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                str(int(row["rank"])),
                str(row["variant"]),
                f"{row['avg_pred_cate']:.4f}",
                f"{row['avg_true_cate']:.4f}",
                f"{row['avg_acc']:.4f}",
                f"{row['avg_rmse']:.4f}",
                f"{row['avg_mae']:.4f}",
                f"{row['avg_pearson']:.4f}",
                f"{row['avg_bias']:.4f}",
            ]
        )
    return columns, rows


def _render_table_page(df: pd.DataFrame, *, title: str, subtitle: str, output_png: Path) -> plt.Figure:
    columns, rows = _format_table(df)

    fig, ax = plt.subplots(figsize=(17, 6.8), dpi=200)
    bg = "#1f1f1f"
    cell_bg = "#2b2b2b"
    header_bg = "#303030"
    edge = "#bdbdbd"
    text = "#f2f2f2"
    muted = "#d0d0d0"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")

    fig.text(0.03, 0.95, title, fontsize=22, fontweight="bold", color=text, ha="left", va="top")
    fig.text(0.03, 0.91, subtitle, fontsize=11.5, color=muted, ha="left", va="top")

    col_widths = [0.06, 0.20, 0.12, 0.12, 0.09, 0.09, 0.09, 0.11, 0.09]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        colLoc="center",
        cellLoc="center",
        bbox=[0.02, 0.08, 0.96, 0.76],
        colWidths=col_widths,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(edge)
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.get_text().set_color(text)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(cell_bg)
            cell.get_text().set_color(text)
            if c == 1:
                cell.get_text().set_ha("left")
                cell.PAD = 0.02

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig


def main() -> int:
    nc_frames = _collect_setting_results(NC_ROOTS)
    surv_frames = _collect_setting_results(SURV_ROOTS)

    common_settings = sorted(set(nc_frames) & set(surv_frames))
    if not common_settings:
        raise RuntimeError("No paired structured settings found across non-censored and censored outputs.")

    nc_table = _build_domain_table("non_censored", nc_frames, common_settings)
    surv_table = _build_domain_table("censored", surv_frames, common_settings)

    nc_table.to_csv(OUTPUT_NC_CSV, index=False)
    surv_table.to_csv(OUTPUT_SURV_CSV, index=False)

    subtitle = (
        f"Aggregated over {len(common_settings)} paired structured settings "
        f"({', '.join(common_settings)}), 12 cases per setting."
    )

    nc_fig = _render_table_page(
        nc_table,
        title="Structured Synthetic Summary (Five Models) - Non-Censored",
        subtitle=subtitle,
        output_png=OUTPUT_NC_PNG,
    )
    surv_fig = _render_table_page(
        surv_table,
        title="Structured Synthetic Summary (Five Models) - Censored",
        subtitle=subtitle,
        output_png=OUTPUT_SURV_PNG,
    )

    with PdfPages(OUTPUT_PDF) as pdf:
        pdf.savefig(nc_fig, facecolor=nc_fig.get_facecolor(), bbox_inches="tight")
        pdf.savefig(surv_fig, facecolor=surv_fig.get_facecolor(), bbox_inches="tight")

    plt.close(nc_fig)
    plt.close(surv_fig)
    print(f"Saved PDF: {OUTPUT_PDF}")
    print(f"Saved PNG: {OUTPUT_NC_PNG}")
    print(f"Saved PNG: {OUTPUT_SURV_PNG}")
    print(f"Saved CSV: {OUTPUT_NC_CSV}")
    print(f"Saved CSV: {OUTPUT_SURV_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
