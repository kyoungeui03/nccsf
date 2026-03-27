#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks.econml_8variant import CASE_SPECS, prepare_case  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmark_basic12_oldc3_ablation_censored"
DESKTOP = Path("/Users/kyoungeuihong/Desktop")
OUTPUT_CSV = DESKTOP / "csf_grf_new_oldc3_ablation_basic12_censored.csv"
OUTPUT_PNG = DESKTOP / "csf_grf_new_oldc3_ablation_basic12_censored.png"
OUTPUT_PDF = DESKTOP / "csf_grf_new_oldc3_ablation_basic12_censored.pdf"

MODEL_ORDER = [
    "Old C3",
    "New C3",
    "SummaryMinimal-DML (PCI)",
    "SummaryMinimal-DML (no PCI)",
    "AugmentedMinimal-DML (PCI)",
    "AugmentedMinimal-DML (no PCI)",
    "SummaryMinimal-GRF (PCI)",
    "SummaryMinimal-GRF (no PCI)",
    "AugmentedMinimal-GRF (PCI)",
    "AugmentedMinimal-GRF (no PCI)",
    "SummarySurv-DML (PCI)",
    "SummarySurv-DML (no PCI)",
    "AugmentedSurv-DML (PCI)",
    "AugmentedSurv-DML (no PCI)",
    "SummarySurv-GRF (PCI)",
    "SummarySurv-GRF (no PCI)",
    "AugmentedSurv-GRF (PCI)",
    "AugmentedSurv-GRF (no PCI)",
]


def _load_results() -> pd.DataFrame:
    files = sorted(OUTPUT_DIR.glob("results_*.csv"))
    if not files:
        raise FileNotFoundError(f"No results_*.csv files found in {OUTPUT_DIR}")
    frames = [pd.read_csv(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["case_id", "name"], keep="last")
    combined["model_order"] = combined["name"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    combined = combined.sort_values(["case_id", "model_order"]).drop(columns=["model_order"])
    return combined


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("name", as_index=False)
        .agg(
            avg_pred_cate=("mean_pred", "mean"),
            avg_true_cate=("mean_true_cate", "mean"),
            avg_acc=("sign_acc", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
            avg_time=("total_time", "mean"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def _format_table(df: pd.DataFrame):
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
        "Avg Time",
    ]
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                str(int(row["rank"])),
                str(row["name"]),
                f"{row['avg_pred_cate']:.4f}",
                f"{row['avg_true_cate']:.4f}",
                f"{row['avg_acc']:.4f}",
                f"{row['avg_rmse']:.4f}",
                f"{row['avg_mae']:.4f}",
                f"{row['avg_pearson']:.4f}",
                f"{row['avg_bias']:.4f}",
                f"{row['avg_time']:.1f}s",
            ]
        )
    return columns, rows


def _render_summary_page(df: pd.DataFrame, *, title: str, subtitle: str, output_png: Path):
    columns, rows = _format_table(df)
    fig, ax = plt.subplots(figsize=(24, 9), dpi=200)
    bg = "#ffffff"
    cell_bg = "#f3f4f6"
    alt_bg = "#ffffff"
    header_bg = "#243042"
    edge = "#cbd5e1"
    text = "#222222"
    muted = "#667085"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")

    fig.text(0.5, 0.965, title, fontsize=22, fontweight="bold", color=text, ha="center", va="top")
    fig.text(0.5, 0.925, subtitle, fontsize=11.5, color=muted, ha="center", va="top")

    col_widths = [0.04, 0.31, 0.07, 0.07, 0.06, 0.06, 0.06, 0.07, 0.06, 0.06]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        colLoc="center",
        cellLoc="center",
        bbox=[0.02, 0.07, 0.96, 0.79],
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.8)
    table.scale(1, 1.5)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(edge)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.get_text().set_color("#ffffff")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(alt_bg if r % 2 == 1 else cell_bg)
            cell.get_text().set_color(text)
            if c == 1:
                cell.get_text().set_ha("left")
                cell.PAD = 0.02

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig


def _prepare_case_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time_display"] = out["total_time"].map(lambda v: f"{float(v):.1f}s")
    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    return out.assign(_order=out["name"].map(order_map)).sort_values(["_order"]).drop(columns=["_order"])


def _case_table_rows(df: pd.DataFrame):
    columns = ["Variant", "Pred CATE", "True CATE", "Bias", "RMSE", "MAE", "Pearson", "Time"]
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                str(row["name"]),
                f"{row['mean_pred']:.4f}",
                f"{row['mean_true_cate']:.4f}",
                f"{row['bias']:.4f}",
                f"{row['rmse']:.4f}",
                f"{row['mae']:.4f}",
                f"{row['pearson']:.4f}",
                str(row["time_display"]),
            ]
        )
    return columns, rows


def _render_case_page(df: pd.DataFrame, *, title: str, subtitle: str):
    columns, rows = _case_table_rows(df)
    fig, ax = plt.subplots(figsize=(24, 9), dpi=200)
    bg = "#ffffff"
    cell_bg = "#f3f4f6"
    alt_bg = "#ffffff"
    header_bg = "#243042"
    edge = "#cbd5e1"
    text = "#222222"
    muted = "#667085"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")

    fig.text(0.5, 0.965, title, fontsize=19.5, fontweight="bold", color=text, ha="center", va="top")
    fig.text(0.5, 0.925, subtitle, fontsize=11.5, color=muted, ha="center", va="top")

    col_widths = [0.44, 0.08, 0.08, 0.08, 0.07, 0.07, 0.08, 0.06]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        colLoc="center",
        cellLoc="center",
        bbox=[0.03, 0.08, 0.94, 0.78],
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.9)
    table.scale(1, 1.42)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(edge)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.get_text().set_color("#ffffff")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(alt_bg if r % 2 == 1 else cell_bg)
            cell.get_text().set_color(text)
            if c == 0:
                cell.get_text().set_ha("left")
                cell.PAD = 0.02
    return fig


def _format_case_title(case_spec: dict) -> str:
    case = prepare_case(case_spec, target="RMST", horizon_quantile=0.60)
    base = str(case_spec["title"]).split(", n=", 1)[0]
    censor_pct = int(round(100 * float(case.cfg.target_censor_rate)))
    return f"{base}, n={case.cfg.n}, p={case.cfg.p_x}, seed={case.cfg.seed}, censoring rate={censor_pct}%"


def main() -> int:
    combined = _load_results()
    if len(combined["case_id"].unique()) != 12:
        raise RuntimeError(f"Expected 12 cases before rendering, found {sorted(combined['case_id'].unique())}")

    summary = _summarize(combined)
    summary.to_csv(OUTPUT_CSV, index=False)

    subtitle = "Aggregated over the original basic 12 censored synthetic cases."
    summary_fig = _render_summary_page(
        summary,
        title="Basic 12-Case Censored Benchmark: Old C3 Feature Ablations",
        subtitle=subtitle,
        output_png=OUTPUT_PNG,
    )

    case_df = _prepare_case_df(combined)
    title_map = {int(spec["case_id"]): _format_case_title(spec) for spec in CASE_SPECS}
    with PdfPages(OUTPUT_PDF) as pdf:
        pdf.savefig(summary_fig, facecolor=summary_fig.get_facecolor(), bbox_inches="tight")
        for case_id in sorted(case_df["case_id"].unique()):
            rows = case_df[case_df["case_id"] == case_id]
            fig = _render_case_page(
                rows,
                title=title_map[int(case_id)],
                subtitle=f"18-model benchmark | Censored | Case {int(case_id):02d}",
            )
            pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)

    plt.close(summary_fig)
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved PNG: {OUTPUT_PNG}")
    print(f"Saved PDF: {OUTPUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
