from __future__ import annotations

import importlib.util
import sys
import time
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


from grf.non_censored.benchmarks import CASE_SPECS as NC_CASE_SPECS  # noqa: E402
from grf.non_censored.benchmarks import _build_case, _make_cfg  # noqa: E402
from grf.benchmarks.econml_8variant import CASE_SPECS as SURV_CASE_SPECS  # noqa: E402
from grf.benchmarks.econml_8variant import _evaluate_predictions as _evaluate_surv_predictions  # noqa: E402
from grf.benchmarks.econml_8variant import prepare_case as prepare_surv_case  # noqa: E402
from grf.methods import EconmlMildShrinkObservedSurvivalForestMatched  # noqa: E402


ROOT = PROJECT_ROOT
DESKTOP = Path("/Users/kyoungeuihong/Desktop")

NONC_FOUR_MODEL_RESULTS = ROOT / "non_censored/outputs/benchmark_four_model_curve_12case_final/all_12case_four_model_curve_results.csv"
SURV_FIVE_MODEL_RESULTS = ROOT / "outputs/benchmark_five_model_curve_12case/all_12case_five_model_results.csv"

OUTPUT_OLD_E2_RESULTS = ROOT / "non_censored/outputs/benchmark_basic12_old_e2_recomputed/all_12case_old_e2_results.csv"
OUTPUT_OLD_E2_SURV_MATCHED_RESULTS = ROOT / "outputs/benchmark_basic12_old_e2_surv_matched/all_12case_old_e2_surv_matched_results.csv"
OUTPUT_NC_CSV = DESKTOP / "csf_grf_new_5model_basic12_non_censored.csv"
OUTPUT_SURV_CSV = DESKTOP / "csf_grf_new_5model_basic12_censored.csv"
OUTPUT_NC_PNG = DESKTOP / "csf_grf_new_5model_basic12_non_censored.png"
OUTPUT_SURV_PNG = DESKTOP / "csf_grf_new_5model_basic12_censored.png"
OUTPUT_PDF = DESKTOP / "csf_grf_new_5model_basic12_summary.pdf"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


STRUCTURED_SUITE = _load_module(
    "structured_suite_for_basic12",
    PROJECT_ROOT / "scripts" / "run_14setting_structured_suite.py",
)


NONC_NAME_MAP = {
    "BestCurveLocal  PCI curve-summary forest": "New C3",
    "Old C3  NC-CSF raw-proxy forest": "Old C3",
    "Old E2  Legacy no-PCI baseline": "Old E2",
    "BestCurveLocal  no-PCI curve-summary baseline": "D2",
    "EconML Baseline": "EconML Baseline",
}

SURV_NAME_MAP = {
    "BestCurveLocal  PCI curve-summary forest": "New C3",
    "Old C3  NC-CSF raw-proxy forest": "Old C3",
    "Old E2  EconML survival-precompute baseline": "Old E2",
    "Old E2  matched nuisance-only no-PCI baseline": "Old E2",
    "BestCurveLocal  no-PCI curve-summary baseline": "D2",
    "R-CSF Baseline": "R-CSF Baseline",
}

SUMMARY_ORDER = ["New C3", "Old C3", "Old E2", "D2", "B2"]
CASE_ORDER = ["New C3", "Old C3", "Old E2", "D2", "B2"]


def _compute_basic12_old_e2_non_censored() -> pd.DataFrame:
    if OUTPUT_OLD_E2_RESULTS.exists():
        existing = pd.read_csv(OUTPUT_OLD_E2_RESULTS)
        if len(existing) == len(NC_CASE_SPECS):
            return existing
    rows = []
    for case_spec in NC_CASE_SPECS:
        cfg = _make_cfg(case_spec)
        cfg.n = 2000
        cfg.p_w = 1
        cfg.p_z = 1
        case = _build_case(cfg, case_spec)
        row = STRUCTURED_SUITE._evaluate_old_e2_nc(case)
        row["case_id"] = int(case_spec["case_id"])
        row["case_slug"] = str(case_spec["slug"])
        rows.append(row)
    df = pd.DataFrame(rows)[
        ["case_id", "case_slug", "name", "mean_pred", "mean_true_cate", "bias", "rmse", "pehe", "mae", "pearson", "sign_acc", "time_sec", "time_str"]
    ]
    OUTPUT_OLD_E2_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_OLD_E2_RESULTS, index=False)
    return df


def _compute_basic12_old_e2_censored_matched() -> pd.DataFrame:
    if OUTPUT_OLD_E2_SURV_MATCHED_RESULTS.exists():
        existing = pd.read_csv(OUTPUT_OLD_E2_SURV_MATCHED_RESULTS)
        if len(existing) == len(SURV_CASE_SPECS):
            return existing

    rows = []
    for case_spec in SURV_CASE_SPECS:
        case = prepare_surv_case(case_spec, target="RMST", horizon_quantile=0.60)
        model = EconmlMildShrinkObservedSurvivalForestMatched(
            target="RMST",
            horizon=None,
        )
        start = time.time()
        model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
        preds = model.effect_from_components(case.X, case.W, case.Z).ravel()
        elapsed = time.time() - start
        row = _evaluate_surv_predictions(
            "Old E2  matched nuisance-only no-PCI baseline",
            preds,
            case.true_cate,
            elapsed,
            backend=model.__class__.__name__,
        )
        row.update(
            case_id=case_spec["case_id"],
            case_slug=case_spec["slug"],
            case_title=case_spec["title"],
            target="RMST",
            estimand_horizon=float(case.horizon),
            horizon_quantile=None,
            n=case.cfg.n,
            p_x=case.cfg.p_x,
            seed=case.cfg.seed,
            target_censor_rate=case.cfg.target_censor_rate,
            actual_censor_rate=float(1.0 - case.delta.mean()),
            linear_treatment=case.cfg.linear_treatment,
            linear_outcome=case.cfg.linear_outcome,
            tau_log_hr=case.cfg.tau_log_hr,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    OUTPUT_OLD_E2_SURV_MATCHED_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_OLD_E2_SURV_MATCHED_RESULTS, index=False)
    return df


def _summarize(df: pd.DataFrame, name_map: dict[str, str], *, time_col: str) -> pd.DataFrame:
    keep = df[df["name"].isin(name_map)].copy()
    keep["variant"] = keep["name"].map(name_map)
    summary = (
        keep.groupby("variant", as_index=False)
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
        .assign(_order=lambda x: x["variant"].map({name: i for i, name in enumerate(SUMMARY_ORDER)}))
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    summary = summary.drop(columns=["_order"], errors="ignore")
    return summary


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
        "Avg Time",
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
                f"{row['avg_time']:.1f}s",
            ]
        )
    return columns, rows


def _render_table_page(df: pd.DataFrame, *, title: str, subtitle: str, output_png: Path) -> plt.Figure:
    columns, rows = _format_table(df)

    fig, ax = plt.subplots(figsize=(18, 7.5), dpi=200)
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

    fig.text(0.5, 0.965, title, fontsize=21, fontweight="bold", color=text, ha="center", va="top")
    fig.text(0.5, 0.925, subtitle, fontsize=11.5, color=muted, ha="center", va="top")

    col_widths = [0.05, 0.18, 0.11, 0.11, 0.08, 0.08, 0.08, 0.10, 0.08, 0.08]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        colLoc="center",
        cellLoc="center",
        bbox=[0.04, 0.08, 0.92, 0.76],
        colWidths=col_widths,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.65)

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


def _format_case_title_nonc(case_spec: dict) -> str:
    cfg = _make_cfg(case_spec)
    base = str(case_spec["title"]).split(", n=", 1)[0]
    return (
        f"{base}, n={cfg.n}, p={cfg.p_x}, seed={cfg.seed}, censoring rate=0%"
    )


def _prepare_case_df(df: pd.DataFrame, name_map: dict[str, str], *, time_col: str) -> pd.DataFrame:
    out = df[df["name"].isin(name_map)].copy()
    out["variant"] = out["name"].map(name_map)
    out["time_display"] = out[time_col].map(lambda v: f"{float(v):.1f}s")
    order_map = {name: i for i, name in enumerate(CASE_ORDER)}
    out = out.assign(_order=out["variant"].map(order_map)).sort_values(["_order", "rmse", "mae"]).drop(columns=["_order"])
    return out


def _case_table_rows(df: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    columns = ["Variant", "Pred CATE", "True CATE", "Bias", "RMSE", "MAE", "Pearson", "Time"]
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                str(row["variant"]),
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


def _render_case_page(df: pd.DataFrame, *, title: str, subtitle: str) -> plt.Figure:
    columns, rows = _case_table_rows(df)
    fig, ax = plt.subplots(figsize=(18, 7.5), dpi=200)
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

    fig.text(0.5, 0.965, title, fontsize=20, fontweight="bold", color=text, ha="center", va="top")
    fig.text(0.5, 0.925, subtitle, fontsize=11.5, color=muted, ha="center", va="top")

    col_widths = [0.34, 0.10, 0.10, 0.09, 0.08, 0.08, 0.09, 0.07]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        colLoc="center",
        cellLoc="center",
        bbox=[0.06, 0.08, 0.88, 0.76],
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)

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


def main() -> int:
    nonc_four = pd.read_csv(NONC_FOUR_MODEL_RESULTS)
    nonc_old_e2 = _compute_basic12_old_e2_non_censored()
    nonc_combined = pd.concat([nonc_four, nonc_old_e2], ignore_index=True)

    surv_five = pd.read_csv(SURV_FIVE_MODEL_RESULTS)
    surv_old_e2_matched = _compute_basic12_old_e2_censored_matched()
    surv_other = surv_five[~surv_five["name"].isin(
        ["Old E2  EconML survival-precompute baseline", "Old E2  matched nuisance-only no-PCI baseline"]
    )].copy()
    surv_combined = pd.concat([surv_other, surv_old_e2_matched], ignore_index=True)

    nonc_summary = _summarize(nonc_combined, NONC_NAME_MAP, time_col="time_sec")
    surv_summary = _summarize(surv_combined, SURV_NAME_MAP, time_col="total_time")

    nonc_summary.to_csv(OUTPUT_NC_CSV, index=False)
    surv_summary.to_csv(OUTPUT_SURV_CSV, index=False)

    subtitle = "Aggregated over the original basic 12 synthetic cases."
    nonc_fig = _render_table_page(
        nonc_summary,
        title="Basic 12-Case Synthetic Summary (Five Models) - Non-Censored",
        subtitle=subtitle,
        output_png=OUTPUT_NC_PNG,
    )
    surv_fig = _render_table_page(
        surv_summary,
        title="Basic 12-Case Synthetic Summary (Five Models) - Censored",
        subtitle=subtitle,
        output_png=OUTPUT_SURV_PNG,
    )

    with PdfPages(OUTPUT_PDF) as pdf:
        pdf.savefig(nonc_fig, facecolor=nonc_fig.get_facecolor(), bbox_inches="tight")
        pdf.savefig(surv_fig, facecolor=surv_fig.get_facecolor(), bbox_inches="tight")

        nonc_case_df = _prepare_case_df(nonc_combined, NONC_NAME_MAP, time_col="time_sec")
        nonc_title_map = {int(spec["case_id"]): _format_case_title_nonc(spec) for spec in NC_CASE_SPECS}
        for case_id in sorted(nonc_case_df["case_id"].unique()):
            case_rows = nonc_case_df[nonc_case_df["case_id"] == case_id]
            fig = _render_case_page(
                case_rows,
                title=nonc_title_map[int(case_id)],
                subtitle=f"5-model benchmark | Non-censored | Case {int(case_id):02d}",
            )
            pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)

        surv_case_df = _prepare_case_df(surv_combined, SURV_NAME_MAP, time_col="total_time")
        for case_id in sorted(surv_case_df["case_id"].unique()):
            case_rows = surv_case_df[surv_case_df["case_id"] == case_id]
            title = str(case_rows["case_title"].iloc[0]).split(", p_x=", 1)[0]
            fig = _render_case_page(
                case_rows,
                title=title,
                subtitle=f"5-model benchmark | Censored | Case {int(case_id):02d}",
            )
            pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)

    plt.close(nonc_fig)
    plt.close(surv_fig)

    print(f"Saved PDF: {OUTPUT_PDF}")
    print(f"Saved non-censored PNG: {OUTPUT_NC_PNG}")
    print(f"Saved censored PNG: {OUTPUT_SURV_PNG}")
    print(f"Saved non-censored CSV: {OUTPUT_NC_CSV}")
    print(f"Saved censored CSV: {OUTPUT_SURV_CSV}")
    print(f"Saved recomputed Old E2 rows: {OUTPUT_OLD_E2_RESULTS}")
    print(f"Saved matched censored Old E2 rows: {OUTPUT_OLD_E2_SURV_MATCHED_RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
