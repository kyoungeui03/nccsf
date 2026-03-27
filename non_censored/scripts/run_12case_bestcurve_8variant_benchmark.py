#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.non_censored.benchmarks import (  # noqa: E402
    CASE_SPECS,
    TABLE_COLUMNS,
    TABLE_KEYS,
    _build_case,
    _make_cfg,
    _metric_row,
    _render_table_png,
    nc_h_from_proxy,
    nc_q_from_proxy,
    oracle_h_from_proxy,
    oracle_q_from_proxy,
)
from grf.non_censored.models import (  # noqa: E402
    BaselineCausalForestDML,
    BestCurveLocalNCCausalForest,
    BestCurveLocalObservedNCCausalForest,
)

matplotlib.use("Agg")


VARIANT_SPECS = [
    {"name": "A1  Oracle BestCurve (all true q/h)", "kind": "oracle", "use_true_q": True, "use_true_h": True},
    {"name": "A2  Oracle BestCurve (true q, est h)", "kind": "oracle", "use_true_q": True, "use_true_h": False},
    {"name": "A3  Oracle BestCurve (all estimated q/h)", "kind": "oracle", "use_true_q": False, "use_true_h": False},
    {"name": "EconML Baseline", "kind": "baseline_xwz"},
    {"name": "C1  BestCurve NC-CSF (all true q/h)", "kind": "nc", "use_true_q": True, "use_true_h": True},
    {"name": "C2  BestCurve NC-CSF (true q, est h)", "kind": "nc", "use_true_q": True, "use_true_h": False},
    {"name": "C3  BestCurve NC-CSF (all estimated q/h)", "kind": "nc", "use_true_q": False, "use_true_h": False},
    {"name": "D2  BestCurve no-PCI baseline", "kind": "d2"},
]

MODEL_ORDER = [spec["name"] for spec in VARIANT_SPECS]
SHORT_LABELS = ["A1", "A2", "A3", "B2", "C1", "C2", "C3", "D2"]
MODEL_COLORS = {
    "A1  Oracle BestCurve (all true q/h)": "#264653",
    "A2  Oracle BestCurve (true q, est h)": "#287271",
    "A3  Oracle BestCurve (all estimated q/h)": "#2a9d8f",
    "EconML Baseline": "#e76f51",
    "C1  BestCurve NC-CSF (all true q/h)": "#7b61ff",
    "C2  BestCurve NC-CSF (true q, est h)": "#6d597a",
    "C3  BestCurve NC-CSF (all estimated q/h)": "#355070",
    "D2  BestCurve no-PCI baseline": "#9c6644",
}

CASE_LABELS = {
    1: "lin / lin / info / strong / bene / large",
    2: "lin / lin / info / weak / harm / small",
    3: "lin / lin / weakproxy / strong / harm / mod",
    4: "lin / lin / weakproxy / weak / near0 / harm",
    5: "lin / nonlin / info / strong / bene / large",
    6: "lin / nonlin / info / weak / harm / small",
    7: "lin / nonlin / weakproxy / strong / harm / mod",
    8: "lin / nonlin / weakproxy / weak / near0 / harm",
    9: "nonlin / nonlin / info / strong / bene / large",
    10: "nonlin / nonlin / info / weak / harm / small",
    11: "nonlin / nonlin / weakproxy / strong / harm / mod",
    12: "nonlin / nonlin / weakproxy / weak / near0 / harm",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 12-case non-censored BestCurve 8-variant benchmark."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "non_censored" / "outputs" / "benchmark_bestcurve_8variant_12case",
    )
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p-w", type=int, default=1)
    parser.add_argument("--p-z", type=int, default=1)
    return parser.parse_args()


def _format_case_title(case_spec, cfg):
    base = str(case_spec["title"]).split(", n=", 1)[0]
    return (
        f"{base}, n={cfg.n}, p_x={cfg.p_x}, p_w={cfg.p_w}, "
        f"p_z={cfg.p_z}, seed={cfg.seed}, censoring rate=0%"
    )


def summarize_results_rmse_first(combined_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        combined_df.groupby("name", as_index=False)
        .agg(
            avg_pred_cate=("mean_pred", "mean"),
            avg_true_cate=("mean_true_cate", "mean"),
            avg_acc=("sign_acc", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_pehe=("pehe", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
            avg_time=("time_sec", "mean"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def _evaluate_b2(case):
    x_full = np.hstack([case["X"], case["W"], case["Z"]])
    start = time.time()
    model = BaselineCausalForestDML(n_estimators=200, min_samples_leaf=20, cv=5, random_state=42)
    model.fit_baseline(x_full, case["A"], case["Y"])
    preds = model.effect(x_full).ravel()
    return _metric_row("EconML Baseline", preds, case["true_cate"], time.time() - start)


def _evaluate_oracle(case, spec):
    start = time.time()
    model = BestCurveLocalNCCausalForest(
        oracle=True,
        use_true_q=spec["use_true_q"],
        use_true_h=spec["use_true_h"],
        q_true_fn=partial(oracle_q_from_proxy, dgp=case["dgp"], cfg=case["cfg"]),
        h_true_fn=partial(oracle_h_from_proxy, cfg=case["cfg"], dgp=case["dgp"]),
    )
    model.fit_oracle(case["X"], case["A"], case["Y"], case["U"])
    preds = model.effect_oracle(case["X"], case["U"]).ravel()
    return _metric_row(spec["name"], preds, case["true_cate"], time.time() - start)


def _evaluate_nc(case, spec):
    start = time.time()
    model = BestCurveLocalNCCausalForest(
        oracle=False,
        use_true_q=spec["use_true_q"],
        use_true_h=spec["use_true_h"],
        q_true_fn=partial(nc_q_from_proxy, dgp=case["dgp"], cfg=case["cfg"]),
        h_true_fn=partial(nc_h_from_proxy, cfg=case["cfg"], dgp=case["dgp"]),
    )
    model.fit_components(case["X"], case["A"], case["Y"], case["Z"], case["W"])
    preds = model.effect_from_components(case["X"], case["W"], case["Z"]).ravel()
    return _metric_row(spec["name"], preds, case["true_cate"], time.time() - start)


def _evaluate_d2(case):
    start = time.time()
    model = BestCurveLocalObservedNCCausalForest()
    model.fit_components(case["X"], case["A"], case["Y"], case["Z"], case["W"])
    preds = model.effect_from_components(case["X"], case["W"], case["Z"]).ravel()
    return _metric_row("D2  BestCurve no-PCI baseline", preds, case["true_cate"], time.time() - start)


def _winner_by_metric(case_rows: pd.DataFrame, metric: str):
    ordered = case_rows.sort_values(metric, ascending=True)
    best_row = ordered.iloc[0]
    return str(best_row["name"]), float(best_row[metric])


def build_case_winner_table(combined_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id in sorted(combined_df["case_id"].unique()):
        case_rows = combined_df[combined_df["case_id"] == case_id].copy()
        rmse_winner, rmse_best = _winner_by_metric(case_rows, "rmse")
        mae_winner, mae_best = _winner_by_metric(case_rows, "mae")
        rows.append(
            {
                "case_id": int(case_id),
                "case_label": CASE_LABELS[int(case_id)],
                "rmse_winner": rmse_winner,
                "rmse_best": rmse_best,
                "mae_winner": mae_winner,
                "mae_best": mae_best,
            }
        )
    return pd.DataFrame(rows)


def build_winner_count_table(case_winner_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        rows.append(
            {
                "name": name,
                "rmse_wins": int((case_winner_df["rmse_winner"] == name).sum()),
                "mae_wins": int((case_winner_df["mae_winner"] == name).sum()),
                "avg_rmse": float(summary_df.loc[summary_df["name"] == name, "avg_rmse"].iloc[0]),
                "avg_mae": float(summary_df.loc[summary_df["name"] == name, "avg_mae"].iloc[0]),
                "avg_pearson": float(summary_df.loc[summary_df["name"] == name, "avg_pearson"].iloc[0]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["rmse_wins", "mae_wins", "avg_rmse"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def render_text_table_png(df: pd.DataFrame, output_path: Path, *, title: str, subtitle: str = ""):
    display = df.copy()
    for col in display.columns:
        if col.endswith("_rmse") or col.endswith("_mae") or col.endswith("_pearson"):
            display[col] = display[col].map(lambda v: f"{float(v):.4f}")
    for col in ["avg_rmse", "avg_mae", "avg_pearson", "rmse_best", "mae_best"]:
        if col in display.columns:
            display[col] = display[col].map(lambda v: f"{float(v):.4f}")
    fig_h = max(4.5, 0.42 * (len(display) + 2))
    fig, ax = plt.subplots(figsize=(22, fig_h))
    ax.axis("off")
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.98)
    if subtitle:
        ax.set_title(subtitle, fontsize=12, color="#4b5563", pad=12)
    table = ax.table(
        cellText=display.values,
        colLabels=list(display.columns),
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.02, 0.98, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("#d1d5db")
            cell.set_linewidth(1.2)
        else:
            cell.set_facecolor("#ffffff" if row % 2 == 1 else "#f3f4f6")
            cell.set_edgecolor("#d1d5db")
            cell.set_linewidth(0.8)
            if col in {0, 1}:
                cell.set_text_props(ha="left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_metric_compare_png(combined_df: pd.DataFrame, output_path: Path, *, metric: str, title: str):
    order = sorted(combined_df["case_id"].unique())
    y = np.arange(len(order))
    height = 0.08
    offsets = np.linspace(-3.5 * height, 3.5 * height, len(MODEL_ORDER))

    fig = plt.figure(figsize=(26, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.2])
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    winner_counts = {name: 0 for name in MODEL_ORDER}
    for idx, name in enumerate(MODEL_ORDER):
        frame = combined_df[combined_df["name"] == name].sort_values("case_id")
        ax.barh(y + offsets[idx], frame[metric], height=height, color=MODEL_COLORS[name], label=SHORT_LABELS[idx])

    for row_idx, case_id in enumerate(order):
        case_rows = combined_df[combined_df["case_id"] == case_id].set_index("name")
        best_name = min(MODEL_ORDER, key=lambda name: float(case_rows.loc[name, metric]))
        best_value = float(case_rows.loc[best_name, metric])
        winner_counts[best_name] += 1
        ax.text(best_value + 0.003, row_idx, SHORT_LABELS[MODEL_ORDER.index(best_name)], va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([CASE_LABELS[i] for i in order], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Case-wise {metric.upper()} comparison", fontsize=22, fontweight="bold", pad=16)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, ncols=2)

    win_values = [winner_counts[name] for name in MODEL_ORDER]
    bars = ax2.bar(SHORT_LABELS, win_values, color=[MODEL_COLORS[n] for n in MODEL_ORDER])
    ax2.set_ylim(0, len(order) + 2)
    ax2.set_ylabel("Number of cases won")
    ax2.set_title("Who wins more cases?", fontsize=22, fontweight="bold", pad=16)
    ax2.tick_params(axis="x", labelrotation=20)
    for bar in bars:
        value = int(bar.get_height())
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{value}", ha="center", va="bottom", fontsize=14, fontweight="bold")

    annotation_lines = [f"{label}: {combined_df[combined_df['name'] == name][metric].mean():.4f}" for label, name in zip(SHORT_LABELS, MODEL_ORDER)]
    ax2.text(
        0.05,
        0.04,
        f"Primary criterion: average {metric.upper()} across 12 cases\n\n" + "\n".join(annotation_lines),
        transform=ax2.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#bbbbbb"),
    )

    fig.suptitle(title, fontsize=28, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_pdf(bundle_paths: list[Path], output_pdf: Path) -> None:
    with PdfPages(output_pdf) as pdf:
        for path in bundle_paths:
            img = plt.imread(path)
            height, width = img.shape[:2]
            fig = plt.figure(figsize=(width / 150, height / 150))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = set(args.case_ids) if args.case_ids else None
    selected_cases = [case for case in CASE_SPECS if selected_ids is None or case["case_id"] in selected_ids]

    case_frames = []
    bundle_paths: list[Path] = []
    for case_spec in selected_cases:
        cfg = _make_cfg(case_spec)
        cfg.n = int(args.n)
        cfg.p_w = int(args.p_w)
        cfg.p_z = int(args.p_z)
        case_title = _format_case_title(case_spec, cfg)
        case = _build_case(cfg, case_spec)
        print("=" * 100)
        print(f"Running case {case_spec['case_id']:02d}")
        print(case_title)
        print("=" * 100)

        rows = []
        for spec in VARIANT_SPECS:
            if spec["kind"] == "baseline_xwz":
                rows.append(_evaluate_b2(case))
            elif spec["kind"] == "oracle":
                rows.append(_evaluate_oracle(case, spec))
            elif spec["kind"] == "nc":
                rows.append(_evaluate_nc(case, spec))
            elif spec["kind"] == "d2":
                rows.append(_evaluate_d2(case))
            else:
                raise ValueError(f"Unknown variant kind: {spec['kind']}")

        case_df = pd.DataFrame(rows)
        case_df.insert(0, "case_id", case_spec["case_id"])
        case_df.insert(1, "case_slug", case_spec["slug"])
        case_df.insert(2, "case_title", case_title)
        case_df["n"] = cfg.n
        case_df["p_x"] = cfg.p_x
        case_df["p_w"] = cfg.p_w
        case_df["p_z"] = cfg.p_z
        case_frames.append(case_df)
        pd.concat(case_frames, ignore_index=True).to_csv(output_dir / "results_partial.csv", index=False)

        case_csv = output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.csv"
        case_png = output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.png"
        case_df.to_csv(case_csv, index=False)
        _render_table_png(
            case_title,
            case_df.to_dict("records"),
            case_png,
            TABLE_COLUMNS,
            TABLE_KEYS,
            meta="BestCurve 8-variant benchmark",
        )
        bundle_paths.append(case_png)

    combined_df = pd.concat(case_frames, ignore_index=True)
    results_csv = output_dir / "all_12case_bestcurve_8variant_results.csv"
    combined_df.to_csv(results_csv, index=False)

    summary_df = summarize_results_rmse_first(combined_df)
    top5_df = summary_df.head(5).copy()
    best_df = summary_df.head(1).copy()

    summary_csv = output_dir / "all_12case_bestcurve_8variant_summary.csv"
    top5_csv = output_dir / "all_12case_bestcurve_8variant_top5.csv"
    best_csv = output_dir / "best_model.csv"
    summary_df.to_csv(summary_csv, index=False)
    top5_df.to_csv(top5_csv, index=False)
    best_df.to_csv(best_csv, index=False)

    summary_png = output_dir / "all_12case_bestcurve_8variant_summary.png"
    top5_png = output_dir / "all_12case_bestcurve_8variant_top5.png"
    summary_subtitle = (
        f"Sorted by avg RMSE, then avg MAE, then avg Pearson | "
        f"n={args.n}, p_x=5, p_w={args.p_w}, p_z={args.p_z}"
    )
    render_text_table_png(summary_df, summary_png, title="Non-censored BestCurve 8-variant average summary", subtitle=summary_subtitle)
    render_text_table_png(top5_df, top5_png, title="Non-censored BestCurve 8-variant top 5", subtitle=f"Top 5 by avg RMSE | n={args.n}, p_w={args.p_w}, p_z={args.p_z}")

    case_winner_df = build_case_winner_table(combined_df)
    case_winner_csv = output_dir / "case_metric_winners.csv"
    case_winner_png = output_dir / "case_metric_winners.png"
    case_winner_df.to_csv(case_winner_csv, index=False)
    render_text_table_png(case_winner_df, case_winner_png, title="12-case RMSE / MAE winners", subtitle="Per-case winning model for each metric")

    winner_count_df = build_winner_count_table(case_winner_df, summary_df)
    winner_count_csv = output_dir / "metric_winner_counts.csv"
    winner_count_png = output_dir / "metric_winner_counts.png"
    winner_count_df.to_csv(winner_count_csv, index=False)
    render_text_table_png(winner_count_df, winner_count_png, title="Winner counts across 12 cases", subtitle="How many cases each model wins on RMSE and MAE")

    rmse_png = output_dir / "bestcurve_8variant_rmse_comparison.png"
    mae_png = output_dir / "bestcurve_8variant_mae_comparison.png"
    render_metric_compare_png(combined_df, rmse_png, metric="rmse", title="Non-censored BestCurve 8-variant benchmark\nRMSE-based comparison")
    render_metric_compare_png(combined_df, mae_png, metric="mae", title="Non-censored BestCurve 8-variant benchmark\nMAE-based comparison")

    pdf_path = output_dir / "bestcurve_8variant_bundle.pdf"
    render_pdf(
        [summary_png, top5_png, case_winner_png, winner_count_png, rmse_png, mae_png, *bundle_paths],
        pdf_path,
    )
    print(f"Saved {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
