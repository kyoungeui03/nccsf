#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.non_censored.benchmarks import CASE_SPECS, _build_case, _make_cfg, _metric_row  # noqa: E402
from grf.non_censored.models import (  # noqa: E402
    B2SummaryBroadDupNCCausalForest,
    B2SummaryBroadDupRichNCCausalForest,
    B2SummaryNCCausalForest,
    B2SummaryRichNCCausalForest,
)


MODEL_SPECS = [
    ("B2Sum (Base)", B2SummaryNCCausalForest),
    ("B2Sum (ProxyDup)", B2SummaryBroadDupNCCausalForest),
    ("B2Sum (Rich)", B2SummaryRichNCCausalForest),
    ("B2Sum (ProxyDup + Rich)", B2SummaryBroadDupRichNCCausalForest),
]

MODEL_ORDER = [name for name, _ in MODEL_SPECS]


def parse_args():
    parser = argparse.ArgumentParser(description="Run non-censored basic12 B2Sum input-grid benchmark.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "non_censored" / "outputs" / "benchmark_basic12_b2sum_input_grid",
    )
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p-w", type=int, default=1)
    parser.add_argument("--p-z", type=int, default=1)
    return parser.parse_args()


def _evaluate_model(case, name, model_cls):
    start = time.time()
    model = model_cls()
    model.fit_components(case["X"], case["A"], case["Y"], case["Z"], case["W"])
    preds = model.effect_from_components(case["X"], case["W"], case["Z"]).ravel()
    return _metric_row(name, preds, case["true_cate"], time.time() - start)


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
            avg_time=("time_sec", "mean"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = set(args.case_ids) if args.case_ids else None
    selected_cases = [case for case in CASE_SPECS if selected_ids is None or case["case_id"] in selected_ids]

    case_frames = []
    for case_spec in selected_cases:
        cfg = _make_cfg(case_spec)
        cfg.n = int(args.n)
        cfg.p_w = int(args.p_w)
        cfg.p_z = int(args.p_z)
        case = _build_case(cfg, case_spec)

        print("=" * 100, flush=True)
        print(f"Running case {case_spec['case_id']:02d}: {case_spec['slug']}", flush=True)
        print("=" * 100, flush=True)

        rows = []
        for name, model_cls in MODEL_SPECS:
            rows.append(_evaluate_model(case, name, model_cls))

        case_df = pd.DataFrame(rows)
        case_df.insert(0, "case_id", case_spec["case_id"])
        case_df.insert(1, "case_slug", case_spec["slug"])
        case_df.insert(2, "case_title", case_spec["title"])
        case_df["n"] = cfg.n
        case_df["p_x"] = cfg.p_x
        case_df["p_w"] = cfg.p_w
        case_df["p_z"] = cfg.p_z
        case_frames.append(case_df)

        partial_df = pd.concat(case_frames, ignore_index=True)
        partial_df["model_order"] = partial_df["name"].map({name: i for i, name in enumerate(MODEL_ORDER)})
        partial_df = partial_df.sort_values(["case_id", "model_order"]).drop(columns=["model_order"])
        partial_df.to_csv(output_dir / f"results_{args.tag}.csv", index=False)

    combined_df = pd.concat(case_frames, ignore_index=True)
    combined_df["model_order"] = combined_df["name"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    combined_df = combined_df.sort_values(["case_id", "model_order"]).drop(columns=["model_order"])
    results_path = output_dir / f"results_{args.tag}.csv"
    summary_path = output_dir / f"summary_{args.tag}.csv"
    combined_df.to_csv(results_path, index=False)
    _summarize(combined_df).to_csv(summary_path, index=False)
    print(f"Saved {results_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
