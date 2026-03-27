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
    UnifiedB2SumBaselineNCCausalForest,
    UnifiedB2SumMildShrinkNCCausalForest,
    UnifiedB2SumSinglePassBaselineNCCausalForest,
)

BASE_MODEL_SPECS = [
    ("baseline", UnifiedB2SumBaselineNCCausalForest),
    ("singlepass", UnifiedB2SumSinglePassBaselineNCCausalForest),
    ("mild", UnifiedB2SumMildShrinkNCCausalForest),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run basic12 non-censored blend search over B2Sum variants.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p-w", type=int, default=1)
    parser.add_argument("--p-z", type=int, default=1)
    parser.add_argument("--grid-denom", type=int, default=50, help="Blend grid denominator. 50 means 0.02 steps.")
    return parser.parse_args()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
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
            w_baseline=("w_baseline", "first"),
            w_singlepass=("w_singlepass", "first"),
            w_mild=("w_mild", "first"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def blend_grid(denom: int):
    for i in range(denom + 1):
        for j in range(denom + 1 - i):
            k = denom - i - j
            yield i / denom, j / denom, k / denom


def blend_name(w_baseline: float, w_singlepass: float, w_mild: float) -> str:
    return f"BlendNC[b={w_baseline:.2f},s={w_singlepass:.2f},m={w_mild:.2f}]"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.case_ids) if args.case_ids else None
    case_specs = [case for case in CASE_SPECS if selected is None or int(case["case_id"]) in selected]
    weight_grid = list(blend_grid(int(args.grid_denom)))

    frames = []
    for case_spec in case_specs:
        print(f"[case {case_spec['case_id']:02d}] start: {case_spec['slug']}", flush=True)
        cfg = _make_cfg(case_spec)
        cfg.n = int(args.n)
        cfg.p_w = int(args.p_w)
        cfg.p_z = int(args.p_z)
        case = _build_case(cfg, case_spec)

        base_preds = {}
        base_times = {}
        for key, model_cls in BASE_MODEL_SPECS:
            print(f"[case {case_spec['case_id']:02d}] fit base model: {key}", flush=True)
            start = time.time()
            model = model_cls()
            model.fit_components(case["X"], case["A"], case["Y"], case["Z"], case["W"])
            base_preds[key] = model.effect_from_components(case["X"], case["W"], case["Z"]).ravel()
            base_times[key] = time.time() - start

        total_time = sum(base_times.values())
        rows = []
        for idx, (w_baseline, w_singlepass, w_mild) in enumerate(weight_grid, start=1):
            if idx % 200 == 0:
                print(f"[case {case_spec['case_id']:02d}] blend {idx}/{len(weight_grid)}", flush=True)
            preds = (
                w_baseline * base_preds["baseline"]
                + w_singlepass * base_preds["singlepass"]
                + w_mild * base_preds["mild"]
            )
            row = _metric_row(blend_name(w_baseline, w_singlepass, w_mild), preds, case["true_cate"], total_time)
            row["w_baseline"] = w_baseline
            row["w_singlepass"] = w_singlepass
            row["w_mild"] = w_mild
            rows.append(row)

        case_df = pd.DataFrame(rows)
        case_df.insert(0, "case_id", case_spec["case_id"])
        case_df.insert(1, "case_slug", case_spec["slug"])
        case_df.insert(2, "case_title", case_spec["title"])
        case_df["n"] = cfg.n
        case_df["p_x"] = cfg.p_x
        case_df["p_w"] = cfg.p_w
        case_df["p_z"] = cfg.p_z
        frames.append(case_df)
        pd.concat(frames, ignore_index=True).to_csv(args.output_dir / f"results_{args.tag}.csv", index=False)
        print(f"[case {case_spec['case_id']:02d}] done", flush=True)

    results = pd.concat(frames, ignore_index=True).sort_values(["case_id", "name"]).reset_index(drop=True)
    results.to_csv(args.output_dir / f"results_{args.tag}.csv", index=False)
    summary = summarize(results)
    summary.to_csv(args.output_dir / f"summary_{args.tag}.csv", index=False)
    summary.head(50).to_csv(args.output_dir / f"summary_top50_{args.tag}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
