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
from grf.non_censored.models import BroadAugSPNCCausalForest  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run non-censored basic12 BroadAugSP direct compare.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--case-ids", nargs="*", type=int)
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
        )
        .sort_values(["avg_rmse", "avg_mae"], ascending=[True, True])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.case_ids) if args.case_ids else None
    case_specs = [case for case in CASE_SPECS if selected is None or case["case_id"] in selected]

    frames = []
    for case_spec in case_specs:
        cfg = _make_cfg(case_spec)
        cfg.n = 2000
        cfg.p_w = 1
        cfg.p_z = 1
        case = _build_case(cfg, case_spec)

        start = time.time()
        model = BroadAugSPNCCausalForest()
        model.fit_components(case["X"], case["A"], case["Y"], case["Z"], case["W"])
        preds = model.effect_from_components(case["X"], case["W"], case["Z"]).ravel()
        row = _metric_row("BroadAugSP (PCI)", preds, case["true_cate"], time.time() - start)

        df = pd.DataFrame([row])
        df.insert(0, "case_id", case_spec["case_id"])
        df.insert(1, "case_slug", case_spec["slug"])
        df.insert(2, "case_title", case_spec["title"])
        df["n"] = cfg.n
        df["p_x"] = cfg.p_x
        df["p_w"] = cfg.p_w
        df["p_z"] = cfg.p_z
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(args.output_dir / f"results_{args.tag}.csv", index=False)

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(args.output_dir / f"results_{args.tag}.csv", index=False)
    summarize(results).to_csv(args.output_dir / f"summary_{args.tag}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
