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

from grf.benchmarks.econml_8variant import CASE_SPECS, _evaluate_predictions, prepare_case  # noqa: E402
from grf.methods.econml_oldc3_ablation_survival import (  # noqa: E402
    UnifiedB2SumBaselineCensoredSurvivalForest,
    UnifiedB2SumMildShrinkCensoredSurvivalForest,
    UnifiedB2SumSinglePassBaselineCensoredSurvivalForest,
)

BASE_MODEL_SPECS = [
    ("baseline", UnifiedB2SumBaselineCensoredSurvivalForest),
    ("singlepass", UnifiedB2SumSinglePassBaselineCensoredSurvivalForest),
    ("mild", UnifiedB2SumMildShrinkCensoredSurvivalForest),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run basic12 censored blend search over B2Sum variants.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--target", choices=["RMST", "survival.probability"], default="RMST")
    parser.add_argument("--horizon-quantile", type=float, default=0.60)
    parser.add_argument("--grid-denom", type=int, default=50, help="Blend grid denominator. 50 means 0.02 steps.")
    return parser.parse_args()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("name", as_index=False)
        .agg(
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
            avg_sign_acc=("sign_acc", "mean"),
            avg_time=("total_time", "mean"),
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
    return f"BlendC[b={w_baseline:.2f},s={w_singlepass:.2f},m={w_mild:.2f}]"


def metric_row(name: str, preds, case, case_spec, *, target: str, elapsed: float, backend: str) -> dict[str, object]:
    row = _evaluate_predictions(name, preds, case.true_cate, elapsed, backend=backend)
    row.update(
        case_id=int(case_spec["case_id"]),
        case_slug=str(case_spec["slug"]),
        case_title=str(case_spec["title"]),
        target=str(target),
        estimand_horizon=float(case.horizon),
        horizon_quantile=float(0.60) if target == "survival.probability" else None,
        n=int(case.cfg.n),
        p_x=int(case.cfg.p_x),
        p_w=int(case.cfg.p_w),
        p_z=int(case.cfg.p_z),
        seed=int(case.cfg.seed),
        target_censor_rate=float(case.cfg.target_censor_rate),
        actual_censor_rate=float(1.0 - case.delta.mean()),
        linear_treatment=bool(case.cfg.linear_treatment),
        linear_outcome=bool(case.cfg.linear_outcome),
        tau_log_hr=float(case.cfg.tau_log_hr),
    )
    return row


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.case_ids) if args.case_ids else None
    case_specs = [case for case in CASE_SPECS if selected is None or int(case["case_id"]) in selected]
    weight_grid = list(blend_grid(int(args.grid_denom)))

    frames = []
    for case_spec in case_specs:
        print(f"[case {case_spec['case_id']:02d}] start: {case_spec['slug']}", flush=True)
        case = prepare_case(case_spec, target=args.target, horizon_quantile=args.horizon_quantile)
        horizon = None if args.target == "RMST" else float(case.horizon)

        base_preds = {}
        base_times = {}
        for key, model_cls in BASE_MODEL_SPECS:
            print(f"[case {case_spec['case_id']:02d}] fit base model: {key}", flush=True)
            start = time.time()
            model = model_cls(target=args.target, horizon=horizon)
            model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
            base_preds[key] = model.effect_from_components(case.X, case.W, case.Z).ravel()
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
            row = metric_row(
                blend_name(w_baseline, w_singlepass, w_mild),
                preds,
                case,
                case_spec,
                target=args.target,
                elapsed=total_time,
                backend="UnifiedB2SumBlendCensoredSearch",
            )
            row["w_baseline"] = w_baseline
            row["w_singlepass"] = w_singlepass
            row["w_mild"] = w_mild
            rows.append(row)

        case_df = pd.DataFrame(rows).sort_values(["case_id", "name"]).reset_index(drop=True)
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
