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
from grf.methods import BroadAugSPCensoredSurvivalForest  # noqa: E402


SURV_MODE_LABEL = {
    "none": "BridgeOnly",
    "pair": "SurvPair",
    "full": "SurvFull",
}

CENSOR_LABEL = {
    "kaplan-meier": "KM",
    "cox": "Cox",
    "nelson-aalen": "NA",
}

MODEL_GRID = [
    (surv_mode, censor_est)
    for surv_mode in ("full", "none", "pair")
    for censor_est in ("kaplan-meier", "cox", "nelson-aalen")
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run basic12 censored BroadAugSP grid over final-pass survival blocks and censoring bridges.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benchmark_basic12_broadaugsp_censored_grid",
    )
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--target", choices=["RMST", "survival.probability"], default="RMST")
    parser.add_argument("--horizon-quantile", type=float, default=0.60)
    return parser.parse_args()


def _model_name(surv_mode: str, censor_estimator: str) -> str:
    return f"BroadAugSP-{SURV_MODE_LABEL[surv_mode]} [{CENSOR_LABEL[censor_estimator]}]"


def _metric_row(
    name: str,
    preds,
    case,
    case_spec,
    *,
    target: str,
    elapsed: float,
    backend: str,
    surv_scalar_mode: str,
    censoring_estimator: str,
) -> dict[str, object]:
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
        surv_scalar_mode=surv_scalar_mode,
        censoring_estimator=censoring_estimator,
    )
    return row


def _evaluate_model(case, *, target: str, surv_scalar_mode: str, censoring_estimator: str):
    label = _model_name(surv_scalar_mode, censoring_estimator)
    print(f"  - {label}", flush=True)
    start = time.time()
    horizon = None if target == "RMST" else float(case.horizon)
    model = BroadAugSPCensoredSurvivalForest(
        target=target,
        horizon=horizon,
        surv_scalar_mode=surv_scalar_mode,
        censoring_estimator=censoring_estimator,
    )
    model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
    preds = model.effect_from_components(case.X, case.W, case.Z).ravel()
    return label, preds, time.time() - start, model.__class__.__name__


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = set(args.case_ids) if args.case_ids else None
    selected_cases = [case for case in CASE_SPECS if selected_ids is None or int(case["case_id"]) in selected_ids]

    case_frames = []
    for case_spec in selected_cases:
        case = prepare_case(case_spec, target=args.target, horizon_quantile=args.horizon_quantile)
        print("=" * 100, flush=True)
        print(f"Running case {int(case_spec['case_id']):02d}: {case_spec['slug']}", flush=True)
        print("=" * 100, flush=True)

        rows = []
        for surv_scalar_mode, censoring_estimator in MODEL_GRID:
            name, preds, elapsed, backend = _evaluate_model(
                case,
                target=args.target,
                surv_scalar_mode=surv_scalar_mode,
                censoring_estimator=censoring_estimator,
            )
            rows.append(
                _metric_row(
                    name,
                    preds,
                    case,
                    case_spec,
                    target=args.target,
                    elapsed=elapsed,
                    backend=backend,
                    surv_scalar_mode=surv_scalar_mode,
                    censoring_estimator=censoring_estimator,
                )
            )

        case_df = pd.DataFrame(rows).sort_values(["case_id", "name"]).reset_index(drop=True)
        case_frames.append(case_df)
        partial_df = pd.concat(case_frames, ignore_index=True).sort_values(["case_id", "name"]).reset_index(drop=True)
        partial_df.to_csv(output_dir / f"results_{args.tag}.csv", index=False)

    combined_df = pd.concat(case_frames, ignore_index=True).sort_values(["case_id", "name"]).reset_index(drop=True)
    results_csv = output_dir / f"results_{args.tag}.csv"
    combined_df.to_csv(results_csv, index=False)

    summary = (
        combined_df.groupby(["name", "surv_scalar_mode", "censoring_estimator"], as_index=False)
        .agg(
            avg_pred=("mean_pred", "mean"),
            avg_true=("mean_true_cate", "mean"),
            avg_acc=("sign_acc", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
            avg_time=("total_time", "mean"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_time"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    summary_csv = output_dir / f"summary_{args.tag}.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved {results_csv}", flush=True)
    print(f"Saved {summary_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
