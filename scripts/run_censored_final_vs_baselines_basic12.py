#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks.econml_8variant import (  # noqa: E402
    CASE_SPECS,
    _evaluate_predictions,
    evaluate_r_csf_variant,
    prepare_case,
)
from grf.censored import (  # noqa: E402
    FinalModelCensoredSurvivalForest,
    ProperNoPCICensoredSurvivalForest,
    RCSFStyleEconmlCensoredBaseline,
    StrictEconmlXWZCensoredSurvivalForest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the censored basic12 benchmark for Current Final Model (full), "
            "Strict EconML baseline, RCSF-style EconML baseline, and the installed R-CSF baseline."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--target", choices=["RMST", "survival.probability"], default="RMST")
    parser.add_argument("--horizon-quantile", type=float, default=0.90)
    parser.add_argument("--num-trees-r", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / "benchmark_censored_final_vs_baselines_basic12").resolve()


def _metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("name", as_index=False)
        .agg(
            avg_pred_cate=("mean_pred", "mean"),
            avg_true_cate=("mean_true_cate", "mean"),
            avg_acc=("sign_acc", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_pehe=("pehe", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
            avg_time=("total_time", "mean"),
            n_cases=("case_id", "count"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def _evaluate_case(case, *, target: str, num_trees_r: int, random_state: int) -> list[dict[str, object]]:
    x = np.asarray(case.X, dtype=float)
    w = np.asarray(case.W, dtype=float)
    z = np.asarray(case.Z, dtype=float)
    a = np.asarray(case.A, dtype=float)
    time_obs = np.asarray(case.Y, dtype=float)
    event = np.asarray(case.delta, dtype=float)
    feature_cols = [*case.x_cols, *case.w_cols, *case.z_cols]
    horizon = float(case.horizon)

    rows: list[dict[str, object]] = []

    python_models = [
        (
            "Final Model (full)",
            FinalModelCensoredSurvivalForest(
                target=target,
                horizon=horizon,
                random_state=random_state,
                surv_scalar_mode="full",
            ),
        ),
        (
            "Strict EconML Censored Baseline",
            StrictEconmlXWZCensoredSurvivalForest(
                target=target,
                horizon=horizon,
                random_state=random_state,
            ),
        ),
        (
            "RCSF-Style EconML Baseline",
            RCSFStyleEconmlCensoredBaseline(
                target=target,
                horizon=horizon,
                random_state=random_state,
            ),
        ),
        (
            "Proper Censored Baseline",
            ProperNoPCICensoredSurvivalForest(
                target=target,
                horizon=horizon,
                random_state=random_state,
            ),
        ),
    ]

    for name, model in python_models:
        t0 = time.time()
        model.fit_components(x, a, time_obs, event, z, w)
        preds = model.effect_from_components(x, w, z).ravel()
        elapsed = time.time() - t0
        rows.append(
            _evaluate_predictions(
                name,
                preds,
                case.true_cate,
                elapsed,
                backend=model.__class__.__name__,
            )
        )

    rows.append(
        evaluate_r_csf_variant(
            "R-CSF Baseline",
            case.obs_df,
            feature_cols,
            case.true_cate,
            horizon,
            num_trees=num_trees_r,
            target=target,
        )
    )
    return rows


def main() -> int:
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = None if not args.case_ids else set(args.case_ids)
    case_specs = [spec for spec in CASE_SPECS if case_ids is None or int(spec["case_id"]) in case_ids]

    frames: list[pd.DataFrame] = []
    for case_spec in case_specs:
        case = prepare_case(case_spec, target=args.target, horizon_quantile=args.horizon_quantile)
        rows = _evaluate_case(
            case,
            target=args.target,
            num_trees_r=args.num_trees_r,
            random_state=args.random_state,
        )
        case_df = pd.DataFrame(rows)
        case_df["case_id"] = int(case_spec["case_id"])
        case_df["case_slug"] = str(case_spec["slug"])
        case_df["target"] = args.target
        case_df["horizon"] = float(case.horizon)
        frames.append(case_df)
        case_df.to_csv(output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.csv", index=False)

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(output_dir / "results_full.csv", index=False)
    _metric_summary(results).to_csv(output_dir / "summary_full.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
