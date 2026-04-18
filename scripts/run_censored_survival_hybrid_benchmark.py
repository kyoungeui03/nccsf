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

from grf.benchmarks import CASE_SPECS, prepare_case  # noqa: E402
from grf.benchmarks.econml_8variant import _evaluate_predictions  # noqa: E402
from grf.censored import (  # noqa: E402
    FinalModelCensoredSurvivalForest,
    StrictEconmlXWZCensoredSurvivalForest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run censored Final Model hybrid experiments that keep the EconML final learner "
            "fixed while varying the survival/censoring nuisance stack."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--target", default="RMST")
    parser.add_argument("--horizon-quantile", type=float, default=0.90)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--survival-forest-num-trees", type=int, default=50)
    parser.add_argument("--survival-forest-min-node-size", type=int, default=15)
    parser.add_argument("--survival-fast-logrank", action="store_true")
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / "benchmark_censored_survival_hybrid_basic12").resolve()


def _metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(
            [
                "name",
                "event_survival_estimator",
                "censoring_estimator",
                "m_pred_mode",
                "finite_horizon",
                "broad_dup",
                "clipping",
            ],
            as_index=False,
        )
        .agg(
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_sign_acc=("sign_acc", "mean"),
            avg_time=("total_time", "mean"),
            n_cases=("case_id", "count"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def _hybrid_variant_specs() -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    for event_estimator in ("cox", "survival_forest"):
        for censoring_estimator in ("nelson-aalen", "survival_forest"):
            for m_pred_mode in ("bridge", "survival"):
                label = (
                    f"Hybrid E={event_estimator} "
                    f"C={censoring_estimator} "
                    f"M={m_pred_mode}"
                )
                if (
                    event_estimator == "cox"
                    and censoring_estimator == "nelson-aalen"
                    and m_pred_mode == "bridge"
                ):
                    label = "Final Model"
                variants.append(
                    {
                        "name": label,
                        "event_survival_estimator": event_estimator,
                        "censoring_estimator": censoring_estimator,
                        "m_pred_mode": m_pred_mode,
                    }
                )
    return variants


def _evaluate_case(case, *, args: argparse.Namespace) -> list[dict[str, object]]:
    x = np.asarray(case.X, dtype=float)
    w = np.asarray(case.W, dtype=float)
    z = np.asarray(case.Z, dtype=float)
    a = np.asarray(case.A, dtype=float)
    time_obs = np.asarray(case.Y, dtype=float)
    event = np.asarray(case.delta, dtype=float)

    rows: list[dict[str, object]] = []
    for spec in _hybrid_variant_specs():
        model = FinalModelCensoredSurvivalForest(
            target=args.target,
            horizon=case.horizon,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            nuisance_feature_mode="broad_dup",
            event_survival_estimator=str(spec["event_survival_estimator"]),
            censoring_estimator=str(spec["censoring_estimator"]),
            m_pred_mode=str(spec["m_pred_mode"]),
            survival_forest_num_trees=args.survival_forest_num_trees,
            survival_forest_min_node_size=args.survival_forest_min_node_size,
            survival_fast_logrank=args.survival_fast_logrank,
            enforce_finite_horizon=True,
        )
        t0 = time.time()
        model.fit_components(x, a, time_obs, event, z, w)
        preds = model.effect_from_components(x, w, z).ravel()
        row = _evaluate_predictions(
            str(spec["name"]),
            preds,
            case.true_cate,
            time.time() - t0,
            backend=model.__class__.__name__,
        )
        row["event_survival_estimator"] = str(spec["event_survival_estimator"])
        row["censoring_estimator"] = str(spec["censoring_estimator"])
        row["m_pred_mode"] = str(spec["m_pred_mode"])
        row["finite_horizon"] = True
        row["broad_dup"] = True
        row["clipping"] = "canonical"
        rows.append(row)

    econml_model = StrictEconmlXWZCensoredSurvivalForest(
        target=args.target,
        horizon=case.horizon,
        random_state=args.random_state,
    )
    t0 = time.time()
    econml_model.fit_components(x, a, time_obs, event, z, w)
    preds = econml_model.effect_from_components(x, w, z).ravel()
    row = _evaluate_predictions(
        "EconML Censored Baseline",
        preds,
        case.true_cate,
        time.time() - t0,
        backend=econml_model.__class__.__name__,
    )
    row["event_survival_estimator"] = "none"
    row["censoring_estimator"] = "nelson-aalen"
    row["m_pred_mode"] = "pseudo_outcome"
    row["finite_horizon"] = True
    row["broad_dup"] = False
    row["clipping"] = "none"
    rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = set(args.case_ids) if args.case_ids else None
    case_specs = [
        spec for spec in CASE_SPECS
        if case_ids is None or int(spec["case_id"]) in case_ids
    ]

    frames: list[pd.DataFrame] = []
    for case_spec in case_specs:
        case = prepare_case(case_spec, target=args.target, horizon_quantile=args.horizon_quantile)
        case_rows = _evaluate_case(case, args=args)
        case_df = pd.DataFrame(case_rows)
        case_df["case_id"] = int(case_spec["case_id"])
        case_df["case_slug"] = str(case_spec["slug"])
        case_df["horizon"] = float(case.horizon)
        case_df.to_csv(output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.csv", index=False)
        frames.append(case_df)

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(output_dir / "results_full.csv", index=False)
    _metric_summary(results).to_csv(output_dir / "summary_full.csv", index=False)


if __name__ == "__main__":
    main()
