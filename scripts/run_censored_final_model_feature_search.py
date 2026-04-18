#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks import CASE_SPECS, prepare_case  # noqa: E402
from grf.benchmarks.econml_8variant import _evaluate_predictions  # noqa: E402
from grf.censored import FinalModelCensoredSurvivalForest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search censored Final Model final-stage feature sets on basic12 while "
            "keeping the EconML final learner and overall model structure fixed."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--target", default="RMST")
    parser.add_argument("--horizon-quantile", type=float, default=0.90)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 1) // 2)))
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / "benchmark_censored_final_model_feature_search_basic12").resolve()


def _base_balanced_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "target": args.target,
        "random_state": args.random_state,
        "n_estimators": 200,
        "min_samples_leaf": 20,
        "cv": 5,
        "nuisance_feature_mode": "broad_dup",
        "prediction_nuisance_mode": "fold_ensemble",
        "include_raw_proxy": True,
        "include_surv_scalar": True,
        "surv_scalar_mode": "full",
        "q_kind": "logit",
        "q_trees": 300,
        "q_min_samples_leaf": 20,
        "q_poly_degree": 2,
        "h_kind": "extra",
        "h_n_estimators": 1200,
        "h_min_samples_leaf": 3,
        "q_clip": 0.03,
        "y_tilde_clip_quantile": 0.98,
        "y_res_clip_percentiles": (2.0, 98.0),
        "censoring_estimator": "kaplan-meier",
        "event_survival_estimator": "cox",
        "m_pred_mode": "bridge",
        "enforce_finite_horizon": True,
        "n_jobs": 1,
    }


def _variant_specs() -> list[dict[str, object]]:
    return [
        {
            "variant_id": "v00_current_default",
            "name": "Current Final default",
            "params": {
                "surv_scalar_mode": "pair",
                "prediction_nuisance_mode": "full_refit",
                "h_n_estimators": 600,
                "h_min_samples_leaf": 5,
                "censoring_estimator": "nelson-aalen",
            },
        },
        {
            "variant_id": "v01_balanced_pair",
            "name": "Balanced pair baseline",
            "params": {
                "surv_scalar_mode": "pair",
            },
        },
        {
            "variant_id": "v02_balanced_full",
            "name": "Balanced full baseline",
            "params": {
                "surv_scalar_mode": "full",
            },
        },
        {
            "variant_id": "v03_qhat_diff_pts",
            "name": "Feature set: qhat diff points",
            "params": {
                "surv_scalar_mode": "qhat_diff_pts",
            },
        },
        {
            "variant_id": "v04_qhat_full_pts",
            "name": "Feature set: qhat full points",
            "params": {
                "surv_scalar_mode": "qhat_full_pts",
            },
        },
        {
            "variant_id": "v05_qhat_stats",
            "name": "Feature set: qhat stats",
            "params": {
                "surv_scalar_mode": "qhat_stats",
            },
        },
        {
            "variant_id": "v06_s_stats",
            "name": "Feature set: survival stats",
            "params": {
                "surv_scalar_mode": "s_stats",
            },
        },
        {
            "variant_id": "v07_censor_stats",
            "name": "Feature set: censor stats",
            "params": {
                "surv_scalar_mode": "censor_stats",
            },
        },
        {
            "variant_id": "v08_disagreement",
            "name": "Feature set: disagreement",
            "params": {
                "surv_scalar_mode": "disagreement",
            },
        },
        {
            "variant_id": "v09_qhat_censor",
            "name": "Feature set: qhat + censor",
            "params": {
                "surv_scalar_mode": "qhat_censor",
            },
        },
        {
            "variant_id": "v10_qhat_disagree",
            "name": "Feature set: qhat + disagreement",
            "params": {
                "surv_scalar_mode": "qhat_disagree",
            },
        },
        {
            "variant_id": "v11_all_lite",
            "name": "Feature set: all lite",
            "params": {
                "surv_scalar_mode": "all_lite",
            },
        },
    ]


def _serialize_params(params: dict[str, object]) -> str:
    serializable: dict[str, object] = {}
    for key, value in params.items():
        serializable[key] = list(value) if isinstance(value, tuple) else value
    return json.dumps(serializable, sort_keys=True)


def _materialize_params(params: dict[str, object], *, args: argparse.Namespace, horizon: float) -> dict[str, object]:
    out = _base_balanced_kwargs(args)
    out.update(params)
    out["horizon"] = float(horizon)
    if isinstance(out.get("y_res_clip_percentiles"), list):
        out["y_res_clip_percentiles"] = tuple(out["y_res_clip_percentiles"])
    return out


def _evaluate_case_specs(
    case_spec: dict[str, object],
    variant_specs: list[dict[str, object]],
    *,
    args_dict: dict[str, object],
) -> list[dict[str, object]]:
    args = argparse.Namespace(**args_dict)
    case = prepare_case(case_spec, target=args.target, horizon_quantile=args.horizon_quantile)
    x = np.asarray(case.X, dtype=float)
    w = np.asarray(case.W, dtype=float)
    z = np.asarray(case.Z, dtype=float)
    a = np.asarray(case.A, dtype=float)
    time_obs = np.asarray(case.Y, dtype=float)
    event = np.asarray(case.delta, dtype=float)

    rows: list[dict[str, object]] = []
    for spec in variant_specs:
        model_kwargs = _materialize_params(spec["params"], args=args, horizon=case.horizon)
        model = FinalModelCensoredSurvivalForest(**model_kwargs)
        t0 = time.time()
        model.fit_components(x, a, time_obs, event, z, w)
        preds = np.asarray(model.effect_from_components(x, w, z), dtype=float).ravel()
        row = _evaluate_predictions(
            str(spec["name"]),
            preds,
            case.true_cate,
            time.time() - t0,
            backend=model.__class__.__name__,
        )
        row["variant_id"] = str(spec["variant_id"])
        row["params_json"] = _serialize_params(spec["params"])
        row["horizon"] = float(case.horizon)
        row["case_id"] = int(case_spec["case_id"])
        row["case_slug"] = str(case_spec["slug"])
        rows.append(row)
        del model
        gc.collect()
    return rows


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["variant_id", "name", "params_json"], as_index=False)
        .agg(
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_bias=("bias", "mean"),
            avg_abs_bias=("abs_bias", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_sign_acc=("sign_acc", "mean"),
            avg_time=("total_time", "mean"),
            n_cases=("case_id", "count"),
        )
        .reset_index(drop=True)
    )
    summary["rmse_rank"] = summary["avg_rmse"].rank(method="min")
    summary["mae_rank"] = summary["avg_mae"].rank(method="min")
    summary["abs_bias_rank"] = summary["avg_abs_bias"].rank(method="min")
    summary["signed_bias_mag_rank"] = summary["avg_bias"].abs().rank(method="min")
    summary["composite_rank"] = (
        summary["rmse_rank"] + summary["mae_rank"] + summary["abs_bias_rank"]
    ) / 3.0
    summary = summary.sort_values(
        ["composite_rank", "avg_rmse", "avg_mae", "avg_abs_bias", "avg_pearson"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = set(args.case_ids) if args.case_ids else None
    case_specs = [
        spec for spec in CASE_SPECS
        if case_ids is None or int(spec["case_id"]) in case_ids
    ]
    variant_specs = _variant_specs()

    args_dict = {
        "target": args.target,
        "horizon_quantile": args.horizon_quantile,
        "random_state": args.random_state,
    }

    rows: list[dict[str, object]] = []
    print(
        f"[feature-search] variants={len(variant_specs)} cases={len(case_specs)} workers={args.workers}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _evaluate_case_specs,
                case_spec,
                variant_specs,
                args_dict=args_dict,
            ): case_spec
            for case_spec in case_specs
        }
        for future in as_completed(futures):
            case_spec = futures[future]
            case_rows = future.result()
            case_df = pd.DataFrame(case_rows)
            case_df.to_csv(
                output_dir / f"case_{int(case_spec['case_id']):02d}_{case_spec['slug']}.csv",
                index=False,
            )
            rows.extend(case_rows)
            print(
                f"[feature-search] completed case_id={int(case_spec['case_id'])} slug={case_spec['slug']} "
                f"rows={len(case_rows)}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    results["abs_bias"] = results["bias"].abs()
    results.to_csv(output_dir / "results_full.csv", index=False)
    summary = _summarize(results)
    summary.to_csv(output_dir / "summary_full.csv", index=False)
    (output_dir / "specs.json").write_text(json.dumps(variant_specs, indent=2))
    print(
        "[feature-search] top="
        f"{summary[['variant_id', 'avg_rmse', 'avg_mae', 'avg_abs_bias']].head(5).to_dict(orient='records')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
