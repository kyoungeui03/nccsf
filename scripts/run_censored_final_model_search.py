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
            "Search over promising censored Final Model variants while keeping the "
            "EconML final learner and overall Final Model structure fixed."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--target", default="RMST")
    parser.add_argument("--horizon-quantile", type=float, default=0.60)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--workers", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    parser.add_argument("--top-stage1", type=int, default=4)
    parser.add_argument("--top-stage2", type=int, default=4)
    parser.add_argument("--top-stage3", type=int, default=4)
    parser.add_argument("--top-stage4", type=int, default=4)
    parser.add_argument("--top-stage5", type=int, default=2)
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / "benchmark_censored_final_model_search_basic12").resolve()


def _reference_results_path() -> Path:
    return PROJECT_ROOT / "outputs" / "benchmark_final_model_bundle_basic12_option_b" / "censored" / "results_full.csv"


def _base_model_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "target": args.target,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf,
        "nuisance_feature_mode": "broad_dup",
        "prediction_nuisance_mode": "full_refit",
        "include_raw_proxy": True,
        "include_surv_scalar": True,
        "surv_scalar_mode": "pair",
        "q_kind": "logit",
        "h_kind": "extra",
        "h_n_estimators": 600,
        "h_min_samples_leaf": 5,
        "q_clip": 0.03,
        "y_tilde_clip_quantile": 0.98,
        "y_res_clip_percentiles": (2.0, 98.0),
        "censoring_estimator": "nelson-aalen",
        "event_survival_estimator": "cox",
        "m_pred_mode": "bridge",
        "survival_forest_num_trees": 100,
        "survival_forest_min_node_size": 10,
        "survival_fast_logrank": False,
        "enforce_finite_horizon": True,
        "n_jobs": 1,
    }


def _stage1_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    idx = 0
    for event_survival_estimator in ("cox", "survival_forest"):
        for censoring_estimator in ("kaplan-meier", "nelson-aalen", "cox", "survival_forest"):
            for m_pred_mode in ("bridge", "survival", "blend_0.25", "blend_0.50", "blend_0.75"):
                idx += 1
                specs.append(
                    {
                        "variant_id": f"s1_{idx:02d}",
                        "stage": "stage1_core",
                        "name": (
                            f"S1 E={event_survival_estimator} "
                            f"C={censoring_estimator} "
                            f"M={m_pred_mode}"
                        ),
                        "params": {
                            "event_survival_estimator": event_survival_estimator,
                            "censoring_estimator": censoring_estimator,
                            "m_pred_mode": m_pred_mode,
                        },
                    }
                )
    return specs


def _stage2_specs(stage1_summary: pd.DataFrame, *, top_k: int) -> list[dict[str, object]]:
    top = stage1_summary.nsmallest(top_k, "composite_rank")
    specs: list[dict[str, object]] = []
    idx = 0
    for _, base in top.iterrows():
        base_params = json.loads(base["params_json"])
        for q_kind in ("logit", "poly2", "hgb", "rf"):
            for h_kind in ("extra", "hgb", "rf"):
                for surv_scalar_mode in ("pair", "full"):
                    idx += 1
                    params = dict(base_params)
                    params.update(
                        {
                            "q_kind": q_kind,
                            "h_kind": h_kind,
                            "surv_scalar_mode": surv_scalar_mode,
                        }
                    )
                    specs.append(
                        {
                            "variant_id": f"s2_{idx:03d}",
                            "stage": "stage2_models",
                            "parent_variant_id": str(base["variant_id"]),
                            "name": (
                                f"S2 parent={base['variant_id']} "
                                f"Q={q_kind} H={h_kind} S={surv_scalar_mode}"
                            ),
                            "params": params,
                        }
                    )
    return specs


def _stage3_specs(stage2_summary: pd.DataFrame, *, top_k: int) -> list[dict[str, object]]:
    top = stage2_summary.nsmallest(top_k, "composite_rank")
    specs: list[dict[str, object]] = []
    idx = 0
    for _, base in top.iterrows():
        base_params = json.loads(base["params_json"])
        for q_trees in (300, 600):
            for q_min_samples_leaf in (10, 20):
                for h_n_estimators in (600, 1200):
                    for h_min_samples_leaf in (3, 5):
                        for prediction_nuisance_mode in ("full_refit", "fold_ensemble"):
                            idx += 1
                            params = dict(base_params)
                            params.update(
                                {
                                    "q_trees": q_trees,
                                    "q_min_samples_leaf": q_min_samples_leaf,
                                    "h_n_estimators": h_n_estimators,
                                    "h_min_samples_leaf": h_min_samples_leaf,
                                    "prediction_nuisance_mode": prediction_nuisance_mode,
                                }
                            )
                            specs.append(
                                {
                                    "variant_id": f"s3_{idx:03d}",
                                    "stage": "stage3_nuisance_tuning",
                                    "parent_variant_id": str(base["variant_id"]),
                                    "name": (
                                        f"S3 parent={base['variant_id']} "
                                        f"qt={q_trees} qleaf={q_min_samples_leaf} "
                                        f"ht={h_n_estimators} hleaf={h_min_samples_leaf} "
                                        f"pred={prediction_nuisance_mode}"
                                    ),
                                    "params": params,
                                }
                            )
    return specs


def _stage4_specs(stage3_summary: pd.DataFrame, *, top_k: int) -> list[dict[str, object]]:
    top = stage3_summary.nsmallest(top_k, "composite_rank")
    specs: list[dict[str, object]] = []
    idx = 0
    for _, base in top.iterrows():
        base_params = json.loads(base["params_json"])
        for q_clip in (0.02, 0.03, 0.05):
            for y_tilde_clip_quantile in (0.97, 0.98):
                for y_res_clip_percentiles in ((1.0, 99.0), (2.0, 98.0)):
                    idx += 1
                    params = dict(base_params)
                    params.update(
                        {
                            "q_clip": q_clip,
                            "y_tilde_clip_quantile": y_tilde_clip_quantile,
                            "y_res_clip_percentiles": tuple(y_res_clip_percentiles),
                        }
                    )
                    specs.append(
                        {
                            "variant_id": f"s4_{idx:03d}",
                            "stage": "stage4_clipping",
                            "parent_variant_id": str(base["variant_id"]),
                            "name": (
                                f"S4 parent={base['variant_id']} "
                                f"qclip={q_clip:.2f} "
                                f"yt={y_tilde_clip_quantile:.2f} "
                                f"yr={y_res_clip_percentiles}"
                            ),
                            "params": params,
                        }
                    )
    return specs


def _stage5_specs(stage4_summary: pd.DataFrame, *, top_k: int) -> list[dict[str, object]]:
    top = stage4_summary.nsmallest(top_k, "composite_rank")
    specs: list[dict[str, object]] = []
    idx = 0
    for _, base in top.iterrows():
        base_params = json.loads(base["params_json"])
        for n_estimators in (200, 400):
            for min_samples_leaf in (10, 20):
                for cv in (3, 5):
                    idx += 1
                    params = dict(base_params)
                    params.update(
                        {
                            "n_estimators": n_estimators,
                            "min_samples_leaf": min_samples_leaf,
                            "cv": cv,
                        }
                    )
                    specs.append(
                        {
                            "variant_id": f"s5_{idx:03d}",
                            "stage": "stage5_final_forest_tuning",
                            "parent_variant_id": str(base["variant_id"]),
                            "name": (
                                f"S5 parent={base['variant_id']} "
                                f"nt={n_estimators} "
                                f"leaf={min_samples_leaf} "
                                f"cv={cv}"
                            ),
                            "params": params,
                        }
                    )
    return specs


def _stage6_specs(stage3_summary: pd.DataFrame, *, top_k: int) -> list[dict[str, object]]:
    sf_top = stage3_summary[
        stage3_summary["params_json"].str.contains("survival_forest")
    ].nsmallest(top_k, "composite_rank")
    specs: list[dict[str, object]] = []
    idx = 0
    for _, base in sf_top.iterrows():
        base_params = json.loads(base["params_json"])
        for num_trees in (50, 100, 200):
            for min_node_size in (10, 15):
                for fast_logrank in (False, True):
                    idx += 1
                    params = dict(base_params)
                    params.update(
                        {
                            "survival_forest_num_trees": num_trees,
                            "survival_forest_min_node_size": min_node_size,
                            "survival_fast_logrank": fast_logrank,
                        }
                    )
                    specs.append(
                        {
                            "variant_id": f"s6_{idx:03d}",
                            "stage": "stage6_survival_forest_tuning",
                            "parent_variant_id": str(base["variant_id"]),
                            "name": (
                                f"S6 parent={base['variant_id']} "
                                f"nt={num_trees} "
                                f"node={min_node_size} "
                                f"logrank={int(fast_logrank)}"
                            ),
                            "params": params,
                        }
                    )
    return specs


def _serialize_params(params: dict[str, object]) -> str:
    serializable: dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            serializable[key] = list(value)
        else:
            serializable[key] = value
    return json.dumps(serializable, sort_keys=True)


def _materialize_params(params: dict[str, object], *, args: argparse.Namespace, horizon: float) -> dict[str, object]:
    out = _base_model_kwargs(args)
    out.update(params)
    out["horizon"] = float(horizon)
    if isinstance(out.get("y_res_clip_percentiles"), list):
        out["y_res_clip_percentiles"] = tuple(out["y_res_clip_percentiles"])
    return out


def _evaluate_case_specs(
    case_spec: dict[str, object],
    stage_specs: list[dict[str, object]],
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
    for spec in stage_specs:
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
        row["stage"] = str(spec["stage"])
        row["parent_variant_id"] = str(spec.get("parent_variant_id", ""))
        row["params_json"] = _serialize_params(spec["params"])
        row["horizon"] = float(case.horizon)
        row["case_id"] = int(case_spec["case_id"])
        row["case_slug"] = str(case_spec["slug"])
        rows.append(row)
        del model
        gc.collect()
    return rows


def _run_stage(
    stage_name: str,
    stage_specs: list[dict[str, object]],
    *,
    case_specs: list[dict[str, object]],
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    stage_dir = output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[search] stage={stage_name} variants={len(stage_specs)} cases={len(case_specs)} workers={args.workers}",
        flush=True,
    )

    args_dict = {
        "target": args.target,
        "horizon_quantile": args.horizon_quantile,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf,
    }
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _evaluate_case_specs,
                case_spec,
                stage_specs,
                args_dict=args_dict,
            ): case_spec
            for case_spec in case_specs
        }
        for future in as_completed(futures):
            case_spec = futures[future]
            case_rows = future.result()
            case_df = pd.DataFrame(case_rows)
            case_df.to_csv(
                stage_dir / f"case_{int(case_spec['case_id']):02d}_{case_spec['slug']}.csv",
                index=False,
            )
            rows.extend(case_rows)
            print(
                f"[search] stage={stage_name} completed case_id={int(case_spec['case_id'])} "
                f"slug={case_spec['slug']} rows={len(case_rows)}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    results["abs_bias"] = results["bias"].abs()
    results.to_csv(stage_dir / "results_full.csv", index=False)
    summary = _summarize_stage(results)
    summary.to_csv(stage_dir / "summary_full.csv", index=False)
    (stage_dir / "specs.json").write_text(json.dumps(stage_specs, indent=2))
    print(
        "[search] stage="
        f"{stage_name} top="
        f"{summary[['variant_id', 'avg_rmse', 'avg_mae', 'avg_abs_bias']].head(3).to_dict(orient='records')}",
        flush=True,
    )
    return summary


def _summarize_stage(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["variant_id", "name", "stage", "parent_variant_id", "params_json"], as_index=False)
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
    summary["composite_rank"] = (
        summary["rmse_rank"] + summary["mae_rank"] + summary["abs_bias_rank"]
    ) / 3.0
    summary = summary.sort_values(
        ["composite_rank", "avg_rmse", "avg_mae", "avg_abs_bias", "avg_pearson"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def _load_reference_metrics() -> pd.DataFrame:
    ref_path = _reference_results_path()
    if not ref_path.exists():
        return pd.DataFrame()
    ref = pd.read_csv(ref_path)
    ref["abs_bias"] = ref["bias"].abs()
    summary = (
        ref.groupby("name", as_index=False)
        .agg(
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_bias=("bias", "mean"),
            avg_abs_bias=("abs_bias", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_sign_acc=("sign_acc", "mean"),
            avg_time=("total_time", "mean"),
        )
        .assign(source="stored_option_b_reference")
    )
    return summary


def _attach_reference_comparison(final_summary: pd.DataFrame, reference_summary: pd.DataFrame) -> pd.DataFrame:
    out = final_summary.copy()
    ref_map = {
        row["name"]: row
        for _, row in reference_summary.iterrows()
    }
    final_ref = ref_map.get("Final Model")
    rcsf_ref = ref_map.get("R-CSF Baseline")
    for label, ref_row in (("final_ref", final_ref), ("rcsf_ref", rcsf_ref)):
        if ref_row is None:
            continue
        out[f"better_rmse_than_{label}"] = out["avg_rmse"] < float(ref_row["avg_rmse"])
        out[f"better_mae_than_{label}"] = out["avg_mae"] < float(ref_row["avg_mae"])
        out[f"better_abs_bias_than_{label}"] = out["avg_abs_bias"] < float(ref_row["avg_abs_bias"])
        out[f"better_signed_bias_mag_than_{label}"] = out["avg_bias"].abs() < abs(float(ref_row["avg_bias"]))
        out[f"better_all3_than_{label}"] = (
            out[f"better_rmse_than_{label}"]
            & out[f"better_mae_than_{label}"]
            & out[f"better_abs_bias_than_{label}"]
        )
    return out


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = set(args.case_ids) if args.case_ids else None
    case_specs = [
        spec for spec in CASE_SPECS
        if case_ids is None or int(spec["case_id"]) in case_ids
    ]

    metadata = {
        "target": args.target,
        "horizon_quantile": args.horizon_quantile,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf,
        "workers": args.workers,
        "n_cases": len(case_specs),
    }
    (output_dir / "search_metadata.json").write_text(json.dumps(metadata, indent=2))

    stage1 = _run_stage("stage1_core", _stage1_specs(), case_specs=case_specs, args=args, output_dir=output_dir)
    stage2_specs = _stage2_specs(stage1, top_k=args.top_stage1)
    stage2 = _run_stage("stage2_models", stage2_specs, case_specs=case_specs, args=args, output_dir=output_dir)
    stage3_specs = _stage3_specs(stage2, top_k=args.top_stage2)
    stage3 = _run_stage("stage3_nuisance_tuning", stage3_specs, case_specs=case_specs, args=args, output_dir=output_dir)
    stage4_specs = _stage4_specs(stage3, top_k=args.top_stage3)
    stage4 = _run_stage("stage4_clipping", stage4_specs, case_specs=case_specs, args=args, output_dir=output_dir)
    stage5_specs = _stage5_specs(stage4, top_k=args.top_stage4)
    stage5 = _run_stage(
        "stage5_final_forest_tuning",
        stage5_specs,
        case_specs=case_specs,
        args=args,
        output_dir=output_dir,
    )
    stage6_specs = _stage6_specs(stage3, top_k=args.top_stage5)
    if stage6_specs:
        stage6 = _run_stage(
            "stage6_survival_forest_tuning",
            stage6_specs,
            case_specs=case_specs,
            args=args,
            output_dir=output_dir,
        )
    else:
        stage6 = pd.DataFrame()

    overall = pd.concat([stage1, stage2, stage3, stage4, stage5, stage6], ignore_index=True)
    overall = overall.sort_values(
        ["composite_rank", "avg_rmse", "avg_mae", "avg_abs_bias", "avg_pearson"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)
    overall["global_rank"] = np.arange(1, len(overall) + 1)

    ref_summary = _load_reference_metrics()
    compared = _attach_reference_comparison(overall, ref_summary)
    compared.to_csv(output_dir / "overall_search_summary.csv", index=False)
    if not ref_summary.empty:
        ref_summary.to_csv(output_dir / "reference_summary.csv", index=False)


if __name__ == "__main__":
    main()
