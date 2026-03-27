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
    _BaseSinglePassBridgeFeatureCensoredSurvivalForest,
)


REFERENCE_MODELS = [
    ("UnifiedB2SumBaseline (Cens)", UnifiedB2SumBaselineCensoredSurvivalForest, {}),
    ("UnifiedB2SumMildShrink (Cens)", UnifiedB2SumMildShrinkCensoredSurvivalForest, {}),
    ("UnifiedB2SumSinglePassBaseline (Cens)", UnifiedB2SumSinglePassBaselineCensoredSurvivalForest, {}),
]

Q_PROFILES = [
    ("q=logit", {"q_kind": "logit"}),
    ("q=poly2", {"q_kind": "poly2", "q_poly_degree": 2}),
    ("q=rf100x20", {"q_kind": "rf", "q_trees": 100, "q_min_samples_leaf": 20}),
]

H_PROFILES = [
    ("h=rf100x20", {"h_kind": "rf", "h_n_estimators": 100, "h_min_samples_leaf": 20}),
    ("h=extra200x10", {"h_kind": "extra", "h_n_estimators": 200, "h_min_samples_leaf": 10}),
    ("h=extra400x10", {"h_kind": "extra", "h_n_estimators": 400, "h_min_samples_leaf": 10}),
    ("h=extra600x5", {"h_kind": "extra", "h_n_estimators": 600, "h_min_samples_leaf": 5}),
    ("h=extra1200x3", {"h_kind": "extra", "h_n_estimators": 1200, "h_min_samples_leaf": 3}),
]

SURV_MODES = ["pair", "full"]
CENSORING_ESTIMATORS = ["nelson-aalen", "kaplan-meier"]
CLIP_PROFILES = [
    ("clip=base", {"q_clip": 0.02, "y_tilde_clip_quantile": 0.99, "y_res_clip_percentiles": (1.0, 99.0)}),
    ("clip=strong", {"q_clip": 0.03, "y_tilde_clip_quantile": 0.98, "y_res_clip_percentiles": (2.0, 98.0)}),
]
FOREST_PROFILES = [
    ("forest=200x20", {"n_estimators": 200, "min_samples_leaf": 20}),
    ("forest=300x10", {"n_estimators": 300, "min_samples_leaf": 10}),
]


def build_model_specs() -> list[tuple[str, type, dict[str, object]]]:
    specs: list[tuple[str, type, dict[str, object]]] = list(REFERENCE_MODELS)
    for surv_mode in SURV_MODES:
        for censoring_estimator in CENSORING_ESTIMATORS:
            for q_label, q_kwargs in Q_PROFILES:
                for h_label, h_kwargs in H_PROFILES:
                    for clip_label, clip_kwargs in CLIP_PROFILES:
                        for forest_label, forest_kwargs in FOREST_PROFILES:
                            kwargs: dict[str, object] = {
                                "include_raw_proxy": True,
                                "include_surv_scalar": True,
                                "surv_scalar_mode": surv_mode,
                                "prediction_nuisance_mode": "full_refit",
                                "observed_only": False,
                                "target": "RMST",
                                "horizon": None,
                                "cv": 5,
                                "random_state": 42,
                                "censoring_estimator": censoring_estimator,
                                "n_jobs": 1,
                                "nuisance_feature_mode": "broad_dup",
                            }
                            kwargs.update(q_kwargs)
                            kwargs.update(h_kwargs)
                            kwargs.update(clip_kwargs)
                            kwargs.update(forest_kwargs)
                            name = (
                                f"SPv2C[{surv_mode}|{censoring_estimator}|{q_label}|"
                                f"{h_label}|{clip_label}|{forest_label}]"
                            )
                            specs.append((name, _BaseSinglePassBridgeFeatureCensoredSurvivalForest, kwargs))
    return specs


MODEL_SPECS = build_model_specs()


def parse_args():
    parser = argparse.ArgumentParser(description="Run basic12 censored single-pass v2 search.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--name-contains", type=str)
    parser.add_argument("--allow-names-file", type=Path)
    parser.add_argument("--max-models", type=int)
    parser.add_argument("--grid-only", action="store_true")
    parser.add_argument("--target", choices=["RMST", "survival.probability"], default="RMST")
    parser.add_argument("--horizon-quantile", type=float, default=0.60)
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
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def metric_row(name: str, preds, case, case_spec, *, target: str, elapsed: float, backend: str, model_kwargs: dict[str, object]) -> dict[str, object]:
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
        model_kwargs=repr(model_kwargs),
    )
    return row


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.case_ids) if args.case_ids else None
    case_specs = [case for case in CASE_SPECS if selected is None or int(case["case_id"]) in selected]
    model_specs = MODEL_SPECS
    if args.grid_only:
        model_specs = model_specs[len(REFERENCE_MODELS):]
    if args.name_contains:
        needle = args.name_contains.lower()
        model_specs = [spec for spec in model_specs if needle in spec[0].lower()]
    if args.allow_names_file:
        allowed = {
            line.strip()
            for line in args.allow_names_file.read_text().splitlines()
            if line.strip()
        }
        model_specs = [spec for spec in model_specs if spec[0] in allowed]
    if args.max_models is not None:
        model_specs = model_specs[: int(args.max_models)]

    frames = []
    for case_spec in case_specs:
        case = prepare_case(case_spec, target=args.target, horizon_quantile=args.horizon_quantile)
        rows = []
        horizon = None if args.target == "RMST" else float(case.horizon)
        total_models = len(model_specs)
        print(f"[case {case_spec['case_id']:02d}] start: {case_spec['slug']} with {total_models} models", flush=True)
        for model_index, (name, model_cls, model_kwargs) in enumerate(model_specs, start=1):
            print(f"[case {case_spec['case_id']:02d}] model {model_index}/{total_models}: {name}", flush=True)
            start = time.time()
            run_kwargs = dict(model_kwargs)
            run_kwargs["target"] = args.target
            run_kwargs["horizon"] = horizon
            model = model_cls(**run_kwargs)
            model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
            preds = model.effect_from_components(case.X, case.W, case.Z).ravel()
            rows.append(
                metric_row(
                    name,
                    preds,
                    case,
                    case_spec,
                    target=args.target,
                    elapsed=time.time() - start,
                    backend=model.__class__.__name__,
                    model_kwargs=run_kwargs,
                )
            )
            partial_case_df = pd.DataFrame(rows).sort_values(["case_id", "name"]).reset_index(drop=True)
            pd.concat(frames + [partial_case_df], ignore_index=True).to_csv(args.output_dir / f"results_{args.tag}.csv", index=False)

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
