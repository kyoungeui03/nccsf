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

from grf.non_censored.benchmarks import (  # noqa: E402
    CASE_SPECS,
    VARIANT_SPECS,
    _build_case,
    _evaluate_case_variant,
    _make_cfg,
    _metric_row,
)
from grf.non_censored.models import (  # noqa: E402
    BestCurveLocalNCCausalForest,
    MildShrinkNCCausalForestDML,
    OldC3AugmentedDMLExtraHNCCausalForest,
    OldC3AugmentedDMLNCCausalForest,
    SinglePassAugmentedFullNCCausalForest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run basic12 non-censored finalization study for single-pass Augmented C3."
    )
    parser.add_argument(
        "--phase",
        choices=["structure", "qh"],
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "non_censored" / "outputs" / "benchmark_basic12_single_pass_augmented_finalization",
    )
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p-w", type=int, default=1)
    parser.add_argument("--p-z", type=int, default=1)
    parser.add_argument("--nuisance-feature-mode", type=str, default="dup")
    parser.add_argument("--prediction-nuisance-mode", type=str, default="fold_ensemble")
    return parser.parse_args()


def _evaluate_b2(case):
    variant = next(v for v in VARIANT_SPECS if v["name"] == "EconML Baseline")
    return _evaluate_case_variant(case, variant, seed=case["cfg"].seed)


def _evaluate_old_c3(case):
    x_final = MildShrinkNCCausalForestDML.stack_final_features(case["X"], case["W"], case["Z"])
    start = time.time()
    model = MildShrinkNCCausalForestDML(
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=42,
        x_core_dim=case["X"].shape[1],
    )
    model.fit_nc(x_final, case["A"], case["Y"], case["Z"], case["W"])
    preds = model.effect(x_final).ravel()
    return _metric_row("Old C3", preds, case["true_cate"], time.time() - start)


def _evaluate_new_c3(case):
    start = time.time()
    model = BestCurveLocalNCCausalForest()
    model.fit_components(case["X"], case["A"], case["Y"], case["Z"], case["W"])
    preds = model.effect_from_components(case["X"], case["W"], case["Z"]).ravel()
    return _metric_row("New C3", preds, case["true_cate"], time.time() - start)


def _evaluate_model(case, name, factory):
    start = time.time()
    model = factory()
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


def _structure_candidates():
    return [
        (
            "SP-AugFull dup refit",
            lambda: SinglePassAugmentedFullNCCausalForest(
                nuisance_feature_mode="dup",
                prediction_nuisance_mode="full_refit",
            ),
        ),
        (
            "SP-AugFull dup ensemble",
            lambda: SinglePassAugmentedFullNCCausalForest(
                nuisance_feature_mode="dup",
                prediction_nuisance_mode="fold_ensemble",
            ),
        ),
        (
            "SP-AugFull broaddup refit",
            lambda: SinglePassAugmentedFullNCCausalForest(
                nuisance_feature_mode="broad_dup",
                prediction_nuisance_mode="full_refit",
            ),
        ),
        (
            "SP-AugFull broaddup ensemble",
            lambda: SinglePassAugmentedFullNCCausalForest(
                nuisance_feature_mode="broad_dup",
                prediction_nuisance_mode="fold_ensemble",
            ),
        ),
    ]


def _qh_candidates(args):
    shared = dict(
        nuisance_feature_mode=args.nuisance_feature_mode,
        prediction_nuisance_mode=args.prediction_nuisance_mode,
    )
    return [
        (
            "SP-AugFull logit + extra(800,5)",
            lambda: SinglePassAugmentedFullNCCausalForest(
                q_kind="logit",
                h_kind="extra",
                h_n_estimators=800,
                h_min_samples_leaf=5,
                **shared,
            ),
        ),
        (
            "SP-AugFull logit + extra(400,10)",
            lambda: SinglePassAugmentedFullNCCausalForest(
                q_kind="logit",
                h_kind="extra",
                h_n_estimators=400,
                h_min_samples_leaf=10,
                **shared,
            ),
        ),
        (
            "SP-AugFull logit + rf(300,20)",
            lambda: SinglePassAugmentedFullNCCausalForest(
                q_kind="logit",
                h_kind="rf",
                h_n_estimators=300,
                h_min_samples_leaf=20,
                **shared,
            ),
        ),
        (
            "SP-AugFull poly2 + extra(800,5)",
            lambda: SinglePassAugmentedFullNCCausalForest(
                q_kind="poly2",
                h_kind="extra",
                h_n_estimators=800,
                h_min_samples_leaf=5,
                **shared,
            ),
        ),
        (
            "SP-AugFull rf-q + extra(800,5)",
            lambda: SinglePassAugmentedFullNCCausalForest(
                q_kind="rf",
                h_kind="extra",
                h_n_estimators=800,
                h_min_samples_leaf=5,
                **shared,
            ),
        ),
        (
            "SP-AugFull logit + extra(1200,3)",
            lambda: SinglePassAugmentedFullNCCausalForest(
                q_kind="logit",
                h_kind="extra",
                h_n_estimators=1200,
                h_min_samples_leaf=3,
                **shared,
            ),
        ),
    ]


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = set(args.case_ids) if args.case_ids else None
    selected_cases = [case for case in CASE_SPECS if selected_ids is None or case["case_id"] in selected_ids]

    if args.phase == "structure":
        candidate_specs = _structure_candidates()
        model_order = [
            "EconML Baseline",
            "Old C3",
            "New C3",
            "Augmented-DML (PCI)",
            "Augmented-DML + Extra h (PCI)",
        ] + [name for name, _ in candidate_specs]
    else:
        candidate_specs = _qh_candidates(args)
        structure_label = f"[{args.nuisance_feature_mode}|{args.prediction_nuisance_mode}]"
        candidate_specs = [(f"{name} {structure_label}", factory) for name, factory in candidate_specs]
        model_order = [
            "EconML Baseline",
            "Old C3",
            "New C3",
            "Augmented-DML (PCI)",
            "Augmented-DML + Extra h (PCI)",
        ] + [name for name, _ in candidate_specs]

    case_frames = []
    for case_spec in selected_cases:
        cfg = _make_cfg(case_spec)
        cfg.n = int(args.n)
        cfg.p_w = int(args.p_w)
        cfg.p_z = int(args.p_z)
        case = _build_case(cfg, case_spec)
        print("=" * 100, flush=True)
        print(f"[{args.phase}] Running case {case_spec['case_id']:02d}: {case_spec['slug']}", flush=True)
        print("=" * 100, flush=True)

        rows = [
            _evaluate_b2(case),
            _evaluate_old_c3(case),
            _evaluate_new_c3(case),
            _evaluate_model(case, "Augmented-DML (PCI)", OldC3AugmentedDMLNCCausalForest),
            _evaluate_model(case, "Augmented-DML + Extra h (PCI)", OldC3AugmentedDMLExtraHNCCausalForest),
        ]
        for name, factory in candidate_specs:
            rows.append(_evaluate_model(case, name, factory))

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
        partial_df["model_order"] = partial_df["name"].map({name: i for i, name in enumerate(model_order)})
        partial_df = partial_df.sort_values(["case_id", "model_order"]).drop(columns=["model_order"])
        partial_df.to_csv(output_dir / f"{args.phase}_results_{args.tag}.csv", index=False)

    combined_df = pd.concat(case_frames, ignore_index=True)
    combined_df["model_order"] = combined_df["name"].map({name: i for i, name in enumerate(model_order)})
    combined_df = combined_df.sort_values(["case_id", "model_order"]).drop(columns=["model_order"])
    results_path = output_dir / f"{args.phase}_results_{args.tag}.csv"
    summary_path = output_dir / f"{args.phase}_summary_{args.tag}.csv"
    combined_df.to_csv(results_path, index=False)
    _summarize(combined_df).to_csv(summary_path, index=False)
    print(f"Saved {results_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
