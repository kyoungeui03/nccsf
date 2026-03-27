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
    BestCurveLocalNCCausalForest,
    MildShrinkNCCausalForestDML,
    OldC3AugmentedDMLNCCausalForest,
    OldC3AugmentedGRFNCCausalForest,
    OldC3AugmentedObservedDMLNCCausalForest,
    OldC3AugmentedObservedGRFNCCausalForest,
    OldC3SummaryDMLNCCausalForest,
    OldC3SummaryGRFNCCausalForest,
    OldC3SummaryObservedDMLNCCausalForest,
    OldC3SummaryObservedGRFNCCausalForest,
)


MODEL_ORDER = [
    "Old C3",
    "New C3",
    "Summary-DML (PCI)",
    "Summary-DML (no PCI)",
    "Augmented-DML (PCI)",
    "Augmented-DML (no PCI)",
    "Summary-GRF (PCI)",
    "Summary-GRF (no PCI)",
    "Augmented-GRF (PCI)",
    "Augmented-GRF (no PCI)",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run non-censored Old C3 ablation models on the basic 12 synthetic cases.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "non_censored" / "outputs" / "benchmark_basic12_oldc3_ablation",
    )
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p-w", type=int, default=1)
    parser.add_argument("--p-z", type=int, default=1)
    return parser.parse_args()


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


def _evaluate_summary_dml(case, *, observed_only: bool):
    start = time.time()
    model_cls = OldC3SummaryObservedDMLNCCausalForest if observed_only else OldC3SummaryDMLNCCausalForest
    model = model_cls()
    model.fit_components(case["X"], case["A"], case["Y"], case["Z"], case["W"])
    preds = model.effect_from_components(case["X"], case["W"], case["Z"]).ravel()
    name = "Summary-DML (no PCI)" if observed_only else "Summary-DML (PCI)"
    return _metric_row(name, preds, case["true_cate"], time.time() - start)


def _evaluate_augmented_dml(case, *, observed_only: bool):
    start = time.time()
    model_cls = OldC3AugmentedObservedDMLNCCausalForest if observed_only else OldC3AugmentedDMLNCCausalForest
    model = model_cls()
    model.fit_components(case["X"], case["A"], case["Y"], case["Z"], case["W"])
    preds = model.effect_from_components(case["X"], case["W"], case["Z"]).ravel()
    name = "Augmented-DML (no PCI)" if observed_only else "Augmented-DML (PCI)"
    return _metric_row(name, preds, case["true_cate"], time.time() - start)


def _evaluate_summary_grf(case, *, observed_only: bool):
    start = time.time()
    model_cls = OldC3SummaryObservedGRFNCCausalForest if observed_only else OldC3SummaryGRFNCCausalForest
    model = model_cls()
    model.fit_components(case["X"], case["A"], case["Y"], case["Z"], case["W"])
    preds = model.effect_from_components(case["X"], case["W"], case["Z"]).ravel()
    name = "Summary-GRF (no PCI)" if observed_only else "Summary-GRF (PCI)"
    return _metric_row(name, preds, case["true_cate"], time.time() - start)


def _evaluate_augmented_grf(case, *, observed_only: bool):
    start = time.time()
    model_cls = OldC3AugmentedObservedGRFNCCausalForest if observed_only else OldC3AugmentedGRFNCCausalForest
    model = model_cls()
    model.fit_components(case["X"], case["A"], case["Y"], case["Z"], case["W"])
    preds = model.effect_from_components(case["X"], case["W"], case["Z"]).ravel()
    name = "Augmented-GRF (no PCI)" if observed_only else "Augmented-GRF (PCI)"
    return _metric_row(name, preds, case["true_cate"], time.time() - start)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = set(args.case_ids) if args.case_ids else None
    selected_cases = [case for case in CASE_SPECS if selected_ids is None or case["case_id"] in selected_ids]

    case_frames = []
    for case_spec in selected_cases:
        cfg = _make_cfg(case_spec)
        cfg.n = int(args.n)
        cfg.p_w = int(args.p_w)
        cfg.p_z = int(args.p_z)
        case = _build_case(cfg, case_spec)
        print("=" * 100, flush=True)
        print(f"Running case {case_spec['case_id']:02d}: {case_spec['slug']}", flush=True)
        print("=" * 100, flush=True)

        rows = [
            _evaluate_old_c3(case),
            _evaluate_new_c3(case),
            _evaluate_summary_dml(case, observed_only=False),
            _evaluate_summary_dml(case, observed_only=True),
            _evaluate_augmented_dml(case, observed_only=False),
            _evaluate_augmented_dml(case, observed_only=True),
            _evaluate_summary_grf(case, observed_only=False),
            _evaluate_summary_grf(case, observed_only=True),
            _evaluate_augmented_grf(case, observed_only=False),
            _evaluate_augmented_grf(case, observed_only=True),
        ]

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
        partial_df["model_order"] = partial_df["name"].map({name: i for i, name in enumerate(MODEL_ORDER)})
        partial_df = partial_df.sort_values(["case_id", "model_order"]).drop(columns=["model_order"])
        partial_df.to_csv(output_dir / f"results_{args.tag}.csv", index=False)

    combined_df = pd.concat(case_frames, ignore_index=True)
    combined_df["model_order"] = combined_df["name"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    combined_df = combined_df.sort_values(["case_id", "model_order"]).drop(columns=["model_order"])
    output_csv = output_dir / f"results_{args.tag}.csv"
    combined_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
