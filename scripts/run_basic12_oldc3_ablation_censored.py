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
from grf.methods import (  # noqa: E402
    BestCurveLocalCensoredPCISurvivalForest,
    EconmlMildShrinkNCSurvivalForest,
    OldC3AugmentedMinimalDMLCensoredSurvivalForest,
    OldC3AugmentedMinimalGRFCensoredSurvivalForest,
    OldC3AugmentedMinimalObservedDMLCensoredSurvivalForest,
    OldC3AugmentedMinimalObservedGRFCensoredSurvivalForest,
    OldC3AugmentedSurvDMLCensoredSurvivalForest,
    OldC3AugmentedSurvGRFCensoredSurvivalForest,
    OldC3AugmentedSurvObservedDMLCensoredSurvivalForest,
    OldC3AugmentedSurvObservedGRFCensoredSurvivalForest,
    OldC3SummaryMinimalDMLCensoredSurvivalForest,
    OldC3SummaryMinimalGRFCensoredSurvivalForest,
    OldC3SummaryMinimalObservedDMLCensoredSurvivalForest,
    OldC3SummaryMinimalObservedGRFCensoredSurvivalForest,
    OldC3SummarySurvDMLCensoredSurvivalForest,
    OldC3SummarySurvGRFCensoredSurvivalForest,
    OldC3SummarySurvObservedDMLCensoredSurvivalForest,
    OldC3SummarySurvObservedGRFCensoredSurvivalForest,
)


MODEL_ORDER = [
    "Old C3",
    "New C3",
    "SummaryMinimal-DML (PCI)",
    "SummaryMinimal-DML (no PCI)",
    "AugmentedMinimal-DML (PCI)",
    "AugmentedMinimal-DML (no PCI)",
    "SummaryMinimal-GRF (PCI)",
    "SummaryMinimal-GRF (no PCI)",
    "AugmentedMinimal-GRF (PCI)",
    "AugmentedMinimal-GRF (no PCI)",
    "SummarySurv-DML (PCI)",
    "SummarySurv-DML (no PCI)",
    "AugmentedSurv-DML (PCI)",
    "AugmentedSurv-DML (no PCI)",
    "SummarySurv-GRF (PCI)",
    "SummarySurv-GRF (no PCI)",
    "AugmentedSurv-GRF (PCI)",
    "AugmentedSurv-GRF (no PCI)",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run censored Old C3 ablation models on the basic 12 synthetic cases.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benchmark_basic12_oldc3_ablation_censored",
    )
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--target", choices=["RMST", "survival.probability"], default="RMST")
    parser.add_argument("--horizon-quantile", type=float, default=0.60)
    return parser.parse_args()


def _metric_row(name: str, preds, case, case_spec, *, target: str, elapsed: float, backend: str) -> dict[str, object]:
    row = _evaluate_predictions(name, preds, case.true_cate, elapsed, backend=backend)
    row.update(
        case_id=int(case_spec["case_id"]),
        case_slug=str(case_spec["slug"]),
        case_title=str(case_spec["title"]),
        target=str(target),
        estimand_horizon=float(case.horizon),
        horizon_quantile=None,
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


def _evaluate_old_c3(case, *, target: str):
    print("  - Old C3", flush=True)
    start = time.time()
    model = EconmlMildShrinkNCSurvivalForest(target=target, horizon=None)
    model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
    preds = model.effect_from_components(case.X, case.W, case.Z).ravel()
    return preds, time.time() - start, model.__class__.__name__


def _evaluate_new_c3(case, *, target: str):
    print("  - New C3", flush=True)
    start = time.time()
    model = BestCurveLocalCensoredPCISurvivalForest(target=target, horizon=None)
    model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
    preds = model.effect_from_components(case.X, case.W, case.Z).ravel()
    return preds, time.time() - start, model.__class__.__name__


def _evaluate_generic(case, *, name: str, model_cls, target: str):
    print(f"  - {name}", flush=True)
    start = time.time()
    model = model_cls(target=target, horizon=None)
    model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
    preds = model.effect_from_components(case.X, case.W, case.Z).ravel()
    return preds, time.time() - start, model.__class__.__name__


MODEL_SPECS = [
    ("SummaryMinimal-DML (PCI)", OldC3SummaryMinimalDMLCensoredSurvivalForest),
    ("SummaryMinimal-DML (no PCI)", OldC3SummaryMinimalObservedDMLCensoredSurvivalForest),
    ("AugmentedMinimal-DML (PCI)", OldC3AugmentedMinimalDMLCensoredSurvivalForest),
    ("AugmentedMinimal-DML (no PCI)", OldC3AugmentedMinimalObservedDMLCensoredSurvivalForest),
    ("SummaryMinimal-GRF (PCI)", OldC3SummaryMinimalGRFCensoredSurvivalForest),
    ("SummaryMinimal-GRF (no PCI)", OldC3SummaryMinimalObservedGRFCensoredSurvivalForest),
    ("AugmentedMinimal-GRF (PCI)", OldC3AugmentedMinimalGRFCensoredSurvivalForest),
    ("AugmentedMinimal-GRF (no PCI)", OldC3AugmentedMinimalObservedGRFCensoredSurvivalForest),
    ("SummarySurv-DML (PCI)", OldC3SummarySurvDMLCensoredSurvivalForest),
    ("SummarySurv-DML (no PCI)", OldC3SummarySurvObservedDMLCensoredSurvivalForest),
    ("AugmentedSurv-DML (PCI)", OldC3AugmentedSurvDMLCensoredSurvivalForest),
    ("AugmentedSurv-DML (no PCI)", OldC3AugmentedSurvObservedDMLCensoredSurvivalForest),
    ("SummarySurv-GRF (PCI)", OldC3SummarySurvGRFCensoredSurvivalForest),
    ("SummarySurv-GRF (no PCI)", OldC3SummarySurvObservedGRFCensoredSurvivalForest),
    ("AugmentedSurv-GRF (PCI)", OldC3AugmentedSurvGRFCensoredSurvivalForest),
    ("AugmentedSurv-GRF (no PCI)", OldC3AugmentedSurvObservedGRFCensoredSurvivalForest),
]


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

        preds, elapsed, backend = _evaluate_old_c3(case, target=args.target)
        rows = [_metric_row("Old C3", preds, case, case_spec, target=args.target, elapsed=elapsed, backend=backend)]
        preds, elapsed, backend = _evaluate_new_c3(case, target=args.target)
        rows.append(_metric_row("New C3", preds, case, case_spec, target=args.target, elapsed=elapsed, backend=backend))
        for name, model_cls in MODEL_SPECS:
            preds, elapsed, backend = _evaluate_generic(case, name=name, model_cls=model_cls, target=args.target)
            rows.append(_metric_row(name, preds, case, case_spec, target=args.target, elapsed=elapsed, backend=backend))

        case_df = pd.DataFrame(rows)
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
