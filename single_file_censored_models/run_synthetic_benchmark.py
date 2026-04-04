#!/usr/bin/env python3
"""Run the self-contained censored models on our synthetic benchmark cases.

This runner is designed to live next to the three standalone model files so
that someone can keep one folder open and both:

    1. inspect the model implementations, and
    2. run the standard basic12 benchmark or any selected synthetic cases.

By default it runs the three single-file models on the full basic12 suite.
It can also run only a subset of cases via case ids or case slugs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
MODEL_DIR = THIS_FILE.parent
PROJECT_ROOT = MODEL_DIR.parent
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks.econml_8variant import CASE_SPECS, _evaluate_predictions, prepare_case  # noqa: E402

try:  # pragma: no cover - allows both script and module execution
    from .final_censored_model import FinalModelCensoredSurvivalForest  # noqa: E402
    from .proper_censored_baseline import ProperNoPCICensoredSurvivalForest  # noqa: E402
    from .strict_censored_baseline import StrictEconmlXWZCensoredSurvivalForest  # noqa: E402
except ImportError:  # pragma: no cover
    from final_censored_model import FinalModelCensoredSurvivalForest  # type: ignore  # noqa: E402
    from proper_censored_baseline import ProperNoPCICensoredSurvivalForest  # type: ignore  # noqa: E402
    from strict_censored_baseline import StrictEconmlXWZCensoredSurvivalForest  # type: ignore  # noqa: E402


MODEL_BUILDERS = {
    "final": (
        "Final Model (single file)",
        lambda target, horizon, random_state: FinalModelCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=random_state,
            surv_scalar_mode="full",
        ),
    ),
    "strict": (
        "Strict EconML Censored Baseline (single file)",
        lambda target, horizon, random_state: StrictEconmlXWZCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=random_state,
        ),
    ),
    "proper": (
        "Proper Censored Baseline (single file)",
        lambda target, horizon, random_state: ProperNoPCICensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=random_state,
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standalone censored-model folder on the default basic12 "
            "suite or on any selected synthetic cases."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where case-level and aggregate CSV outputs will be written.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of integer case ids to run. If omitted, all basic12 cases are used.",
    )
    parser.add_argument(
        "--case-slugs",
        nargs="*",
        default=None,
        help="Optional list of case slugs to run, e.g. basic1 basic7.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(MODEL_BUILDERS),
        default=sorted(MODEL_BUILDERS),
        help="Subset of single-file models to run. Defaults to final strict proper.",
    )
    parser.add_argument(
        "--target",
        choices=["RMST", "survival.probability"],
        default="RMST",
        help="Target estimand used by the synthetic benchmark.",
    )
    parser.add_argument(
        "--horizon-quantile",
        type=float,
        default=0.60,
        help=(
            "Quantile used by the synthetic case helper when target=survival.probability. "
            "For RMST the current helper still uses the full observed maximum horizon."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed passed to the models.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print the available synthetic case ids and slugs, then exit.",
    )
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / "single_file_censored_models_basic12").resolve()


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


def _selected_case_specs(case_ids: list[int] | None, case_slugs: list[str] | None) -> list[dict[str, object]]:
    if not case_ids and not case_slugs:
        return list(CASE_SPECS)

    id_set = set(case_ids or [])
    slug_set = set(case_slugs or [])
    selected = [
        spec
        for spec in CASE_SPECS
        if int(spec["case_id"]) in id_set or str(spec["slug"]) in slug_set
    ]
    if not selected:
        raise ValueError("No synthetic cases matched the requested ids/slugs.")
    return selected


def _evaluate_case(case, *, target: str, random_state: int, model_keys: list[str]) -> list[dict[str, object]]:
    x = np.asarray(case.X, dtype=float)
    w = np.asarray(case.W, dtype=float)
    z = np.asarray(case.Z, dtype=float)
    a = np.asarray(case.A, dtype=float)
    time_obs = np.asarray(case.Y, dtype=float)
    event = np.asarray(case.delta, dtype=float)
    horizon = float(case.horizon)

    rows: list[dict[str, object]] = []
    for model_key in model_keys:
        display_name, builder = MODEL_BUILDERS[model_key]
        model = builder(target, horizon, random_state)
        t0 = time.time()
        model.fit_components(x, a, time_obs, event, z, w)
        preds = model.effect_from_components(x, w, z).ravel()
        elapsed = time.time() - t0
        rows.append(
            _evaluate_predictions(
                display_name,
                preds,
                case.true_cate,
                elapsed,
                backend=model.__class__.__name__,
            )
        )
    return rows


def main() -> int:
    args = parse_args()
    if args.list_cases:
        for spec in CASE_SPECS:
            print(f"{int(spec['case_id']):2d}  {spec['slug']}")
        return 0

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_specs = _selected_case_specs(args.case_ids, args.case_slugs)

    frames: list[pd.DataFrame] = []
    for case_spec in case_specs:
        case = prepare_case(case_spec, target=args.target, horizon_quantile=args.horizon_quantile)
        rows = _evaluate_case(
            case,
            target=args.target,
            random_state=args.random_state,
            model_keys=list(args.models),
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
