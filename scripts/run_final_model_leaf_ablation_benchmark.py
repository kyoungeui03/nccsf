#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks.econml_8variant import (  # noqa: E402
    CASE_SPECS,
    _evaluate_predictions,
    prepare_case,
    render_avg_summary_png,
    render_case_table_png,
    render_top5_png,
)
from grf.censored import (  # noqa: E402
    FinalModelCensoredSurvivalForest,
    StrictEconmlXWZCensoredSurvivalForest,
)

matplotlib.use("Agg")


MODEL_BUILDERS = {
    "strict": (
        "Strict Baseline",
        lambda target, horizon: StrictEconmlXWZCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=42,
        ),
    ),
    "no_pci_x_only": (
        "Final No PCI, X only",
        lambda target, horizon: FinalModelCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=42,
            observed_only=True,
            include_raw_proxy=False,
            surv_scalar_mode="raw",
        ),
    ),
    "pci_x_only": (
        "Final PCI, X only",
        lambda target, horizon: FinalModelCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=42,
            observed_only=False,
            include_raw_proxy=False,
            surv_scalar_mode="raw",
        ),
    ),
    "pci_xwz": (
        "Final PCI, [X,W,Z]",
        lambda target, horizon: FinalModelCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=42,
            observed_only=False,
            include_raw_proxy=True,
            surv_scalar_mode="raw",
        ),
    ),
    "no_pci_full": (
        "Final No PCI, full",
        lambda target, horizon: FinalModelCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=42,
            observed_only=True,
            include_raw_proxy=True,
            surv_scalar_mode="full",
        ),
    ),
    "pci_full": (
        "Final PCI, full",
        lambda target, horizon: FinalModelCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=42,
            observed_only=False,
            include_raw_proxy=True,
            surv_scalar_mode="full",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the six-model leaf/splitting ablation for the censored Final Model "
            "on the basic12 synthetic benchmark."
        )
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        choices=["RMST", "survival.probability"],
        default=["RMST", "survival.probability"],
        help="Benchmark targets to evaluate.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of basic12 case ids.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where per-case and summary outputs will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(MODEL_BUILDERS),
        default=list(MODEL_BUILDERS),
        help="Optional subset of models to run.",
    )
    parser.add_argument(
        "--horizon-quantile",
        type=float,
        default=0.90,
        help="Quantile used when target=survival.probability.",
    )
    parser.add_argument(
        "--skip-pngs",
        action="store_true",
        help="Skip rendering PNG tables to reduce runtime.",
    )
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / "benchmark_final_model_leaf_ablation_basic12").resolve()


def _selected_case_specs(case_ids: list[int] | None) -> list[dict[str, object]]:
    if not case_ids:
        return list(CASE_SPECS)
    wanted = set(case_ids)
    selected = [spec for spec in CASE_SPECS if int(spec["case_id"]) in wanted]
    if not selected:
        raise ValueError("No synthetic cases matched the requested case ids.")
    return selected


def _format_case_title(case_spec: dict[str, object], case) -> str:
    base = str(case_spec["title"]).split(", n=", 1)[0]
    censor_pct = int(round(100 * float(case.cfg.target_censor_rate)))
    return (
        f"{base}, n={case.cfg.n}, p_x={case.cfg.p_x}, p_w={case.W.shape[1]}, "
        f"p_z={case.Z.shape[1]}, seed={case.cfg.seed}, censoring rate={censor_pct}%"
    )


def _summarize(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        results_df.groupby("name", as_index=False)
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
        .sort_values(["avg_pehe", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary, summary.head(5).copy()


def _evaluate_case(case, *, target: str, model_keys: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_key in model_keys:
        display_name, builder = MODEL_BUILDERS[model_key]
        model = builder(target, float(case.horizon))
        t0 = time.time()
        model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
        preds = model.effect_from_components(case.X, case.W, case.Z).ravel()
        rows.append(
            _evaluate_predictions(
                display_name,
                preds,
                case.true_cate,
                time.time() - t0,
                backend=model.__class__.__name__,
            )
        )
    return rows


def _run_target(
    *,
    target: str,
    case_specs: list[dict[str, object]],
    output_root: Path,
    model_keys: list[str],
    horizon_quantile: float,
    render_pngs: bool,
) -> None:
    target_slug = "surv" if target == "survival.probability" else "rmst"
    output_dir = output_root / target_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for case_spec in case_specs:
        case = prepare_case(case_spec, target=target, horizon_quantile=horizon_quantile)
        rows = _evaluate_case(case, target=target, model_keys=model_keys)
        case_df = pd.DataFrame(rows)
        case_df["case_id"] = int(case_spec["case_id"])
        case_df["case_slug"] = str(case_spec["slug"])
        case_df["case_title"] = _format_case_title(case_spec, case)
        case_df["target"] = target
        case_df["horizon"] = float(case.horizon)
        frames.append(case_df)

        case_base = f"case_{case_spec['case_id']:02d}_{case_spec['slug']}"
        case_df.to_csv(output_dir / f"{case_base}.csv", index=False)
        if render_pngs:
            render_case_table_png(case_df, output_dir / f"{case_base}.png")

    results_df = pd.concat(frames, ignore_index=True)
    summary_df, top5_df = _summarize(results_df)
    results_df.to_csv(output_dir / "results_full.csv", index=False)
    summary_df.to_csv(output_dir / "summary_full.csv", index=False)
    top5_df.to_csv(output_dir / "top5_full.csv", index=False)
    if render_pngs:
        render_avg_summary_png(summary_df, output_dir / "summary_full.png")
        render_top5_png(top5_df, output_dir / "top5_full.png")


def main() -> int:
    args = parse_args()
    output_root = _resolve_output_dir(args)
    case_specs = _selected_case_specs(args.case_ids)
    model_keys = list(args.models)
    render_pngs = not args.skip_pngs

    for target in args.targets:
        _run_target(
            target=target,
            case_specs=case_specs,
            output_root=output_root,
            model_keys=model_keys,
            horizon_quantile=float(args.horizon_quantile),
            render_pngs=render_pngs,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
