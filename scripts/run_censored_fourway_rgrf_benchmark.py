#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks.econml_8variant import (  # noqa: E402
    CASE_SPECS as C_CASE_SPECS,
    _evaluate_predictions,
    evaluate_r_csf_variant,
    prepare_case,
)
from grf.censored import (  # noqa: E402
    FinalModelCSFFinalCensoredSurvivalForest,
    FinalModelCensoredSurvivalForest,
    FinalModelRCSFCensoredSurvivalForest,
)


SETTINGS = [
    {"setting_id": "S01", "n": 1000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S02", "n": 2000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S03", "n": 4000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S04", "n": 8000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S05", "n": 2000, "p_x": 5, "p_w": 3, "p_z": 3},
    {"setting_id": "S06", "n": 2000, "p_x": 5, "p_w": 5, "p_z": 5},
    {"setting_id": "S07", "n": 2000, "p_x": 5, "p_w": 10, "p_z": 10},
    {"setting_id": "S08", "n": 2000, "p_x": 10, "p_w": 1, "p_z": 1},
    {"setting_id": "S09", "n": 2000, "p_x": 20, "p_w": 1, "p_z": 1},
    {"setting_id": "S10", "n": 2000, "p_x": 10, "p_w": 5, "p_z": 5},
    {"setting_id": "S11", "n": 2000, "p_x": 20, "p_w": 5, "p_z": 5},
    {"setting_id": "S12", "n": 1000, "p_x": 10, "p_w": 5, "p_z": 5},
    {"setting_id": "S13", "n": 2000, "p_x": 10, "p_w": 10, "p_z": 10},
    {"setting_id": "S14", "n": 4000, "p_x": 20, "p_w": 10, "p_z": 10},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a four-way censored benchmark with unified horizon/target settings: "
            "Final Model, R-CSF Baseline, Final Model (R grf final forest), "
            "and Final Model (R grf survival final forest)."
        )
    )
    parser.add_argument("--dataset", choices=["basic12", "structured14"], required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--setting-ids", nargs="*")
    parser.add_argument("--target", choices=["RMST", "survival.probability"], default="RMST")
    parser.add_argument("--horizon-quantile", type=float, default=0.60)
    parser.add_argument("--num-trees-r", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / f"benchmark_censored_fourway_rgrf_{args.dataset}").resolve()


def _setting_slug(setting: dict[str, int | str]) -> str:
    return (
        f"{setting['setting_id']}_n{setting['n']}_px{setting['p_x']}"
        f"_pw{setting['p_w']}_pz{setting['p_z']}"
    )


def _case_with_overrides(case_spec, *, n=None, p_x=None, p_w=None, p_z=None):
    case_copy = dict(case_spec)
    cfg_updates = dict(case_spec["cfg"])
    if n is not None:
        cfg_updates["n"] = int(n)
    if p_x is not None:
        cfg_updates["p_x"] = int(p_x)
    if p_w is not None:
        cfg_updates["p_w"] = int(p_w)
    if p_z is not None:
        cfg_updates["p_z"] = int(p_z)
    case_copy["cfg"] = cfg_updates
    return case_copy


def _metric_summary(df: pd.DataFrame, *, time_col: str) -> pd.DataFrame:
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
            avg_time=(time_col, "mean"),
            n_cases=("case_id", "count"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def _metric_summary_by_setting(df: pd.DataFrame, *, time_col: str) -> pd.DataFrame:
    return (
        df.groupby(["setting_id", "n", "p_x", "p_w", "p_z", "name"], as_index=False)
        .agg(
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
            avg_time=(time_col, "mean"),
        )
        .sort_values(["setting_id", "avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )


def _evaluate_case(case, *, num_trees_r: int, target: str, random_state: int) -> list[dict[str, object]]:
    x = np.asarray(case.X, dtype=float)
    w = np.asarray(case.W, dtype=float)
    z = np.asarray(case.Z, dtype=float)
    a = np.asarray(case.A, dtype=float)
    time_obs = np.asarray(case.Y, dtype=float)
    event = np.asarray(case.delta, dtype=float)
    feature_cols = [*case.x_cols, *case.w_cols, *case.z_cols]
    horizon = float(case.horizon)

    rows: list[dict[str, object]] = []

    models = [
        ("Final Model", FinalModelCensoredSurvivalForest),
        ("Final Model (R grf final forest)", FinalModelRCSFCensoredSurvivalForest),
        ("Final Model (R grf survival final forest)", FinalModelCSFFinalCensoredSurvivalForest),
    ]

    for name, model_cls in models:
        model = model_cls(
            target=target,
            horizon=horizon,
            random_state=random_state,
        )
        import time as _time

        t0 = _time.time()
        model.fit_components(x, a, time_obs, event, z, w)
        preds = model.effect_from_components(x, w, z).ravel()
        rows.append(
            _evaluate_predictions(
                name,
                preds,
                case.true_cate,
                _time.time() - t0,
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


def _run_basic12(
    case_specs: list[dict[str, object]],
    output_dir: Path,
    *,
    num_trees_r: int,
    target: str,
    horizon_quantile: float,
    random_state: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for case_spec in case_specs:
        case = prepare_case(case_spec, target=target, horizon_quantile=horizon_quantile)
        rows = _evaluate_case(
            case,
            num_trees_r=num_trees_r,
            target=target,
            random_state=random_state,
        )
        case_df = pd.DataFrame(rows)
        case_df["case_id"] = int(case_spec["case_id"])
        case_df["case_slug"] = str(case_spec["slug"])
        frames.append(case_df)
        case_df.to_csv(output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.csv", index=False)

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(output_dir / "results_full.csv", index=False)
    _metric_summary(results, time_col="total_time").to_csv(output_dir / "summary_full.csv", index=False)
    return results


def _run_structured14(
    settings: list[dict[str, int | str]],
    *,
    case_ids: set[int] | None,
    output_dir: Path,
    num_trees_r: int,
    target: str,
    horizon_quantile: float,
    random_state: int,
    skip_existing: bool,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []
    for setting in settings:
        setting_dir = output_dir / _setting_slug(setting)
        results_path = setting_dir / "results.csv"
        if skip_existing and results_path.exists():
            all_frames.append(pd.read_csv(results_path))
            continue
        setting_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for raw_case_spec in C_CASE_SPECS:
            if case_ids is not None and int(raw_case_spec["case_id"]) not in case_ids:
                continue
            case_spec = _case_with_overrides(
                raw_case_spec,
                n=setting["n"],
                p_x=setting["p_x"],
                p_w=setting["p_w"],
                p_z=setting["p_z"],
            )
            case = prepare_case(case_spec, target=target, horizon_quantile=horizon_quantile)
            rows = _evaluate_case(
                case,
                num_trees_r=num_trees_r,
                target=target,
                random_state=random_state,
            )
            case_df = pd.DataFrame(rows)
            case_df["setting_id"] = str(setting["setting_id"])
            case_df["case_id"] = int(case_spec["case_id"])
            case_df["case_slug"] = str(case_spec["slug"])
            case_df["n"] = int(setting["n"])
            case_df["p_x"] = int(setting["p_x"])
            case_df["p_w"] = int(setting["p_w"])
            case_df["p_z"] = int(setting["p_z"])
            frames.append(case_df)
        results = pd.concat(frames, ignore_index=True)
        results.to_csv(results_path, index=False)
        _metric_summary(results, time_col="total_time").to_csv(setting_dir / "summary.csv", index=False)
        all_frames.append(results)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(output_dir / "all_settings_results.csv", index=False)
    _metric_summary_by_setting(combined, time_col="total_time").to_csv(output_dir / "all_settings_summary.csv", index=False)
    return combined


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    case_ids = None if not args.case_ids else set(args.case_ids)
    setting_ids = None if not args.setting_ids else set(args.setting_ids)

    if args.dataset == "basic12":
        case_specs = [spec for spec in C_CASE_SPECS if case_ids is None or int(spec["case_id"]) in case_ids]
        _run_basic12(
            case_specs,
            output_dir,
            num_trees_r=args.num_trees_r,
            target=args.target,
            horizon_quantile=args.horizon_quantile,
            random_state=args.random_state,
        )
        return

    settings = [setting for setting in SETTINGS if setting_ids is None or str(setting["setting_id"]) in setting_ids]
    _run_structured14(
        settings,
        case_ids=case_ids,
        output_dir=output_dir,
        num_trees_r=args.num_trees_r,
        target=args.target,
        horizon_quantile=args.horizon_quantile,
        random_state=args.random_state,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
