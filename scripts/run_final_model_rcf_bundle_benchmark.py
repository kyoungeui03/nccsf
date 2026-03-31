#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))
ROOT_SCRIPTS = PROJECT_ROOT / "scripts"
if str(ROOT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS))

from grf.benchmarks.econml_8variant import (  # noqa: E402
    CASE_SPECS as C_CASE_SPECS,
    _evaluate_predictions,
    evaluate_r_csf_variant,
    prepare_case,
)
from grf.censored import (  # noqa: E402
    FinalModelCensoredSurvivalForest,
    FinalModelRCSFCensoredSurvivalForest,
)
from grf.non_censored import (  # noqa: E402
    FinalModelNCCausalForest,
    FinalModelRCFNCCausalForest,
    StrictEconmlXWZNCCausalForest,
)
from grf.non_censored.benchmarks import (  # noqa: E402
    CASE_SPECS as NC_CASE_SPECS,
    _build_case,
    _make_cfg,
    _metric_row,
)
from grf.r_runtime import resolve_rscript  # noqa: E402
from preprocess_rhc import build_cleaned_rhc  # noqa: E402


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

R_CF_SCRIPT = PROJECT_ROOT / "scripts" / "run_grf_cf_baseline.R"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the finalized-model benchmark with the NC R grf final-forest variant "
            "and the censored fair-horizon R grf final-forest variant."
        )
    )
    parser.add_argument("--dataset", choices=["basic12", "structured14", "rhc"], required=True)
    parser.add_argument("--domain", choices=["non_censored", "censored", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--setting-ids", nargs="*")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p-x", type=int, default=None)
    parser.add_argument("--p-w", type=int, default=1)
    parser.add_argument("--p-z", type=int, default=1)
    parser.add_argument("--input-csv", type=Path, default=PROJECT_ROOT / "data" / "rhc" / "raw_rhc.csv")
    parser.add_argument("--target", choices=["RMST", "survival.probability"], default="RMST")
    parser.add_argument("--horizon-quantile", type=float, default=0.60)
    parser.add_argument("--rhc-horizon", type=float, default=30.0)
    parser.add_argument("--num-trees-r", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / f"benchmark_final_model_rcf_bundle_{args.dataset}").resolve()


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


def _setting_slug(setting: dict[str, int | str]) -> str:
    return (
        f"{setting['setting_id']}_n{setting['n']}_px{setting['p_x']}"
        f"_pw{setting['p_w']}_pz{setting['p_z']}"
    )


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


def _prediction_summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return (
        df.groupby(["domain", "name"], as_index=False)
        .agg(
            n_obs=("n_obs", "max"),
            mean_pred=("mean_pred", "mean"),
            std_pred=("std_pred", "mean"),
            median_pred=("median_pred", "mean"),
            pct_positive=("pct_positive", "mean"),
            min_pred=("min_pred", "mean"),
            max_pred=("max_pred", "mean"),
            time_sec=("time_sec", "mean"),
        )
        .sort_values(["domain", "name"])
        .reset_index(drop=True)
    )


def _prediction_row(name: str, preds: np.ndarray, elapsed: float, *, domain: str) -> dict[str, object]:
    preds = np.asarray(preds, dtype=float).ravel()
    return {
        "domain": domain,
        "name": name,
        "n_obs": int(len(preds)),
        "mean_pred": float(np.mean(preds)),
        "std_pred": float(np.std(preds)),
        "median_pred": float(np.median(preds)),
        "pct_positive": float(np.mean(preds > 0)),
        "min_pred": float(np.min(preds)),
        "max_pred": float(np.max(preds)),
        "time_sec": float(elapsed),
    }


def _run_r_cf_baseline(
    obs_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    output_root: Path,
    num_trees: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root, prefix="r_cf_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        input_path = tmp_dir_path / "input.csv"
        output_path = tmp_dir_path / "predictions.csv"
        obs_df.to_csv(input_path, index=False)
        cmd = [
            resolve_rscript(),
            str(R_CF_SCRIPT),
            str(input_path),
            ",".join(feature_cols),
            str(int(num_trees)),
            str(output_path),
            str(int(seed)),
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError(
                "R causal_forest baseline failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        preds = pd.read_csv(output_path)["pred"].to_numpy(dtype=float)
    return preds, elapsed


def _run_r_csf_baseline_direct(
    obs_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    output_root: Path,
    horizon: float,
    num_trees: int,
    target: str,
) -> tuple[np.ndarray, float]:
    with tempfile.TemporaryDirectory(dir=output_root, prefix="r_csf_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        input_path = tmp_dir_path / "input.csv"
        output_path = tmp_dir_path / "predictions.csv"
        obs_df.loc[:, ["time", "event", "A", *feature_cols]].to_csv(input_path, index=False)
        cmd = [
            resolve_rscript(),
            str(PROJECT_ROOT / "scripts" / "run_grf_csf_baseline.R"),
            str(input_path),
            ",".join(feature_cols),
            str(float(horizon)),
            str(int(num_trees)),
            str(output_path),
            target,
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError(
                "R causal_survival_forest baseline failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        preds = pd.read_csv(output_path)["prediction"].to_numpy(dtype=float)
    return preds, elapsed


def _evaluate_synthetic_nc_case(case: dict[str, object], *, output_root: Path, num_trees_r: int, random_state: int):
    x = np.asarray(case["X"], dtype=float)
    w = np.asarray(case["W"], dtype=float)
    z = np.asarray(case["Z"], dtype=float)
    a = np.asarray(case["A"], dtype=float)
    y = np.asarray(case["Y"], dtype=float)
    true_cate = np.asarray(case["true_cate"], dtype=float)

    rows = []

    final_model = FinalModelNCCausalForest(random_state=random_state)
    t0 = time.time()
    final_model.fit_components(x, a, y, z, w)
    preds = final_model.effect_from_components(x, w, z).ravel()
    rows.append(_metric_row("Final Model", preds, true_cate, time.time() - t0))

    rcf_final_model = FinalModelRCFNCCausalForest(random_state=random_state)
    t0 = time.time()
    rcf_final_model.fit_components(x, a, y, z, w)
    preds = rcf_final_model.effect_from_components(x, w, z).ravel()
    rows.append(_metric_row("Final Model (R grf final forest)", preds, true_cate, time.time() - t0))

    econ_model = StrictEconmlXWZNCCausalForest(random_state=random_state, cv=2)
    t0 = time.time()
    econ_model.fit_components(x, a, y, z, w)
    preds = econ_model.effect_from_components(x, w, z).ravel()
    rows.append(_metric_row("EconML Baseline", preds, true_cate, time.time() - t0))

    feature_cols = [*case["x_cols"], *case["w_cols"], *case["z_cols"]]
    obs_df = case["obs_df"].loc[:, [*feature_cols, "A"]].copy()
    obs_df["Y"] = case["Y"]
    preds, elapsed = _run_r_cf_baseline(
        obs_df,
        feature_cols,
        output_root=output_root,
        num_trees=num_trees_r,
        seed=random_state,
    )
    rows.append(_metric_row("R-CF Baseline", preds, true_cate, elapsed))
    return rows


def _evaluate_synthetic_c_case(case, *, output_root: Path, num_trees_r: int, target: str, random_state: int):
    x = np.asarray(case.X, dtype=float)
    w = np.asarray(case.W, dtype=float)
    z = np.asarray(case.Z, dtype=float)
    a = np.asarray(case.A, dtype=float)
    time_obs = np.asarray(case.Y, dtype=float)
    event = np.asarray(case.delta, dtype=float)
    horizon = float(case.horizon)

    rows = []

    final_model = FinalModelCensoredSurvivalForest(
        target=target,
        horizon=horizon,
        random_state=random_state,
    )
    t0 = time.time()
    final_model.fit_components(x, a, time_obs, event, z, w)
    preds = final_model.effect_from_components(x, w, z).ravel()
    rows.append(
        _evaluate_predictions(
            "Final Model",
            preds,
            case.true_cate,
            time.time() - t0,
            backend=final_model.__class__.__name__,
        )
    )

    r_final_model = FinalModelRCSFCensoredSurvivalForest(
        target=target,
        horizon=horizon,
        random_state=random_state,
    )
    t0 = time.time()
    r_final_model.fit_components(x, a, time_obs, event, z, w)
    preds = r_final_model.effect_from_components(x, w, z).ravel()
    rows.append(
        _evaluate_predictions(
            "Final Model (R grf final forest)",
            preds,
            case.true_cate,
            time.time() - t0,
            backend=r_final_model.__class__.__name__,
        )
    )

    feature_cols = [*case.x_cols, *case.w_cols, *case.z_cols]
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


def _run_basic12_non_censored(case_specs: list[dict[str, object]], output_dir: Path, *, num_trees_r: int, random_state: int) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for case_spec in case_specs:
        case = _build_case(_make_cfg(case_spec), case_spec)
        rows = _evaluate_synthetic_nc_case(case, output_root=output_dir, num_trees_r=num_trees_r, random_state=random_state)
        case_df = pd.DataFrame(rows)
        case_df["case_id"] = int(case_spec["case_id"])
        case_df["case_slug"] = str(case_spec["slug"])
        frames.append(case_df)
        case_df.to_csv(output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.csv", index=False)

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(output_dir / "results_full.csv", index=False)
    _metric_summary(results, time_col="time_sec").to_csv(output_dir / "summary_full.csv", index=False)
    return results


def _run_basic12_censored(
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
        rows = _evaluate_synthetic_c_case(
            case,
            output_root=output_dir,
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


def _run_structured14_non_censored(
    settings: list[dict[str, int | str]],
    *,
    case_ids: set[int] | None,
    output_dir: Path,
    num_trees_r: int,
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
        for raw_case_spec in NC_CASE_SPECS:
            if case_ids is not None and int(raw_case_spec["case_id"]) not in case_ids:
                continue
            case_spec = _case_with_overrides(
                raw_case_spec,
                n=setting["n"],
                p_x=setting["p_x"],
                p_w=setting["p_w"],
                p_z=setting["p_z"],
            )
            case = _build_case(_make_cfg(case_spec), case_spec)
            rows = _evaluate_synthetic_nc_case(case, output_root=setting_dir, num_trees_r=num_trees_r, random_state=random_state)
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
        _metric_summary(results, time_col="time_sec").to_csv(setting_dir / "summary.csv", index=False)
        all_frames.append(results)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(output_dir / "all_settings_results.csv", index=False)
    _metric_summary_by_setting(combined, time_col="time_sec").to_csv(output_dir / "all_settings_summary.csv", index=False)
    return combined


def _run_structured14_censored(
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
            rows = _evaluate_synthetic_c_case(
                case,
                output_root=setting_dir,
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


def _build_rhc_non_censored(raw_df: pd.DataFrame):
    analysis_df = build_cleaned_rhc(raw_df)
    z_cols = [col for col in ["pafi1", "paco21"] if col in analysis_df.columns]
    w_cols = [col for col in ["ph1", "hema1"] if col in analysis_df.columns]
    x_cols = [col for col in analysis_df.columns if col not in {"Y", "A", *z_cols, *w_cols}]
    return analysis_df, x_cols, w_cols, z_cols


def _build_rhc_censored(raw_df: pd.DataFrame):
    cleaned = build_cleaned_rhc(raw_df)
    features = cleaned.drop(columns=["Y", "A"], errors="ignore").copy()
    analysis_df = pd.DataFrame(
        {
            "time": raw_df["t3d30"].astype(float),
            "event": (raw_df["dth30"] == "Yes").astype(int),
            "A": cleaned["A"].astype(int),
        }
    )
    analysis_df = pd.concat([analysis_df, features], axis=1)
    z_cols = [col for col in ["pafi1", "paco21"] if col in analysis_df.columns]
    w_cols = [col for col in ["ph1", "hema1"] if col in analysis_df.columns]
    x_cols = [col for col in analysis_df.columns if col not in {"time", "event", "A", *z_cols, *w_cols}]
    return analysis_df, x_cols, w_cols, z_cols


def _run_rhc_non_censored(raw_df: pd.DataFrame, output_dir: Path, *, num_trees_r: int, random_state: int) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_df, x_cols, w_cols, z_cols = _build_rhc_non_censored(raw_df)
    x = analysis_df[x_cols].to_numpy(dtype=float)
    w = analysis_df[w_cols].to_numpy(dtype=float)
    z = analysis_df[z_cols].to_numpy(dtype=float)
    a = analysis_df["A"].to_numpy(dtype=float)
    y = analysis_df["Y"].to_numpy(dtype=float)

    prediction_frames = []
    summary_rows = []

    final_model = FinalModelNCCausalForest(random_state=random_state)
    t0 = time.time()
    final_model.fit_components(x, a, y, z, w)
    preds = final_model.effect_from_components(x, w, z).ravel()
    elapsed = time.time() - t0
    prediction_frames.append(pd.DataFrame({"row_id": np.arange(len(preds)), "domain": "non_censored", "name": "Final Model", "prediction": preds}))
    summary_rows.append(_prediction_row("Final Model", preds, elapsed, domain="non_censored"))

    rcf_final_model = FinalModelRCFNCCausalForest(random_state=random_state)
    t0 = time.time()
    rcf_final_model.fit_components(x, a, y, z, w)
    preds = rcf_final_model.effect_from_components(x, w, z).ravel()
    elapsed = time.time() - t0
    prediction_frames.append(
        pd.DataFrame({"row_id": np.arange(len(preds)), "domain": "non_censored", "name": "Final Model (R grf final forest)", "prediction": preds})
    )
    summary_rows.append(_prediction_row("Final Model (R grf final forest)", preds, elapsed, domain="non_censored"))

    econ_model = StrictEconmlXWZNCCausalForest(random_state=random_state, cv=2)
    t0 = time.time()
    econ_model.fit_components(x, a, y, z, w)
    preds = econ_model.effect_from_components(x, w, z).ravel()
    elapsed = time.time() - t0
    prediction_frames.append(pd.DataFrame({"row_id": np.arange(len(preds)), "domain": "non_censored", "name": "EconML Baseline", "prediction": preds}))
    summary_rows.append(_prediction_row("EconML Baseline", preds, elapsed, domain="non_censored"))

    feature_cols = [*x_cols, *w_cols, *z_cols]
    obs_df = analysis_df.loc[:, [*feature_cols, "A", "Y"]].copy()
    preds, elapsed = _run_r_cf_baseline(obs_df, feature_cols, output_root=output_dir, num_trees=num_trees_r, seed=random_state)
    prediction_frames.append(pd.DataFrame({"row_id": np.arange(len(preds)), "domain": "non_censored", "name": "R-CF Baseline", "prediction": preds}))
    summary_rows.append(_prediction_row("R-CF Baseline", preds, elapsed, domain="non_censored"))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary = _prediction_summary(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def _run_rhc_censored(
    raw_df: pd.DataFrame,
    output_dir: Path,
    *,
    num_trees_r: int,
    target: str,
    rhc_horizon: float,
    random_state: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_df, x_cols, w_cols, z_cols = _build_rhc_censored(raw_df)
    x = analysis_df[x_cols].to_numpy(dtype=float)
    w = analysis_df[w_cols].to_numpy(dtype=float)
    z = analysis_df[z_cols].to_numpy(dtype=float)
    a = analysis_df["A"].to_numpy(dtype=float)
    time_obs = analysis_df["time"].to_numpy(dtype=float)
    event = analysis_df["event"].to_numpy(dtype=float)

    prediction_frames = []
    summary_rows = []

    final_model = FinalModelCensoredSurvivalForest(
        target=target,
        horizon=rhc_horizon,
        random_state=random_state,
    )
    t0 = time.time()
    final_model.fit_components(x, a, time_obs, event, z, w)
    preds = final_model.effect_from_components(x, w, z).ravel()
    elapsed = time.time() - t0
    prediction_frames.append(pd.DataFrame({"row_id": np.arange(len(preds)), "domain": "censored", "name": "Final Model", "prediction": preds}))
    summary_rows.append(_prediction_row("Final Model", preds, elapsed, domain="censored"))

    r_final_model = FinalModelRCSFCensoredSurvivalForest(
        target=target,
        horizon=rhc_horizon,
        random_state=random_state,
    )
    t0 = time.time()
    r_final_model.fit_components(x, a, time_obs, event, z, w)
    preds = r_final_model.effect_from_components(x, w, z).ravel()
    elapsed = time.time() - t0
    prediction_frames.append(
        pd.DataFrame({"row_id": np.arange(len(preds)), "domain": "censored", "name": "Final Model (R grf final forest)", "prediction": preds})
    )
    summary_rows.append(_prediction_row("Final Model (R grf final forest)", preds, elapsed, domain="censored"))

    feature_cols = [*x_cols, *w_cols, *z_cols]
    preds, elapsed = _run_r_csf_baseline_direct(
        analysis_df,
        feature_cols,
        output_root=output_dir,
        horizon=rhc_horizon,
        num_trees=num_trees_r,
        target=target,
    )
    prediction_frames.append(pd.DataFrame({"row_id": np.arange(len(preds)), "domain": "censored", "name": "R-CSF Baseline", "prediction": preds}))
    summary_rows.append(_prediction_row("R-CSF Baseline", preds, elapsed, domain="censored"))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary = _prediction_summary(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def _write_combined_summary(output_dir: Path, frames: list[pd.DataFrame], filename: str) -> None:
    valid = [df for df in frames if df is not None and not df.empty]
    if not valid:
        return
    pd.concat(valid, ignore_index=True).to_csv(output_dir / filename, index=False)


def main() -> int:
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_ids = set(args.case_ids) if args.case_ids else None
    domains = ["non_censored", "censored"] if args.domain == "both" else [args.domain]

    if args.dataset == "basic12":
        combined = []
        if "non_censored" in domains:
            nc_cases = [
                _case_with_overrides(case, n=args.n, p_x=args.p_x, p_w=args.p_w, p_z=args.p_z)
                for case in NC_CASE_SPECS
                if case_ids is None or int(case["case_id"]) in case_ids
            ]
            combined.append(
                _run_basic12_non_censored(
                    nc_cases,
                    output_dir / "non_censored",
                    num_trees_r=args.num_trees_r,
                    random_state=args.random_state,
                ).assign(domain="non_censored")
            )
        if "censored" in domains:
            c_cases = [
                _case_with_overrides(case, n=args.n, p_x=args.p_x, p_w=args.p_w, p_z=args.p_z)
                for case in C_CASE_SPECS
                if case_ids is None or int(case["case_id"]) in case_ids
            ]
            combined.append(
                _run_basic12_censored(
                    c_cases,
                    output_dir / "censored",
                    num_trees_r=args.num_trees_r,
                    target=args.target,
                    horizon_quantile=args.horizon_quantile,
                    random_state=args.random_state,
                ).assign(domain="censored")
            )
        _write_combined_summary(output_dir, combined, "combined_results_full.csv")
        return 0

    if args.dataset == "structured14":
        setting_ids = set(args.setting_ids) if args.setting_ids else None
        settings = [s for s in SETTINGS if setting_ids is None or s["setting_id"] in setting_ids]
        combined = []
        if "non_censored" in domains:
            combined.append(
                _run_structured14_non_censored(
                    settings,
                    case_ids=case_ids,
                    output_dir=output_dir / "non_censored",
                    num_trees_r=args.num_trees_r,
                    random_state=args.random_state,
                    skip_existing=args.skip_existing,
                ).assign(domain="non_censored")
            )
        if "censored" in domains:
            combined.append(
                _run_structured14_censored(
                    settings,
                    case_ids=case_ids,
                    output_dir=output_dir / "censored",
                    num_trees_r=args.num_trees_r,
                    target=args.target,
                    horizon_quantile=args.horizon_quantile,
                    random_state=args.random_state,
                    skip_existing=args.skip_existing,
                ).assign(domain="censored")
            )
        _write_combined_summary(output_dir, combined, "combined_all_settings_results.csv")
        return 0

    raw_df = pd.read_csv(args.input_csv.resolve())
    combined = []
    if "non_censored" in domains:
        combined.append(
            _run_rhc_non_censored(
                raw_df,
                output_dir / "non_censored",
                num_trees_r=args.num_trees_r,
                random_state=args.random_state,
            )
        )
    if "censored" in domains:
        combined.append(
            _run_rhc_censored(
                raw_df,
                output_dir / "censored",
                num_trees_r=args.num_trees_r,
                target=args.target,
                rhc_horizon=args.rhc_horizon,
                random_state=args.random_state,
            )
        )
    _write_combined_summary(output_dir, combined, "combined_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
