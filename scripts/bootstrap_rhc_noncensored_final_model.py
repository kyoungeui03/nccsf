#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))
ROOT_SCRIPTS = PROJECT_ROOT / "scripts"
if str(ROOT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS))

from grf.non_censored import FinalModelNCCausalForest  # noqa: E402
from preprocess_rhc import build_cleaned_rhc  # noqa: E402


_X: np.ndarray | None = None
_W: np.ndarray | None = None
_Z: np.ndarray | None = None
_A: np.ndarray | None = None
_Y: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nonparametric bootstrap inference for the uncensored final model "
            "on the SUPPORT RHC binary 30-day survival analysis."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=PROJECT_ROOT / "data" / "rhc" / "raw_rhc.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "rhc_noncensored_final_bootstrap")
    parser.add_argument("--point-predictions", type=Path, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def build_rhc_non_censored(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    analysis_df = build_cleaned_rhc(raw_df)
    analysis_df["Y"] = (raw_df["dth30"] != "Yes").astype(float)
    z_cols = [col for col in ["pafi1", "paco21"] if col in analysis_df.columns]
    w_cols = [col for col in ["ph1", "hema1"] if col in analysis_df.columns]
    x_cols = [col for col in analysis_df.columns if col not in {"Y", "A", *z_cols, *w_cols}]
    return analysis_df, x_cols, w_cols, z_cols


def _init_worker(x: np.ndarray, w: np.ndarray, z: np.ndarray, a: np.ndarray, y: np.ndarray) -> None:
    global _X, _W, _Z, _A, _Y
    _X = x
    _W = w
    _Z = z
    _A = a
    _Y = y
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _run_bootstrap_rep(rep: int, seed: int, random_state: int) -> dict[str, float | int]:
    if _X is None or _W is None or _Z is None or _A is None or _Y is None:
        raise RuntimeError("Worker arrays were not initialized.")

    rng = np.random.default_rng(seed)
    n = int(_Y.shape[0])
    idx = rng.integers(0, n, size=n)

    t0 = time.time()
    model = FinalModelNCCausalForest(random_state=int(random_state) + int(rep))
    model.fit_components(_X[idx], _A[idx], _Y[idx], _Z[idx], _W[idx])
    pred_boot_sample = np.asarray(model.effect_from_components(_X[idx], _W[idx], _Z[idx]), dtype=float).ravel()
    pred_original_sample = np.asarray(model.effect_from_components(_X, _W, _Z), dtype=float).ravel()
    elapsed = time.time() - t0

    return {
        "rep": int(rep),
        "seed": int(seed),
        "n": n,
        "ate_boot_sample": float(np.mean(pred_boot_sample)),
        "ate_original_sample": float(np.mean(pred_original_sample)),
        "elapsed_sec": float(elapsed),
    }


def _point_estimate_from_predictions(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    pred_df = pd.read_csv(path)
    pred_df = pred_df[pred_df["name"] == "Final Model"]
    if pred_df.empty:
        return None
    return float(pred_df["prediction"].astype(float).mean())


def _summarize(draws: pd.Series, point_estimate: float) -> dict[str, float | int | str]:
    values = draws.astype(float).to_numpy()
    return {
        "method": "patient-level nonparametric bootstrap",
        "statistic": draws.name,
        "bootstrap_reps": int(values.shape[0]),
        "point_estimate": float(point_estimate),
        "bootstrap_mean": float(np.mean(values)),
        "bootstrap_se": float(np.std(values, ddof=1)),
        "normal_ci_low": float(point_estimate - 1.96 * np.std(values, ddof=1)),
        "normal_ci_high": float(point_estimate + 1.96 * np.std(values, ddof=1)),
        "percentile_ci_low": float(np.quantile(values, 0.025)),
        "percentile_ci_high": float(np.quantile(values, 0.975)),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    draws_path = output_dir / "bootstrap_draws.csv"
    summary_path = output_dir / "bootstrap_summary.csv"

    raw_df = pd.read_csv(args.input_csv.resolve())
    analysis_df, x_cols, w_cols, z_cols = build_rhc_non_censored(raw_df)
    x = analysis_df[x_cols].to_numpy(dtype=float)
    w = analysis_df[w_cols].to_numpy(dtype=float)
    z = analysis_df[z_cols].to_numpy(dtype=float)
    a = analysis_df["A"].to_numpy(dtype=float)
    y = analysis_df["Y"].to_numpy(dtype=float)

    point_predictions = args.point_predictions
    if point_predictions is None:
        point_predictions = PROJECT_ROOT / "outputs" / "rhc_final_model_rcf_bundle_surv30_binary" / "non_censored" / "predictions.csv"
    point_estimate = _point_estimate_from_predictions(point_predictions)
    if point_estimate is None:
        t0 = time.time()
        model = FinalModelNCCausalForest(random_state=args.random_state)
        model.fit_components(x, a, y, z, w)
        point_estimate = float(np.mean(np.asarray(model.effect_from_components(x, w, z), dtype=float).ravel()))
        pd.DataFrame([{"point_estimate": point_estimate, "elapsed_sec": time.time() - t0}]).to_csv(
            output_dir / "point_estimate.csv",
            index=False,
        )

    completed = pd.DataFrame()
    completed_reps: set[int] = set()
    if args.resume and draws_path.exists():
        completed = pd.read_csv(draws_path)
        completed_reps = set(completed["rep"].astype(int).tolist())

    rng = np.random.default_rng(args.seed)
    rep_seeds = rng.integers(1, np.iinfo(np.int32).max, size=args.bootstrap_reps, dtype=np.int64)
    jobs = [
        (rep, int(rep_seeds[rep]), int(args.random_state))
        for rep in range(args.bootstrap_reps)
        if rep not in completed_reps
    ]

    rows: list[dict[str, float | int]] = []
    if not completed.empty:
        rows.extend(completed.to_dict("records"))

    if jobs:
        with ProcessPoolExecutor(
            max_workers=max(1, int(args.n_jobs)),
            initializer=_init_worker,
            initargs=(x, w, z, a, y),
        ) as executor:
            futures = [executor.submit(_run_bootstrap_rep, rep, seed, random_state) for rep, seed, random_state in jobs]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                done = len(rows)
                if done % max(1, int(args.save_every)) == 0 or done == args.bootstrap_reps:
                    pd.DataFrame(rows).sort_values("rep").to_csv(draws_path, index=False)
                    print(f"[bootstrap] completed {done}/{args.bootstrap_reps}", flush=True)

    draws_df = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)
    draws_df.to_csv(draws_path, index=False)

    summary = pd.DataFrame(
        [
            _summarize(draws_df["ate_boot_sample"], point_estimate),
            _summarize(draws_df["ate_original_sample"], point_estimate),
        ]
    )
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
