#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.methods import BestCurveLocalCensoredPCISurvivalForest  # noqa: E402


def _parse_cols(value: str) -> list[str]:
    cols = [item.strip() for item in value.split(",") if item.strip()]
    if not cols:
        raise argparse.ArgumentTypeError("Column list must not be empty.")
    return cols


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return {
        "mean_pred_cate": float(np.mean(pred)),
        "mean_true_cate": float(np.mean(truth)),
        "bias": float(np.mean(pred - truth)),
        "sqrt_pehe": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "mae": float(np.mean(np.abs(pred - truth))),
        "pearson": float(np.corrcoef(pred, truth)[0, 1]),
    }


def _extract_features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df.loc[:, cols].to_numpy(dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the refactored New C3 (BestCurveLocal) on a user-supplied survival dataset."
    )
    parser.add_argument("--train-csv", type=Path, required=True, help="Training CSV or Parquet.")
    parser.add_argument("--predict-csv", type=Path, default=None, help="Optional prediction CSV or Parquet.")
    parser.add_argument("--x-cols", type=_parse_cols, required=True, help="Comma-separated X columns.")
    parser.add_argument("--w-cols", type=_parse_cols, required=True, help="Comma-separated W proxy columns.")
    parser.add_argument("--z-cols", type=_parse_cols, required=True, help="Comma-separated Z proxy columns.")
    parser.add_argument("--treatment-col", required=True)
    parser.add_argument("--time-col", required=True)
    parser.add_argument("--event-col", required=True)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--truth-col", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "c3_bestcurve",
        help="Directory for predictions and summary artifacts.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target", choices=["RMST", "survival.probability"], default="RMST")
    parser.add_argument("--horizon", type=float, default=None)
    args = parser.parse_args()

    if args.target == "survival.probability" and args.horizon is None:
        raise ValueError("--horizon is required when --target=survival.probability")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = _read_frame(args.train_csv.resolve())
    predict_df = train_df if args.predict_csv is None else _read_frame(args.predict_csv.resolve())

    required_train_cols = [*args.x_cols, *args.w_cols, *args.z_cols, args.treatment_col, args.time_col, args.event_col]
    missing_train = [col for col in required_train_cols if col not in train_df.columns]
    if missing_train:
        raise ValueError(f"Missing training columns: {missing_train}")

    required_predict_cols = [*args.x_cols, *args.w_cols, *args.z_cols]
    if args.id_col is not None:
        required_predict_cols.append(args.id_col)
    if args.truth_col is not None:
        required_predict_cols.append(args.truth_col)
    missing_predict = [col for col in required_predict_cols if col not in predict_df.columns]
    if missing_predict:
        raise ValueError(f"Missing prediction columns: {missing_predict}")

    x_train = _extract_features(train_df, args.x_cols)
    w_train = _extract_features(train_df, args.w_cols)
    z_train = _extract_features(train_df, args.z_cols)
    a_train = train_df.loc[:, args.treatment_col].to_numpy(dtype=float)
    time_train = train_df.loc[:, args.time_col].to_numpy(dtype=float)
    event_train = train_df.loc[:, args.event_col].to_numpy(dtype=float)

    x_pred = _extract_features(predict_df, args.x_cols)
    w_pred = _extract_features(predict_df, args.w_cols)
    z_pred = _extract_features(predict_df, args.z_cols)

    model = BestCurveLocalCensoredPCISurvivalForest(
        target=args.target,
        horizon=args.horizon,
        random_state=args.random_state,
    )

    t0 = time.time()
    model.fit_components(x_train, a_train, time_train, event_train, z_train, w_train)
    pred_cate = model.effect_from_components(x_pred, w_pred, z_pred).ravel()
    elapsed = time.time() - t0

    prediction_df = pd.DataFrame({"pred_cate": pred_cate})
    if args.id_col is not None:
        prediction_df.insert(0, args.id_col, predict_df.loc[:, args.id_col].to_numpy())
    if args.truth_col is not None:
        prediction_df[args.truth_col] = predict_df.loc[:, args.truth_col].to_numpy(dtype=float)

    summary = {
        "model": "New C3 BestCurveLocal",
        "backend": "grf.methods.BestCurveLocalCensoredPCISurvivalForest",
        "train_csv": str(args.train_csv.resolve()),
        "predict_csv": str(args.predict_csv.resolve()) if args.predict_csv is not None else str(args.train_csv.resolve()),
        "x_cols": args.x_cols,
        "w_cols": args.w_cols,
        "z_cols": args.z_cols,
        "treatment_col": args.treatment_col,
        "time_col": args.time_col,
        "event_col": args.event_col,
        "n_train": int(len(train_df)),
        "n_predict": int(len(predict_df)),
        "time_sec": float(elapsed),
        "random_state": args.random_state,
        "target": args.target,
        "horizon": args.horizon,
        "summary_feature_mode": "curve_x_interact",
        "nuisance_feature_mode": "interact",
        "n_estimators": 400,
        "min_samples_leaf": 30,
        "summary_curve_knots": 8,
        "censoring_estimator": "cox",
    }
    if args.truth_col is not None:
        summary.update(_metrics(pred_cate, predict_df.loc[:, args.truth_col].to_numpy(dtype=float)))

    prediction_path = output_dir / "predictions.csv"
    summary_path = output_dir / "summary.json"
    prediction_df.to_csv(prediction_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"New C3 BestCurveLocal completed in {elapsed:.2f}s")
    print(f"Saved predictions: {prediction_path}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
