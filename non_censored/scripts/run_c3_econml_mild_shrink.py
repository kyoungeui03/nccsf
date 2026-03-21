#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.non_censored import MildShrinkNCCausalForestDML  # noqa: E402


def _split_cols(value: str):
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the legacy Old C3 non-censored model (single-model mild shrink) on a CSV dataset.")
    parser.add_argument("--train-csv", required=True, help="CSV used to fit the model.")
    parser.add_argument("--predict-csv", help="Optional CSV to score. Defaults to train CSV.")
    parser.add_argument("--x-cols", required=True, help="Comma-separated X columns.")
    parser.add_argument("--w-cols", required=True, help="Comma-separated W columns.")
    parser.add_argument("--z-cols", required=True, help="Comma-separated Z columns.")
    parser.add_argument("--treatment-col", required=True)
    parser.add_argument("--outcome-col", required=True)
    parser.add_argument("--truth-col", help="Optional true CATE column for evaluation.")
    parser.add_argument("--id-col", help="Optional ID column to keep in predictions output.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    predict_df = train_df if args.predict_csv is None else pd.read_csv(args.predict_csv)

    x_cols = _split_cols(args.x_cols)
    w_cols = _split_cols(args.w_cols)
    z_cols = _split_cols(args.z_cols)

    X_train = train_df[x_cols].to_numpy()
    W_train = train_df[w_cols].to_numpy()
    Z_train = train_df[z_cols].to_numpy()
    A_train = train_df[args.treatment_col].to_numpy()
    Y_train = train_df[args.outcome_col].to_numpy()

    X_pred = predict_df[x_cols].to_numpy()
    W_pred = predict_df[w_cols].to_numpy()
    Z_pred = predict_df[z_cols].to_numpy()

    X_train_full = MildShrinkNCCausalForestDML.stack_final_features(X_train, W_train, Z_train)
    X_pred_full = MildShrinkNCCausalForestDML.stack_final_features(X_pred, W_pred, Z_pred)

    model = MildShrinkNCCausalForestDML(
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=42,
        x_core_dim=len(x_cols),
        duplicate_proxies_in_nuisance=True,
        oracle=False,
        use_true_q=False,
        use_true_h=False,
        q_kind="logit",
        q_clip=0.02,
        y_clip_quantile=0.99,
        y_res_clip_percentiles=(1.0, 99.0),
        h_n_estimators=300,
        h_min_samples_leaf=20,
    )
    model.fit_nc(X_train_full, A_train, Y_train, Z_train, W_train)
    preds = model.effect(X_pred_full).ravel()

    prediction_df = pd.DataFrame({"pred_cate": preds})
    if args.id_col:
        prediction_df.insert(0, args.id_col, predict_df[args.id_col].to_numpy())
    prediction_path = output_dir / "predictions.csv"
    prediction_df.to_csv(prediction_path, index=False)

    summary = {
        "n_train": int(len(train_df)),
        "n_predict": int(len(predict_df)),
        "x_cols": x_cols,
        "w_cols": w_cols,
        "z_cols": z_cols,
        "treatment_col": args.treatment_col,
        "outcome_col": args.outcome_col,
        "prediction_path": str(prediction_path),
        "mean_pred_cate": float(np.mean(preds)),
        "std_pred_cate": float(np.std(preds)),
        "c3_variant": "old_c3_mild_shrink_single_model",
    }

    if args.truth_col and args.truth_col in predict_df.columns:
        truth = predict_df[args.truth_col].to_numpy()
        summary.update(
            {
                "truth_col": args.truth_col,
                "mean_true_cate": float(np.mean(truth)),
                "rmse": float(np.sqrt(np.mean((preds - truth) ** 2))),
                "pehe": float(np.sqrt(np.mean((preds - truth) ** 2))),
                "mae": float(np.mean(np.abs(preds - truth))),
                "pearson": float(np.corrcoef(preds, truth)[0, 1]),
                "bias": float(np.mean(preds - truth)),
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
