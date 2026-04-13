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
ROOT_SCRIPTS = PROJECT_ROOT / "scripts"
if str(ROOT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS))

from grf.non_censored import FinalModelNCCausalForest, StrictEconmlXWZNCCausalForest  # noqa: E402
from preprocess_rhc import build_cleaned_rhc  # noqa: E402


MODEL_SPECS = [
    ("Final Model", FinalModelNCCausalForest),
    ("No-UMC Baseline", StrictEconmlXWZNCCausalForest),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a paper-style RHC analysis with binary 30-day survival outcome, "
            "Z=(pafi1,paco21), W=(ph1,hema1), and all remaining cleaned covariates as X."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=PROJECT_ROOT / "data" / "rhc" / "raw_rhc.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rhc_paper_style_binary_comparison",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bootstrap-reps", type=int, default=30)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    return parser.parse_args()


def _build_rhc_paper_style(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    cleaned = build_cleaned_rhc(raw_df).copy()
    cleaned["Y"] = (raw_df["dth30"] != "Yes").astype(float).to_numpy()
    z_cols = [col for col in ["pafi1", "paco21"] if col in cleaned.columns]
    w_cols = [col for col in ["ph1", "hema1"] if col in cleaned.columns]
    x_cols = [col for col in cleaned.columns if col not in {"Y", "A", *z_cols, *w_cols}]
    return cleaned.loc[:, ["Y", "A", *x_cols, *z_cols, *w_cols]], x_cols, w_cols, z_cols


def _fit_predict(model_cls, x, a, y, z, w, *, random_state: int) -> tuple[np.ndarray, float]:
    model = model_cls(random_state=random_state)
    t0 = time.time()
    model.fit_components(x, a, y, z, w)
    preds = np.asarray(model.effect_from_components(x, w, z), dtype=float).ravel()
    elapsed = time.time() - t0
    return preds, elapsed


def _bootstrap_ates(model_cls, x, a, y, z, w, *, reps: int, seed: int, random_state: int) -> np.ndarray:
    if reps <= 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    ates = np.empty(reps, dtype=float)
    for rep in range(reps):
        idx = rng.integers(0, n, size=n)
        model = model_cls(random_state=random_state + rep + 1)
        model.fit_components(x[idx], a[idx], y[idx], z[idx], w[idx])
        preds = np.asarray(model.effect_from_components(x, w, z), dtype=float).ravel()
        ates[rep] = float(np.mean(preds))
    return ates


def _summary_row(name: str, preds: np.ndarray, elapsed: float, boot_ates: np.ndarray) -> dict[str, object]:
    preds = np.asarray(preds, dtype=float).ravel()
    estimate = float(np.mean(preds))
    if boot_ates.size:
        se = float(np.std(boot_ates, ddof=1)) if boot_ates.size > 1 else float("nan")
        ci_low, ci_high = np.quantile(boot_ates, [0.025, 0.975])
        ci_low = float(ci_low)
        ci_high = float(ci_high)
    else:
        se = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
    return {
        "name": name,
        "n_obs": int(len(preds)),
        "estimate": estimate,
        "estimate_pct_points": float(100.0 * estimate),
        "se": se,
        "se_pct_points": float(100.0 * se) if np.isfinite(se) else float("nan"),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_low_pct_points": float(100.0 * ci_low) if np.isfinite(ci_low) else float("nan"),
        "ci_high_pct_points": float(100.0 * ci_high) if np.isfinite(ci_high) else float("nan"),
        "std_pred": float(np.std(preds)),
        "median_pred": float(np.median(preds)),
        "pct_positive": float(np.mean(preds > 0)),
        "min_pred": float(np.min(preds)),
        "max_pred": float(np.max(preds)),
        "time_sec": float(elapsed),
        "bootstrap_reps": int(boot_ates.size),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input_csv.resolve())
    analysis_df, x_cols, w_cols, z_cols = _build_rhc_paper_style(raw_df)

    x = analysis_df[x_cols].to_numpy(dtype=float)
    w = analysis_df[w_cols].to_numpy(dtype=float)
    z = analysis_df[z_cols].to_numpy(dtype=float)
    a = analysis_df["A"].to_numpy(dtype=float)
    y = analysis_df["Y"].to_numpy(dtype=float)

    prediction_frames = []
    summary_rows = []
    bootstrap_rows = []

    for name, model_cls in MODEL_SPECS:
        preds, elapsed = _fit_predict(
            model_cls,
            x,
            a,
            y,
            z,
            w,
            random_state=args.random_state,
        )
        boot_ates = _bootstrap_ates(
            model_cls,
            x,
            a,
            y,
            z,
            w,
            reps=args.bootstrap_reps,
            seed=args.bootstrap_seed,
            random_state=args.random_state,
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_id": np.arange(len(preds)),
                    "name": name,
                    "prediction": preds,
                }
            )
        )
        summary_rows.append(_summary_row(name, preds, elapsed, boot_ates))
        if boot_ates.size:
            bootstrap_rows.append(
                pd.DataFrame(
                    {
                        "name": name,
                        "bootstrap_rep": np.arange(len(boot_ates)),
                        "ate": boot_ates,
                    }
                )
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    paper_table = summary.loc[:, ["name", "estimate", "se", "ci_low", "ci_high", "n_obs", "time_sec"]].copy()

    treated_mask = analysis_df["A"].to_numpy(dtype=int) == 1
    control_mask = ~treated_mask
    metadata = {
        "dataset": str(args.input_csv.resolve()),
        "n_rows": int(len(analysis_df)),
        "raw_column_count": int(len(raw_df.columns)),
        "procedure": {
            "treatment": "A = 1 if swang1 indicates RHC within 24 hours",
            "outcome": "Y = 1[dth30 != 'Yes'] (30-day survival)",
            "z_proxies": z_cols,
            "w_proxies": w_cols,
            "x_definition": "All remaining cleaned covariates after removing A, Y, Z, W",
        },
        "feature_counts": {
            "x": int(len(x_cols)),
            "z": int(len(z_cols)),
            "w": int(len(w_cols)),
            "total_analysis_columns": int(len(analysis_df.columns)),
        },
        "x_cols": x_cols,
        "w_cols": w_cols,
        "z_cols": z_cols,
        "observed_survival_rate_30_treated": float(analysis_df.loc[treated_mask, "Y"].mean()),
        "observed_survival_rate_30_control": float(analysis_df.loc[control_mask, "Y"].mean()),
        "observed_survival_diff_30": float(analysis_df.loc[treated_mask, "Y"].mean() - analysis_df.loc[control_mask, "Y"].mean()),
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": int(args.bootstrap_seed),
        "notes": [
            "This follows the paper's binary-outcome RHC setup more closely than the censored-survival script.",
            "The local raw_rhc.csv has 63 columns, so this matches the available cleaned covariates rather than the original paper's literal 72-column count.",
            "The No-UMC baseline here is a generic X+W+Z causal forest, not an exact reimplementation of Vermeulen and Vansteelandt (2015).",
        ],
    }

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    paper_table.to_csv(output_dir / "paper_table.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if bootstrap_rows:
        pd.concat(bootstrap_rows, ignore_index=True).to_csv(output_dir / "bootstrap_ates.csv", index=False)

    print(f"Saved predictions: {output_dir / 'predictions.csv'}")
    print(f"Saved summary: {output_dir / 'summary.csv'}")
    print(f"Saved paper table: {output_dir / 'paper_table.csv'}")
    print(f"Saved metadata: {output_dir / 'metadata.json'}")
    if bootstrap_rows:
        print(f"Saved bootstrap ATEs: {output_dir / 'bootstrap_ates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
