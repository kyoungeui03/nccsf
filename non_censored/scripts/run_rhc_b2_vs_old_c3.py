#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))
ROOT_SCRIPTS = PROJECT_ROOT / "scripts"
if str(ROOT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS))

from grf.non_censored import BaselineCausalForestDML, MildShrinkNCCausalForestDML  # noqa: E402
from preprocess_rhc import build_cleaned_rhc  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run non-censored B2 vs legacy Old C3 on the historical RHC direct-outcome setup.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "rhc" / "raw_rhc.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "non_censored" / "outputs" / "rhc_b2_vs_old_c3",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def build_analysis_table(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    cleaned = build_cleaned_rhc(raw_df)
    z_cols = [col for col in ["pafi1", "paco21"] if col in cleaned.columns]
    w_cols = [col for col in ["ph1", "hema1"] if col in cleaned.columns]
    x_cols = [col for col in cleaned.columns if col not in {"Y", "A", *z_cols, *w_cols}]
    metadata = {
        "n": int(len(cleaned)),
        "treatment_rate": float(cleaned["A"].mean()),
        "mean_outcome": float(cleaned["Y"].mean()),
        "x_cols": x_cols,
        "w_cols": w_cols,
        "z_cols": z_cols,
    }
    return cleaned, metadata


def _stack_final_features(x: np.ndarray, w: np.ndarray, z: np.ndarray) -> np.ndarray:
    return MildShrinkNCCausalForestDML.stack_final_features(x, w, z)


def crossfit_predictions(
    x: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    pred_b2 = np.zeros(n, dtype=float)
    pred_c3 = np.zeros(n, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(x), start=1):
        x_tr, x_te = x[train_idx], x[test_idx]
        w_tr, w_te = w[train_idx], w[test_idx]
        z_tr, z_te = z[train_idx], z[test_idx]
        a_tr, y_tr = a[train_idx], y[train_idx]

        x_tr_full = _stack_final_features(x_tr, w_tr, z_tr)
        x_te_full = _stack_final_features(x_te, w_te, z_te)

        b2 = BaselineCausalForestDML(n_estimators=200, min_samples_leaf=20, random_state=random_state + fold_id, cv=5)
        b2.fit_baseline(x_tr_full, a_tr, y_tr)
        pred_b2[test_idx] = b2.effect(x_te_full).ravel()

        c3 = MildShrinkNCCausalForestDML(
            n_estimators=200,
            min_samples_leaf=20,
            cv=5,
            random_state=random_state + fold_id,
            x_core_dim=x.shape[1],
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
        c3.fit_nc(x_tr_full, a_tr, y_tr, z_tr, w_tr)
        pred_c3[test_idx] = c3.effect(x_te_full).ravel()

    return pred_b2, pred_c3


def crossfit_dr_components(features: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, n_splits: int, random_state: int):
    n = len(outcome)
    e_hat = np.zeros(n, dtype=float)
    mu1_hat = np.zeros(n, dtype=float)
    mu0_hat = np.zeros(n, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kf.split(features):
        x_tr, x_te = features[train_idx], features[test_idx]
        a_tr, y_tr = treatment[train_idx], outcome[train_idx]

        e_model = LogisticRegression(max_iter=10000)
        e_model.fit(x_tr, a_tr)
        e_hat[test_idx] = np.clip(e_model.predict_proba(x_te)[:, 1], 0.02, 0.98)

        mu1_model = RandomForestRegressor(n_estimators=300, min_samples_leaf=20, random_state=random_state)
        mu0_model = RandomForestRegressor(n_estimators=300, min_samples_leaf=20, random_state=random_state)
        mu1_model.fit(x_tr[a_tr == 1], y_tr[a_tr == 1])
        mu0_model.fit(x_tr[a_tr == 0], y_tr[a_tr == 0])
        mu1_hat[test_idx] = mu1_model.predict(x_te)
        mu0_hat[test_idx] = mu0_model.predict(x_te)

    psi1 = mu1_hat + treatment / e_hat * (outcome - mu1_hat)
    psi0 = mu0_hat + (1.0 - treatment) / (1.0 - e_hat) * (outcome - mu0_hat)
    return psi1, psi0


def policy_metrics(pred: np.ndarray, psi1: np.ndarray, psi0: np.ndarray, top_frac: float = 0.20):
    pred = np.asarray(pred, dtype=float)
    tau_dr = np.asarray(psi1 - psi0, dtype=float)

    positive_policy = (pred > 0).astype(int)
    positive_value = float(np.mean(positive_policy * psi1 + (1 - positive_policy) * psi0))
    positive_gain_vs_none = float(positive_value - np.mean(psi0))

    cutoff = np.quantile(pred, 1.0 - top_frac)
    top_policy = (pred >= cutoff).astype(int)
    top_value = float(np.mean(top_policy * psi1 + (1 - top_policy) * psi0))
    top_gain_vs_none = float(top_value - np.mean(psi0))
    top_subset = pred >= cutoff
    top_ate = float(np.mean(tau_dr[top_subset]))

    return {
        "mean_pred_cate": float(np.mean(pred)),
        "std_pred_cate": float(np.std(pred)),
        "median_pred_cate": float(np.median(pred)),
        "pct_positive_pred": float(np.mean(pred > 0)),
        "dr_ate_all": float(np.mean(tau_dr)),
        "policy_value_positive": positive_value,
        "policy_gain_positive_vs_none": positive_gain_vs_none,
        "policy_value_top20": top_value,
        "policy_gain_top20_vs_none": top_gain_vs_none,
        "dr_ate_top20": top_ate,
    }


def subgroup_curve(pred: np.ndarray, tau_dr: np.ndarray, quantiles):
    rows = []
    for q in quantiles:
        cutoff = np.quantile(pred, 1.0 - q)
        selected = pred >= cutoff
        rows.append({"share_treated_by_policy": q, "dr_ate_in_selected_group": float(np.mean(tau_dr[selected]))})
    return pd.DataFrame(rows)


def render_table_png(df: pd.DataFrame, title: str, subtitle: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(max(12, 1.8 * len(df.columns)), 1.2 + 0.75 * (len(df) + 1)))
    fig.patch.set_facecolor("white")
    ax.set_axis_off()
    fmt_df = df.copy()
    for col in fmt_df.columns:
        if fmt_df[col].dtype.kind in {"f", "i"}:
            if "pct_" in col:
                fmt_df[col] = (fmt_df[col] * 100.0).map(lambda x: f"{x:.1f}%")
            else:
                fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.4f}")
    table = ax.table(
        cellText=fmt_df.values,
        colLabels=fmt_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.02, 0.96, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
            cell.set_edgecolor("white")
            cell.set_linewidth(2.0)
        else:
            cell.set_facecolor("#f8fafc" if row % 2 else "#eef2f7")
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.75)
    fig.text(0.5, 0.95, title, ha="center", va="top", fontsize=22, fontweight="bold")
    fig.text(0.5, 0.90, subtitle, ha="center", va="top", fontsize=12, color="#475569")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_curve_png(curve_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, group in curve_df.groupby("model"):
        ax.plot(
            group["share_treated_by_policy"] * 100.0,
            group["dr_ate_in_selected_group"],
            marker="o",
            linewidth=2.5,
            label=name,
        )
    ax.set_xlabel("Top-k policy share treated (%)")
    ax.set_ylabel("DR estimated effect in selected group")
    ax.set_title("RHC direct outcome: non-censored B2 vs C3 ranking curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_pdf(bundle_paths, output_pdf: Path):
    with PdfPages(output_pdf) as pdf:
        for path in bundle_paths:
            img = plt.imread(path)
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(w / 150, h / 150))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input_csv.resolve())
    analysis_df, metadata = build_analysis_table(raw_df)
    analysis_df.to_csv(output_dir / "rhc_direct_outcome_analysis.csv", index=False)
    (output_dir / "rhc_direct_outcome_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    x_cols = metadata["x_cols"]
    w_cols = metadata["w_cols"]
    z_cols = metadata["z_cols"]
    x = analysis_df.loc[:, x_cols].to_numpy(dtype=float)
    w = analysis_df.loc[:, w_cols].to_numpy(dtype=float)
    z = analysis_df.loc[:, z_cols].to_numpy(dtype=float)
    a = analysis_df["A"].to_numpy(dtype=float)
    y = analysis_df["Y"].to_numpy(dtype=float)

    t0 = time.time()
    pred_b2, pred_c3 = crossfit_predictions(
        x,
        w,
        z,
        a,
        y,
        n_splits=args.cv_folds,
        random_state=args.random_state,
    )
    elapsed = time.time() - t0

    features_all = _stack_final_features(x, w, z)
    psi1, psi0 = crossfit_dr_components(features_all, a, y, n_splits=args.cv_folds, random_state=args.random_state)
    tau_dr = psi1 - psi0

    rows = []
    for model_name, pred in [
        ("B2  EconML baseline (X+W+Z)", pred_b2),
        ("C3  NC-CSF Mild Shrink", pred_c3),
    ]:
        metrics = policy_metrics(pred, psi1, psi0)
        metrics["model"] = model_name
        rows.append(metrics)
    summary_df = pd.DataFrame(rows)[
        [
            "model",
            "mean_pred_cate",
            "std_pred_cate",
            "median_pred_cate",
            "pct_positive_pred",
            "dr_ate_all",
            "dr_ate_top20",
            "policy_gain_positive_vs_none",
            "policy_gain_top20_vs_none",
            "policy_value_positive",
            "policy_value_top20",
        ]
    ]
    summary_df.to_csv(output_dir / "rhc_b2_vs_c3_summary_long.csv", index=False)

    curve_frames = []
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5]
    for model_name, pred in [
        ("B2  EconML baseline (X+W+Z)", pred_b2),
        ("C3  NC-CSF Mild Shrink", pred_c3),
    ]:
        curve = subgroup_curve(pred, tau_dr, quantiles)
        curve["model"] = model_name
        curve_frames.append(curve)
    curve_df = pd.concat(curve_frames, ignore_index=True)
    curve_df.to_csv(output_dir / "rhc_b2_vs_c3_curve.csv", index=False)

    prediction_df = pd.DataFrame(
        {
            "A": a,
            "Y": y,
            "pred_b2": pred_b2,
            "pred_c3": pred_c3,
            "tau_dr": tau_dr,
        }
    )
    prediction_df.to_csv(output_dir / "rhc_b2_vs_c3_predictions.csv", index=False)

    subtitle = (
        f"n={metadata['n']}, treatment rate={metadata['treatment_rate']:.1%}, "
        f"mean outcome={metadata['mean_outcome']:.2f}, total runtime={elapsed:.1f}s"
    )
    summary_png = output_dir / "rhc_b2_vs_c3_summary.png"
    curve_png = output_dir / "rhc_b2_vs_c3_curve.png"
    render_table_png(summary_df, "RHC non-censored B2 vs C3", subtitle, summary_png)
    render_curve_png(curve_df, curve_png)
    render_pdf([summary_png, curve_png], output_dir / "rhc_b2_vs_c3_bundle.pdf")


if __name__ == "__main__":
    main()
