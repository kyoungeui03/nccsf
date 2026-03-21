#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))
ROOT_SCRIPTS = PROJECT_ROOT / "scripts"
if str(ROOT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS))

from grf.methods import EconmlMildShrinkNCSurvivalForest  # noqa: E402
from preprocess_rhc import build_cleaned_rhc  # noqa: E402


R_CSF_SCRIPT = PROJECT_ROOT / "scripts" / "run_grf_csf_baseline.R"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B2 vs legacy Old C3 on raw RHC for 30-day RMST.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "rhc" / "raw_rhc.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rhc_b2_vs_old_c3_30day",
    )
    parser.add_argument("--num-trees-b2", type=int, default=200)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def build_analysis_table(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    cleaned = build_cleaned_rhc(raw_df)
    analysis_df = pd.DataFrame(
        {
            "time": raw_df["t3d30"].astype(float),
            "event": (raw_df["dth30"] == "Yes").astype(int),
            "survival_probability_30": (raw_df["dth30"] != "Yes").astype(float),
            "rmst_30": raw_df["t3d30"].astype(float),
            "A": cleaned["A"].astype(int),
        }
    )
    feature_df = cleaned.drop(columns=["Y", "A", "lstctdte"], errors="ignore").copy()
    analysis_df = pd.concat([analysis_df, feature_df], axis=1)

    z_cols = [col for col in ["pafi1", "paco21"] if col in analysis_df.columns]
    w_cols = [col for col in ["ph1", "hema1"] if col in analysis_df.columns]
    x_cols = [col for col in analysis_df.columns if col not in {"time", "event", "survival_probability_30", "rmst_30", "A", *z_cols, *w_cols}]
    metadata = {
        "n": int(len(analysis_df)),
        "treatment_rate": float(analysis_df["A"].mean()),
        "event_rate_30d": float(analysis_df["event"].mean()),
        "survival_rate_30d": float(analysis_df["survival_probability_30"].mean()),
        "mean_rmst_30": float(analysis_df["rmst_30"].mean()),
        "x_cols": x_cols,
        "w_cols": w_cols,
        "z_cols": z_cols,
    }
    return analysis_df, metadata


def evaluate_r_csf_variant(obs_df: pd.DataFrame, feature_cols: list[str], horizon: float, num_trees: int, target: str) -> tuple[np.ndarray, float]:
    (PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs", prefix="rhc_r_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        input_path = tmp_dir / "input.csv"
        output_path = tmp_dir / "predictions.csv"
        obs_df.loc[:, ["time", "event", "A", *feature_cols]].to_csv(input_path, index=False)
        cmd = [
            "Rscript",
            str(R_CSF_SCRIPT),
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
            raise RuntimeError(f"Installed R grf baseline failed.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        preds = pd.read_csv(output_path)["prediction"].to_numpy(dtype=float)
    return preds, elapsed


def evaluate_c3_variant(X: np.ndarray, W: np.ndarray, Z: np.ndarray, A: np.ndarray, time_obs: np.ndarray, event: np.ndarray, *, target: str, horizon: float) -> tuple[np.ndarray, float]:
    model = EconmlMildShrinkNCSurvivalForest(
        target=target,
        horizon=float(horizon),
        n_estimators=200,
        min_samples_leaf=20,
        cv=3,
        random_state=42,
    )
    t0 = time.time()
    model.fit_components(X, A, time_obs, event, Z, W)
    preds = model.effect_from_components(X, W, Z).ravel()
    elapsed = time.time() - t0
    return preds, elapsed


def crossfit_dr_components(features: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, n_splits: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(outcome)
    e_hat = np.zeros(n, dtype=float)
    mu1_hat = np.zeros(n, dtype=float)
    mu0_hat = np.zeros(n, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(features):
        x_tr = features[train_idx]
        x_te = features[test_idx]
        a_tr = treatment[train_idx]
        y_tr = outcome[train_idx]

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


def policy_metrics(pred: np.ndarray, psi1: np.ndarray, psi0: np.ndarray, top_frac: float = 0.20) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    psi1 = np.asarray(psi1, dtype=float)
    psi0 = np.asarray(psi0, dtype=float)
    tau_dr = psi1 - psi0

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


def subgroup_curve(pred: np.ndarray, tau_dr: np.ndarray, quantiles: list[float]) -> pd.DataFrame:
    rows = []
    for q in quantiles:
        cutoff = np.quantile(pred, 1.0 - q)
        selected = pred >= cutoff
        rows.append({"share_treated_by_policy": q, "dr_ate_in_selected_group": float(np.mean(tau_dr[selected]))})
    return pd.DataFrame(rows)


def render_table_png(df: pd.DataFrame, title: str, subtitle: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(12, 1.8 * len(df.columns)), 1.2 + 0.75 * (len(df) + 1)))
    fig.patch.set_facecolor("white")
    ax.set_axis_off()
    fmt_df = df.copy()
    for col in fmt_df.columns:
        if fmt_df[col].dtype.kind in {"f", "i"}:
            if "pct_" in col:
                fmt_df[col] = (fmt_df[col] * 100.0).map(lambda x: f"{x:.1f}%")
            else:
                fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x))
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


def render_curve_png(curve_df: pd.DataFrame, target_label: str, output_path: Path) -> None:
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
    ax.set_title(f"RHC 30-day {target_label}: B2 vs C3 ranking curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_pdf(bundle_paths: list[Path], output_pdf: Path) -> None:
    with PdfPages(output_pdf) as pdf:
        for path in bundle_paths:
            img = plt.imread(path)
            height, width = img.shape[:2]
            fig = plt.figure(figsize=(width / 150, height / 150))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input_csv)
    analysis_df, metadata = build_analysis_table(raw_df)
    x_cols = metadata["x_cols"]
    w_cols = metadata["w_cols"]
    z_cols = metadata["z_cols"]
    feature_cols_b2 = x_cols + w_cols + z_cols

    analysis_csv = output_dir / "rhc_30day_survival_analysis.csv"
    analysis_json = output_dir / "rhc_30day_survival_metadata.json"
    analysis_df.to_csv(analysis_csv, index=False)
    analysis_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    X = analysis_df[x_cols].to_numpy(dtype=float)
    W = analysis_df[w_cols].to_numpy(dtype=float)
    Z = analysis_df[z_cols].to_numpy(dtype=float)
    A = analysis_df["A"].to_numpy(dtype=float)
    time_obs = analysis_df["time"].to_numpy(dtype=float)
    event = analysis_df["event"].to_numpy(dtype=float)
    x_obs = analysis_df[feature_cols_b2].to_numpy(dtype=float)

    cohort_df = pd.DataFrame(
        [
            {
                "n": metadata["n"],
                "treatment_rate": metadata["treatment_rate"],
                "event_rate_30d": metadata["event_rate_30d"],
                "survival_rate_30d": metadata["survival_rate_30d"],
                "mean_rmst_30": metadata["mean_rmst_30"],
                "num_x_features": len(x_cols),
                "num_w_features": len(w_cols),
                "num_z_features": len(z_cols),
            }
        ]
    )
    cohort_png = output_dir / "rhc_cohort_overview.png"
    render_table_png(
        cohort_df,
        "RHC 30-day cohort overview",
        "Raw RHC with treatment A=swang1, time=t3d30, event=dth30",
        cohort_png,
    )

    bundle_paths = [cohort_png]
    target_summaries = []
    all_curves = []

    for target, horizon, outcome_col, target_label in [
        ("RMST", 30.0, "rmst_30", "RMST at 30 days"),
        ("survival.probability", 30.0, "survival_probability_30", "survival probability at 30 days"),
    ]:
        outcome = analysis_df[outcome_col].to_numpy(dtype=float)
        psi1, psi0 = crossfit_dr_components(x_obs, A, outcome, n_splits=args.cv_folds, random_state=args.random_state)
        tau_dr = psi1 - psi0

        b2_pred, b2_time = evaluate_r_csf_variant(analysis_df, feature_cols_b2, horizon, args.num_trees_b2, target)
        c3_pred, c3_time = evaluate_c3_variant(X, W, Z, A, time_obs, event, target=target, horizon=horizon)

        model_rows = []
        for model_name, preds, elapsed, backend in [
            ("B2  R-CSF baseline (X+W+Z)", b2_pred, b2_time, "installed R grf"),
            ("C3  NC-CSF econml mild shrink", c3_pred, c3_time, "econml"),
        ]:
            metrics = policy_metrics(preds, psi1, psi0)
            model_rows.append(
                {
                    "Model": model_name,
                    "Mean Pred CATE": metrics["mean_pred_cate"],
                    "SD Pred CATE": metrics["std_pred_cate"],
                    "Median Pred CATE": metrics["median_pred_cate"],
                    "Pct Positive Pred": metrics["pct_positive_pred"],
                    "DR ATE (all)": metrics["dr_ate_all"],
                    "DR ATE (top20)": metrics["dr_ate_top20"],
                    "Policy Gain vs None (positive)": metrics["policy_gain_positive_vs_none"],
                    "Policy Gain vs None (top20)": metrics["policy_gain_top20_vs_none"],
                    "Time (s)": elapsed,
                    "Backend": backend,
                    "target": target,
                }
            )
            curve = subgroup_curve(preds, tau_dr, quantiles=[0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00])
            curve["model"] = model_name
            curve["target"] = target
            all_curves.append(curve)

        results_df = pd.DataFrame(model_rows)
        target_summaries.append(results_df)

        results_csv = output_dir / f"rhc_b2_vs_c3_{target.replace('.', '_')}_results.csv"
        results_df.to_csv(results_csv, index=False)

        table_png = output_dir / f"rhc_b2_vs_c3_{target.replace('.', '_')}_table.png"
        render_table_png(
            results_df.drop(columns=["target"]),
            f"RHC 30-day {target_label}",
            "B2 vs C3 on raw RHC using doubly robust policy-style evaluation",
            table_png,
        )
        bundle_paths.append(table_png)

        curve_df = pd.concat([df for df in all_curves if df["target"].iloc[0] == target], ignore_index=True)
        curve_csv = output_dir / f"rhc_b2_vs_c3_{target.replace('.', '_')}_curve.csv"
        curve_df.to_csv(curve_csv, index=False)
        curve_png = output_dir / f"rhc_b2_vs_c3_{target.replace('.', '_')}_curve.png"
        render_curve_png(curve_df, target_label, curve_png)
        bundle_paths.append(curve_png)

    summary_csv = output_dir / "rhc_b2_vs_c3_summary_long.csv"
    pd.concat(target_summaries, ignore_index=True).to_csv(summary_csv, index=False)

    pdf_path = output_dir / "rhc_b2_vs_c3_results_bundle.pdf"
    render_pdf(bundle_paths, pdf_path)

    rmst_pdf_path = output_dir / "rhc_b2_vs_c3_RMST_bundle.pdf"
    rmst_bundle = [
        cohort_png,
        output_dir / "rhc_b2_vs_c3_RMST_table.png",
        output_dir / "rhc_b2_vs_c3_RMST_curve.png",
    ]
    render_pdf(rmst_bundle, rmst_pdf_path)

    print(f"Saved analysis CSV: {analysis_csv}")
    print(f"Saved metadata JSON: {analysis_json}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved PDF bundle: {pdf_path}")
    print(f"Saved RMST PDF bundle: {rmst_pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
