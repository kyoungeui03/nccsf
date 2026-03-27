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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))
ROOT_SCRIPTS = PROJECT_ROOT / "scripts"
if str(ROOT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS))

from grf.non_censored import BaselineCausalForestDML, BestCurveLocalNCCausalForest, MildShrinkNCCausalForestDML  # noqa: E402
from preprocess_rhc import MCF_POLICY_FEATURES, build_rhc_mcf_matched  # noqa: E402


MCF_REFERENCE = {
    "Paper": "Bodory et al. (mcf)",
    "Outcome": "6-month survival / mortality",
    "ATE": 0.0480,
    "ATET": 0.0650,
    "IATE Mean": -0.0400,
    "IATE Range": "[-0.12, 0.08]",
    "Notes": "IATE has mass on both sides of zero; 6 of 8 policy features show GATE-ATE differences.",
}


CAT1_LABELS = {
    0: "ARF",
    1: "CHF",
    2: "COPD",
    3: "Cirrhosis",
    4: "Colon Cancer",
    5: "Coma",
    6: "Lung Cancer",
    7: "MOSF w/Malignancy",
    8: "MOSF w/Sepsis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an MCF-matched RHC benchmark for B2 / Old C3 / New C3 using 6-month fixed-horizon survival."
    )
    parser.add_argument("--input-csv", type=Path, default=PROJECT_ROOT / "data" / "rhc" / "raw_rhc.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rhc_mcf_matched_b2_old_new_c3",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-trees-b2", type=int, default=200)
    return parser.parse_args()


def crossfit_dr_components(
    features: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
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

        e_model = LogisticRegression(max_iter=5000, solver="liblinear")
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


def summarize_predictions(pred: np.ndarray, treatment: np.ndarray, psi1: np.ndarray, psi0: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    tau_dr = np.asarray(psi1 - psi0, dtype=float)
    treated_mask = treatment == 1
    return {
        "Mean Pred CATE": float(np.mean(pred)),
        "SD Pred CATE": float(np.std(pred)),
        "Q10 Pred CATE": float(np.quantile(pred, 0.10)),
        "Median Pred CATE": float(np.median(pred)),
        "Q90 Pred CATE": float(np.quantile(pred, 0.90)),
        "Pct Positive Pred": float(np.mean(pred > 0)),
        "DR ATE (all)": float(np.mean(tau_dr)),
        "DR ATET": float(np.mean(tau_dr[treated_mask])),
        "IATE Mean": float(np.mean(pred)),
        "IATE Range": f"[{np.min(pred):.4f}, {np.max(pred):.4f}]",
    }


def fit_b2(features_all: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, random_state: int) -> tuple[np.ndarray, float]:
    model = BaselineCausalForestDML(
        n_estimators=200,
        min_samples_leaf=20,
        random_state=random_state,
        cv=5,
    )
    t0 = time.time()
    model.fit_baseline(features_all, treatment, outcome)
    pred = model.effect(features_all).ravel()
    return pred, time.time() - t0


def fit_old_c3(
    x: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, float]:
    x_full = MildShrinkNCCausalForestDML.stack_final_features(x, w, z)
    model = MildShrinkNCCausalForestDML(
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=random_state,
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
    t0 = time.time()
    model.fit_nc(x_full, treatment, outcome, z, w)
    pred = model.effect(x_full).ravel()
    return pred, time.time() - t0


def fit_new_c3(
    x: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, float]:
    model = BestCurveLocalNCCausalForest(random_state=random_state)
    t0 = time.time()
    model.fit_components(x, treatment, outcome, z, w)
    pred = model.effect_from_components(x, w, z).ravel()
    return pred, time.time() - t0


def render_table_png(df: pd.DataFrame, title: str, subtitle: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(14, 1.8 * len(df.columns)), 1.2 + 0.75 * (len(df) + 1)))
    fig.patch.set_facecolor("white")
    ax.set_axis_off()
    fmt_df = df.copy()
    for col in fmt_df.columns:
        if fmt_df[col].dtype.kind in {"f", "i"}:
            if "Pct " in col or "Rate" in col:
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
    table.set_fontsize(11)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
            cell.set_edgecolor("white")
            cell.set_linewidth(2.0)
        else:
            cell.set_facecolor("#f8fafc" if row % 2 else "#eef2f7")
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.75)
    fig.text(0.5, 0.95, title, ha="center", va="top", fontsize=20, fontweight="bold")
    fig.text(0.5, 0.90, subtitle, ha="center", va="top", fontsize=11, color="#475569")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_pdf(image_paths: list[Path], output_pdf: Path) -> None:
    with PdfPages(output_pdf) as pdf:
        for path in image_paths:
            img = plt.imread(path)
            height, width = img.shape[:2]
            fig = plt.figure(figsize=(width / 150, height / 150))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def make_policy_groups(series: pd.Series, feature_name: str) -> pd.Series:
    if feature_name == "cat1":
        return series.astype(int).map(CAT1_LABELS).astype(str)
    if feature_name == "dnr1":
        return series.astype(int).map({0: "No", 1: "Yes"}).astype(str)
    if series.nunique(dropna=False) <= 8:
        return series.astype(int).astype(str)
    quantiles = pd.qcut(series, q=4, duplicates="drop")
    return quantiles.astype(str)


def compute_gate_table(
    analysis_df: pd.DataFrame,
    pred_map: dict[str, np.ndarray],
    tau_dr: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for feature in MCF_POLICY_FEATURES:
        groups = make_policy_groups(analysis_df[feature], feature)
        for subgroup in pd.Index(groups).dropna().unique():
            mask = groups == subgroup
            rows.append(
                {
                    "feature": feature,
                    "subgroup": str(subgroup),
                    "n": int(mask.sum()),
                    "share": float(mask.mean()),
                    "dr_ate_subgroup": float(np.mean(tau_dr[mask])),
                    **{
                        f"{model_key}_mean_pred_cate": float(np.mean(pred_map[model_key][mask]))
                        for model_key in pred_map
                    },
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input_csv.resolve())
    analysis_df, metadata = build_rhc_mcf_matched(raw_df)
    analysis_csv = output_dir / "rhc_mcf_matched_analysis.csv"
    metadata_json = output_dir / "rhc_mcf_matched_metadata.json"
    analysis_df.to_csv(analysis_csv, index=False)
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    x_cols = metadata["x_cols"]
    w_cols = metadata["w_cols"]
    z_cols = metadata["z_cols"]

    x = analysis_df[x_cols].to_numpy(dtype=float)
    w = analysis_df[w_cols].to_numpy(dtype=float)
    z = analysis_df[z_cols].to_numpy(dtype=float)
    a = analysis_df["A"].to_numpy(dtype=float)
    y = analysis_df["Y"].to_numpy(dtype=float)
    features_all = analysis_df[[*x_cols, *z_cols, *w_cols]].to_numpy(dtype=float)

    psi1, psi0 = crossfit_dr_components(features_all, a, y, args.cv_folds, args.random_state)
    tau_dr = psi1 - psi0

    print("[fit] B2")
    b2_pred, b2_time = fit_b2(features_all, a, y, args.random_state)
    print(f"[done] B2 in {b2_time:.1f}s")
    print("[fit] Old C3")
    old_pred, old_time = fit_old_c3(x, w, z, a, y, args.random_state)
    print(f"[done] Old C3 in {old_time:.1f}s")
    print("[fit] New C3")
    new_pred, new_time = fit_new_c3(x, w, z, a, y, args.random_state)
    print(f"[done] New C3 in {new_time:.1f}s")

    pred_map = {
        "B2": b2_pred,
        "OldC3": old_pred,
        "NewC3": new_pred,
    }

    result_rows = []
    for label, pred, elapsed in [
        ("B2", b2_pred, b2_time),
        ("Old C3", old_pred, old_time),
        ("New C3", new_pred, new_time),
    ]:
        metrics = summarize_predictions(pred, a.astype(int), psi1, psi0)
        result_rows.append(
            {
                "Model": label,
                **metrics,
                "Runtime (s)": elapsed,
                "Outcome": "6-month survival (paper-matched dth30 coding)",
            }
        )
    results_df = pd.DataFrame(result_rows)
    results_csv = output_dir / "rhc_mcf_matched_results.csv"
    results_df.to_csv(results_csv, index=False)

    predictions_df = pd.DataFrame(
        {
            "A": a,
            "Y": y,
            "tau_dr": tau_dr,
            "pred_b2": b2_pred,
            "pred_old_c3": old_pred,
            "pred_new_c3": new_pred,
        }
    )
    predictions_csv = output_dir / "rhc_mcf_matched_predictions.csv"
    predictions_df.to_csv(predictions_csv, index=False)

    gate_df = compute_gate_table(analysis_df, pred_map, tau_dr)
    gate_csv = output_dir / "rhc_mcf_matched_gate.csv"
    gate_df.to_csv(gate_csv, index=False)
    print(f"[done] GATE table saved to {gate_csv}")

    comparison_df = pd.DataFrame(
        [
            {
                "Model": MCF_REFERENCE["Paper"],
                "Outcome": MCF_REFERENCE["Outcome"],
                "ATE": MCF_REFERENCE["ATE"],
                "ATET": MCF_REFERENCE["ATET"],
                "IATE Mean": MCF_REFERENCE["IATE Mean"],
                "IATE Range": MCF_REFERENCE["IATE Range"],
                "Notes": MCF_REFERENCE["Notes"],
            },
            *[
                {
                    "Model": row["Model"],
                    "Outcome": row["Outcome"],
                    "ATE": row["DR ATE (all)"],
                    "ATET": row["DR ATET"],
                    "IATE Mean": row["IATE Mean"],
                    "IATE Range": row["IATE Range"],
                    "Notes": "Predicted individual treatment effects on the paper-matched fixed-horizon survival scale.",
                }
                for row in result_rows
            ],
        ]
    )
    comparison_csv = output_dir / "rhc_mcf_matched_paper_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    cohort_df = pd.DataFrame(
        [
            {
                "n": metadata["n"],
                "Treatment Rate": metadata["treatment_rate"],
                "Control 6m Survival": metadata["control_survival_rate"],
                "Treated 6m Survival": metadata["treated_survival_rate"],
                "Num X": len(x_cols),
                "Num W": len(w_cols),
                "Num Z": len(z_cols),
            }
        ]
    )

    cohort_png = output_dir / "rhc_mcf_matched_cohort.png"
    results_png = output_dir / "rhc_mcf_matched_results.png"
    comparison_png = output_dir / "rhc_mcf_matched_paper_comparison.png"
    render_table_png(
        cohort_df,
        "RHC MCF-matched cohort overview",
        "n=5735, 55 confounders, 8 policy features, outcome coded as 6-month survival from dth30",
        cohort_png,
    )
    render_table_png(
        results_df.drop(columns=["Outcome", "IATE Range"]),
        "RHC MCF-matched benchmark",
        "B2 vs Old C3 vs New C3 on the paper-matched fixed-horizon survival outcome",
        results_png,
    )
    render_table_png(
        comparison_df,
        "RHC paper comparison",
        "mcf reference row plus our matched-setting benchmark rows",
        comparison_png,
    )
    pdf_path = output_dir / "rhc_mcf_matched_bundle.pdf"
    render_pdf([cohort_png, results_png, comparison_png], pdf_path)

    print(f"Saved analysis CSV: {analysis_csv}")
    print(f"Saved metadata JSON: {metadata_json}")
    print(f"Saved results CSV: {results_csv}")
    print(f"Saved predictions CSV: {predictions_csv}")
    print(f"Saved GATE CSV: {gate_csv}")
    print(f"Saved comparison CSV: {comparison_csv}")
    print(f"Saved PDF bundle: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
