#!/usr/bin/env python3
"""Paper-style ACTG175 HIV runner with bootstrap, BLP, and R-CSF comparison.

This script reproduces the real-data setup from Section 5 of Cui et al. (2023)
on the ACTG175 HIV dataset and extends it to our single-file models.

Core dataset / setup choices mirrored from the paper:
    - dataset: ACTG175 from UCI / ucimlrepo (dataset id 890)
    - treatment subset: ZDV+ddI vs ddI monotherapy
    - treatment mapping: A=1 for ZDV+ddI, A=0 for ddI
    - target: RMST
    - horizon: h = 1000 days
    - 14 covariates used in the HIV application

Model variants:
    1. final_conditional_x14:
       our Final Conditional model with all 14 paper covariates passed via X.
    2. final_conditional_x12_w2_z2:
       a clinical split with 12 baseline covariates in X, 2 follow-up immune
       markers in W, and 2 treatment-selection proxies in Z.
    3. final_conditional_rec_a:
       recommendation A: stable patient profile in X, severity / progression
       markers in W, and treatment-selection history in Z.
    4. final_conditional_rec_b:
       recommendation B: broad demographic / behavior profile in X, lab markers
       in W, and clinician decision proxies in Z.
    5. r_csf_x14:
       installed R grf::causal_survival_forest baseline on the same 14
       covariates.

Paper-style outputs:
    - Figure 3 style histogram of observed follow-up time
    - Figure 4 style age sweep with bootstrap confidence bands
    - Table 4 style random sample of 10 subjects with bootstrap SEs
    - Table 5 style BLP regressions using HC3 robust standard errors
    - direct comparison artifacts across the selected models
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ucimlrepo import fetch_ucirepo

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_FILE = Path(__file__).resolve()
MODEL_DIR = THIS_FILE.parent
PROJECT_ROOT = MODEL_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

try:  # pragma: no cover
    from .final_censored_model_conditional import FinalModelConditionalCensoredSurvivalForest
    from .grf_censored_baseline import GRFCensoredBaseline
except ImportError:  # pragma: no cover
    from single_file_censored_models.final_censored_model_conditional import (  # type: ignore
        FinalModelConditionalCensoredSurvivalForest,
    )
    from single_file_censored_models.grf_censored_baseline import GRFCensoredBaseline  # type: ignore


PAPER_HORIZON = 1000.0
DATASET_ID = 890
EPS = 1e-8

PAPER_14_COVARIATES = [
    "age",
    "wtkg",
    "karnof",
    "cd40",
    "cd80",
    "gender",
    "homo",
    "race",
    "symptom",
    "drugs",
    "hemo",
    "str2",
    "cd420",
    "cd820",
]

TABLE4_STYLE_COLUMNS = ["hemo", "gender", "homo", "str2"]

BLP_LABELS = {
    "const": "Constant",
    "age": "Age",
    "wtkg": "Weight",
    "karnof": "Karnofsky score",
    "cd40": "CD4 count",
    "cd80": "CD8 count",
    "gender": "Gender",
    "homo": "Homosexual activity",
    "race": "Race",
    "symptom": "Symptomatic status",
    "drugs": "Intravenous drug use",
    "hemo": "Haemophilia",
    "str2": "Antiretroviral history",
    "cd420": "CD4 count 20± 5 weeks",
    "cd820": "CD8 count 20± 5 weeks",
}

MODEL_SPECS = [
    {
        "model_name": "final_conditional_x14",
        "display_name": "Final Conditional (All 14 in X)",
        "model_kind": "final_conditional",
        "x_cols": PAPER_14_COVARIATES,
        "w_cols": [],
        "z_cols": [],
        "w_placeholder_zero": True,
        "z_placeholder_zero": True,
        "notes": (
            "Paper-faithful covariate inclusion: all 14 ACTG175 covariates are passed "
            "through X. EconML does not accept 0-column W/Z arrays, so this runner passes "
            "single all-zero placeholder columns for W and Z. They act as implementation "
            "placeholders rather than informative proxies."
        ),
    },
    {
        "model_name": "final_conditional_x12_w2_z2",
        "display_name": "Final Conditional (X=12 baseline, W=2 immune, Z=2 selection)",
        "model_kind": "final_conditional",
        "x_cols": [
            "age",
            "wtkg",
            "karnof",
            "cd40",
            "cd80",
            "gender",
            "homo",
            "race",
            "symptom",
            "drugs",
            "hemo",
            "str2",
        ],
        "w_cols": ["cd420", "cd820"],
        "z_cols": ["race", "str2"],
        "w_placeholder_zero": False,
        "z_placeholder_zero": False,
        "notes": (
            "Clinical baseline split: X keeps the 12 baseline paper covariates, W uses the two "
            "20-week immune markers (cd420, cd820) as outcome-side progression proxies, and Z "
            "uses race and prior antiretroviral history as treatment-selection proxies. This "
            "keeps all 14 paper covariates while exposing explicit PCI channels."
        ),
    },
    {
        "model_name": "final_conditional_rec_a",
        "display_name": "Final Conditional (Rec A: profile X, severity W, history Z)",
        "model_kind": "final_conditional",
        "x_cols": ["age", "wtkg", "gender", "homo", "race"],
        "w_cols": ["karnof", "symptom", "cd40", "cd80", "cd420", "cd820"],
        "z_cols": ["str2", "drugs", "hemo"],
        "w_placeholder_zero": False,
        "z_placeholder_zero": False,
        "notes": (
            "Recommendation A: X holds relatively stable background descriptors, W holds disease "
            "severity and immune progression summaries, and Z holds variables that a clinician "
            "could plausibly use when choosing regimen intensity or anticipating adherence risk."
        ),
    },
    {
        "model_name": "final_conditional_rec_b",
        "display_name": "Final Conditional (Rec B: broad X, lab W, decision Z)",
        "model_kind": "final_conditional",
        "x_cols": ["age", "wtkg", "gender", "homo", "race", "drugs"],
        "w_cols": ["cd40", "cd80", "cd420", "cd820"],
        "z_cols": ["karnof", "symptom", "str2", "hemo"],
        "w_placeholder_zero": False,
        "z_placeholder_zero": False,
        "notes": (
            "Recommendation B: X keeps broad patient profile features, W isolates lab-based immune "
            "measurements, and Z concentrates variables that can drive treatment assignment or "
            "physician concern at decision time."
        ),
    },
    {
        "model_name": "r_csf_x14",
        "display_name": "R-CSF Baseline (All 14 in X)",
        "model_kind": "r_csf",
        "x_cols": PAPER_14_COVARIATES,
        "w_cols": [],
        "z_cols": [],
        "w_placeholder_zero": False,
        "z_placeholder_zero": False,
        "notes": (
            "Installed R grf::causal_survival_forest baseline using the same 14 paper covariates "
            "as ordinary forest inputs."
        ),
    },
]

DEFAULT_MODEL_NAMES = [
    "final_conditional_x14",
    "final_conditional_x12_w2_z2",
    "final_conditional_rec_a",
    "final_conditional_rec_b",
]


def _empty_block(n_rows: int) -> np.ndarray:
    return np.empty((int(n_rows), 0), dtype=float)


def _zero_placeholder_block(n_rows: int) -> np.ndarray:
    return np.zeros((int(n_rows), 1), dtype=float)


def _cleanup_model(model) -> None:
    cleanup = getattr(model, "cleanup", None)
    if callable(cleanup):
        cleanup()


def fetch_actg175_dataframe(*, cache_csv: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Fetch the UCI ACTG175 dataset and return a merged pandas DataFrame."""

    if cache_csv is not None and cache_csv.exists() and not refresh:
        return pd.read_csv(cache_csv)

    dataset = fetch_ucirepo(id=DATASET_ID)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    df = pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)
    if cache_csv is not None:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
    return df


def prepare_actg175_paper_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact ddI vs ZDV+ddI subset used in the HIV application."""

    subset = df.loc[df["trt"].isin([1, 3]), :].copy()
    subset["A"] = (subset["trt"] == 1).astype(int)
    subset["time_obs"] = subset["time"].astype(float)
    subset["event"] = subset["cid"].astype(int)
    subset["treatment_label"] = np.where(subset["A"] == 1, "ZDV+ddI", "ddI")
    subset["subject_id"] = np.arange(len(subset), dtype=int)
    subset.reset_index(drop=True, inplace=True)

    treated_n = int((subset["A"] == 1).sum())
    control_n = int((subset["A"] == 0).sum())
    if len(subset) != 1083 or treated_n != 522 or control_n != 561:
        raise RuntimeError(
            "ACTG175 paper subset validation failed. "
            f"Expected n=1083 with treated=522 and control=561, got "
            f"n={len(subset)}, treated={treated_n}, control={control_n}."
        )
    return subset


def _block_from_columns(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return _empty_block(len(df))
    return df.loc[:, cols].to_numpy(dtype=float)


def build_feature_blocks(df: pd.DataFrame, spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build X, W, Z blocks according to a model specification."""

    x = _block_from_columns(df, spec["x_cols"])
    w = _block_from_columns(df, spec["w_cols"])
    z = _block_from_columns(df, spec["z_cols"])
    if w.shape[1] == 0 and spec.get("w_placeholder_zero", False):
        w = _zero_placeholder_block(len(df))
    if z.shape[1] == 0 and spec.get("z_placeholder_zero", False):
        z = _zero_placeholder_block(len(df))
    return x, w, z


def build_median_profile_frame(subset: pd.DataFrame, *, age_grid: np.ndarray) -> pd.DataFrame:
    """Create the Figure-4-style profile: age varies, other covariates fixed."""

    medians = subset.loc[:, PAPER_14_COVARIATES].median(axis=0)
    profile = pd.DataFrame(
        np.repeat(medians.to_numpy(dtype=float)[None, :], len(age_grid), axis=0),
        columns=PAPER_14_COVARIATES,
    )
    profile["age"] = np.asarray(age_grid, dtype=float)
    return profile


def build_table4_sample_frame(subset: pd.DataFrame, sample_idx: np.ndarray) -> pd.DataFrame:
    """Return the fixed Table-4-style sample used across all models."""

    return subset.iloc[np.asarray(sample_idx, dtype=int)].copy().reset_index(drop=True)


def summarize_predictions(tau_hat: np.ndarray) -> dict[str, float]:
    tau_hat = np.asarray(tau_hat, dtype=float).ravel()
    return {
        "n_subjects": int(tau_hat.shape[0]),
        "cate_mean": float(np.mean(tau_hat)),
        "cate_std": float(np.std(tau_hat)),
        "cate_min": float(np.min(tau_hat)),
        "cate_q25": float(np.quantile(tau_hat, 0.25)),
        "cate_median": float(np.median(tau_hat)),
        "cate_q75": float(np.quantile(tau_hat, 0.75)),
        "cate_max": float(np.max(tau_hat)),
    }


def save_histogram(subset: pd.DataFrame, output_path: Path) -> None:
    """Save a Figure-3-style histogram of observed follow-up time."""

    censored = subset.loc[subset["event"] == 0, "time_obs"].to_numpy(dtype=float)
    uncensored = subset.loc[subset["event"] == 1, "time_obs"].to_numpy(dtype=float)

    plt.figure(figsize=(8.5, 5.5))
    bins = np.linspace(0.0, max(float(subset["time_obs"].max()), PAPER_HORIZON), 30)
    plt.hist(
        [uncensored, censored],
        bins=bins,
        label=["Failure observed", "Censored"],
        color=["#4c78a8", "#f58518"],
        alpha=0.75,
        stacked=False,
    )
    plt.axvline(PAPER_HORIZON, color="crimson", linestyle="--", linewidth=2, label="h = 1000")
    plt.xlabel("Follow-up time (days)")
    plt.ylabel("Count")
    plt.title("ACTG175 ddI vs ZDV+ddI subset")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_age_profile_plot(age_df: pd.DataFrame, display_name: str, output_path: Path) -> None:
    """Save a Figure-4-style age sweep plot with optional bootstrap bands."""

    plt.figure(figsize=(8.5, 5.5))
    x = age_df["age_years"].to_numpy(dtype=float)
    y = age_df["cate_days"].to_numpy(dtype=float)
    plt.plot(x, y, color="#4c78a8", linewidth=2, label=display_name)
    if {"cate_ci_lower", "cate_ci_upper"}.issubset(age_df.columns):
        lo = age_df["cate_ci_lower"].to_numpy(dtype=float)
        hi = age_df["cate_ci_upper"].to_numpy(dtype=float)
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            plt.fill_between(x, lo, hi, color="#4c78a8", alpha=0.18, label="Bootstrap 95% band")
    plt.axhline(0.0, color="black", linewidth=1, linestyle=":")
    plt.xlabel("Age (years)")
    plt.ylabel("Estimated CATE (days)")
    plt.title(f"{display_name}: RMST treatment effect vs age")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_age_profile_comparison_plot(age_df: pd.DataFrame, output_path: Path) -> None:
    """Save a direct comparison plot across all selected models."""

    plt.figure(figsize=(9.0, 5.8))
    for display_name, group in age_df.groupby("display_name", sort=False):
        group = group.sort_values("age_years")
        x = group["age_years"].to_numpy(dtype=float)
        y = group["cate_days"].to_numpy(dtype=float)
        plt.plot(x, y, linewidth=2, label=display_name)
        if {"cate_ci_lower", "cate_ci_upper"}.issubset(group.columns):
            lo = group["cate_ci_lower"].to_numpy(dtype=float)
            hi = group["cate_ci_upper"].to_numpy(dtype=float)
            if np.isfinite(lo).any() and np.isfinite(hi).any():
                plt.fill_between(x, lo, hi, alpha=0.08)
    plt.axhline(0.0, color="black", linewidth=1, linestyle=":")
    plt.xlabel("Age (years)")
    plt.ylabel("Estimated CATE (days)")
    plt.title("ACTG175 comparison: RMST treatment effect vs age")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def format_table4_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert binary coded Table-4 columns into the labels used in the paper."""

    out = df.copy()
    out["hemo"] = np.where(out["hemo"].astype(int) == 1, "Yes", "No")
    out["gender"] = np.where(out["gender"].astype(int) == 1, "Male", "Female")
    out["homo"] = np.where(out["homo"].astype(int) == 1, "Yes", "No")
    out["str2"] = np.where(out["str2"].astype(int) == 1, "Experienced", "Naive")
    return out


def make_model(spec: dict, args: argparse.Namespace):
    """Instantiate a model according to the selected spec."""

    if spec["model_kind"] == "final_conditional":
        return FinalModelConditionalCensoredSurvivalForest(
            target="RMST",
            horizon=PAPER_HORIZON,
            n_estimators=args.num_trees,
            min_samples_leaf=args.min_samples_leaf,
            cv=args.cv,
            random_state=args.random_state,
            q_trees=args.q_trees,
            q_min_samples_leaf=args.q_min_samples_leaf,
            h_n_estimators=args.h_trees,
            h_min_samples_leaf=args.h_min_samples_leaf,
        )

    if spec["model_kind"] == "r_csf":
        return GRFCensoredBaseline(
            target="RMST",
            horizon=PAPER_HORIZON,
            n_estimators=args.grf_num_trees,
            min_samples_leaf=args.grf_min_node_size,
            random_state=args.random_state,
        )

    raise ValueError(f"Unsupported model_kind={spec['model_kind']!r}")


def fit_model(spec: dict, train_df: pd.DataFrame, args: argparse.Namespace):
    """Fit the selected model on one dataset slice."""

    x, w, z = build_feature_blocks(train_df, spec)
    a = train_df["A"].to_numpy(dtype=float)
    time_obs = train_df["time_obs"].to_numpy(dtype=float)
    event = train_df["event"].to_numpy(dtype=float)

    model = make_model(spec, args)
    model.fit_components(x, a, time_obs, event, z, w)
    return model


def predict_effects(spec: dict, model, df: pd.DataFrame) -> np.ndarray:
    """Predict CATEs on a given dataframe using the fitted model."""

    x, w, z = build_feature_blocks(df, spec)
    return np.asarray(model.effect_from_components(x, w, z), dtype=float).ravel()


def compute_dr_scores(spec: dict, model, df: pd.DataFrame, tau_hat: np.ndarray) -> np.ndarray | None:
    """Compute the paper-style doubly robust pseudo-outcome used for BLP."""

    tau_hat = np.asarray(tau_hat, dtype=float).ravel()

    if spec["model_kind"] == "r_csf":
        return np.asarray(model.dr_scores_training(), dtype=float).ravel()

    if spec["model_kind"] != "final_conditional":
        return None

    x, w, z = build_feature_blocks(df, spec)
    a = df["A"].to_numpy(dtype=float).ravel()
    time_obs = df["time_obs"].to_numpy(dtype=float).ravel()
    event = df["event"].to_numpy(dtype=float).ravel()
    y_packed = np.column_stack([time_obs, event])

    w_nuis, z_nuis = model._prepare_nuisance_inputs(w, z)
    nuisance_outputs = model._feature_nuisance.predict(
        y_packed,
        a,
        X=x,
        W=w_nuis,
        Z=z_nuis,
    )
    y_res = nuisance_outputs[0]
    a_res = nuisance_outputs[1]
    bridge = model._feature_nuisance.predict_bridge_outputs(
        X=x,
        W=w_nuis,
        Z=z_nuis,
    )
    q_pred = np.asarray(bridge["q_pred"], dtype=float).ravel()
    a_res = np.asarray(a_res, dtype=float).ravel()
    y_res = np.asarray(y_res, dtype=float).ravel()
    denom = np.maximum(q_pred * (1.0 - q_pred), EPS)
    psi_tau = a_res * (y_res - tau_hat * a_res)
    return tau_hat + psi_tau / denom


def fit_blp_table(gamma: np.ndarray, df: pd.DataFrame, cols: list[str], design_name: str, model_name: str, display_name: str) -> pd.DataFrame:
    """Fit the Table-5-style BLP regression with HC3 robust standard errors."""

    design = df.loc[:, cols].copy()
    design = sm.add_constant(design, has_constant="add")
    fit = sm.OLS(np.asarray(gamma, dtype=float), design).fit(cov_type="HC3")
    ci = fit.conf_int()

    out = pd.DataFrame(
        {
            "model_name": model_name,
            "display_name": display_name,
            "design": design_name,
            "term": [BLP_LABELS.get(term, term) for term in fit.params.index],
            "coef": fit.params.to_numpy(dtype=float),
            "se": fit.bse.to_numpy(dtype=float),
            "tvalue": fit.tvalues.to_numpy(dtype=float),
            "pvalue": fit.pvalues.to_numpy(dtype=float),
            "ci_lower": ci.iloc[:, 0].to_numpy(dtype=float),
            "ci_upper": ci.iloc[:, 1].to_numpy(dtype=float),
        }
    )
    return out


def bootstrap_predictions(
    spec: dict,
    subset: pd.DataFrame,
    age_profile_frame: pd.DataFrame,
    table4_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Bootstrap the age profile and Table-4 subject predictions."""

    if args.bootstrap_reps <= 0:
        return None, None, None, None

    rng = np.random.default_rng(args.bootstrap_seed + abs(hash(spec["model_name"])) % (2**31))
    n = len(subset)
    age_draws = []
    table4_draws = []

    for rep in range(args.bootstrap_reps):
        boot_idx = rng.integers(0, n, size=n)
        boot_df = subset.iloc[boot_idx].reset_index(drop=True)
        model = fit_model(spec, boot_df, args)
        try:
            age_tau = predict_effects(spec, model, age_profile_frame)
            table_tau = predict_effects(spec, model, table4_frame)
        finally:
            _cleanup_model(model)

        age_draws.append(
            pd.DataFrame(
                {
                    "model_name": spec["model_name"],
                    "rep": rep,
                    "age_years": age_profile_frame["age"].to_numpy(dtype=float),
                    "cate_days": age_tau,
                }
            )
        )
        table4_draws.append(
            pd.DataFrame(
                {
                    "model_name": spec["model_name"],
                    "rep": rep,
                    "subject_id": table4_frame["subject_id"].to_numpy(dtype=int),
                    "cate_days": table_tau,
                }
            )
        )

    age_draws_df = pd.concat(age_draws, ignore_index=True)
    table4_draws_df = pd.concat(table4_draws, ignore_index=True)

    alpha = float(args.bootstrap_alpha)
    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0

    age_summary = (
        age_draws_df.groupby(["model_name", "age_years"], as_index=False)["cate_days"]
        .agg(
            cate_bootstrap_se=lambda s: float(np.std(s.to_numpy(dtype=float), ddof=1)) if len(s) > 1 else math.nan,
            cate_ci_lower=lambda s: float(np.quantile(s.to_numpy(dtype=float), lo_q)),
            cate_ci_upper=lambda s: float(np.quantile(s.to_numpy(dtype=float), hi_q)),
        )
    )
    table4_summary = (
        table4_draws_df.groupby(["model_name", "subject_id"], as_index=False)["cate_days"]
        .agg(
            cate_bootstrap_se=lambda s: float(np.std(s.to_numpy(dtype=float), ddof=1)) if len(s) > 1 else math.nan,
            cate_ci_lower=lambda s: float(np.quantile(s.to_numpy(dtype=float), lo_q)),
            cate_ci_upper=lambda s: float(np.quantile(s.to_numpy(dtype=float), hi_q)),
        )
    )
    return age_draws_df, table4_draws_df, age_summary, table4_summary


def run_single_model(
    *,
    subset: pd.DataFrame,
    spec: dict,
    args: argparse.Namespace,
    output_dir: Path,
    age_profile_frame: pd.DataFrame,
    table4_frame: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one model, save all outputs, and return summary tables."""

    model_dir = output_dir / spec["model_name"]
    model_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = fit_model(spec, subset, args)
    fit_time = float(time.time() - t0)

    try:
        tau_hat = predict_effects(spec, model, subset)
        subject_effects = subset.loc[:, ["subject_id", "treatment_label", "time_obs", "event"] + PAPER_14_COVARIATES].copy()
        subject_effects["model_name"] = spec["model_name"]
        subject_effects["display_name"] = spec["display_name"]
        subject_effects["cate_days"] = tau_hat

        dr_scores = compute_dr_scores(spec, model, subset, tau_hat)
        if dr_scores is not None:
            dr_df = subset.loc[:, ["subject_id"] + PAPER_14_COVARIATES].copy()
            dr_df["model_name"] = spec["model_name"]
            dr_df["display_name"] = spec["display_name"]
            dr_df["dr_score"] = np.asarray(dr_scores, dtype=float)
            dr_df.to_csv(model_dir / "dr_scores.csv", index=False)
        else:
            dr_df = pd.DataFrame(columns=["subject_id", "dr_score"])

        age_tau = predict_effects(spec, model, age_profile_frame)
        age_df = pd.DataFrame(
            {
                "model_name": spec["model_name"],
                "display_name": spec["display_name"],
                "age_years": age_profile_frame["age"].to_numpy(dtype=float),
                "cate_days": age_tau,
            }
        )

        table4_tau = predict_effects(spec, model, table4_frame)
        table4_df = table4_frame.loc[:, ["subject_id"] + TABLE4_STYLE_COLUMNS].copy()
        table4_df["model_name"] = spec["model_name"]
        table4_df["display_name"] = spec["display_name"]
        table4_df["cate_days"] = table4_tau
        table4_df = format_table4_covariates(table4_df)

        age_draws_df, table4_draws_df, age_boot_df, table4_boot_df = bootstrap_predictions(
            spec,
            subset,
            age_profile_frame,
            table4_frame,
            args,
        )
    finally:
        _cleanup_model(model)

    if age_boot_df is not None:
        age_df = age_df.merge(age_boot_df, on=["model_name", "age_years"], how="left")
        age_draws_df.to_csv(model_dir / "bootstrap_age_draws.csv", index=False)

    if table4_boot_df is not None:
        table4_df = table4_df.merge(table4_boot_df, on=["model_name", "subject_id"], how="left")
        table4_draws_df.to_csv(model_dir / "bootstrap_table4_draws.csv", index=False)

    subject_effects.to_csv(model_dir / "subject_level_cates.csv", index=False)
    age_df.to_csv(model_dir / "age_profile.csv", index=False)
    table4_df.to_csv(model_dir / "table4_style_random_10.csv", index=False)
    save_age_profile_plot(age_df, spec["display_name"], model_dir / "age_profile.png")

    blp_tables = []
    if dr_scores is not None:
        blp_tables.append(
            fit_blp_table(
                dr_scores,
                subset,
                PAPER_14_COVARIATES,
                "All covariates",
                spec["model_name"],
                spec["display_name"],
            )
        )
        blp_tables.append(
            fit_blp_table(
                dr_scores,
                subset,
                ["age"],
                "Age only",
                spec["model_name"],
                spec["display_name"],
            )
        )
        blp_df = pd.concat(blp_tables, ignore_index=True)
        blp_df.to_csv(model_dir / "table5_style_blp.csv", index=False)
    else:
        blp_df = pd.DataFrame(
            columns=[
                "model_name",
                "display_name",
                "design",
                "term",
                "coef",
                "se",
                "tvalue",
                "pvalue",
                "ci_lower",
                "ci_upper",
            ]
        )

    summary = {
        "model_name": spec["model_name"],
        "display_name": spec["display_name"],
        "model_kind": spec["model_kind"],
        "fit_time_sec": fit_time,
        "x_dim": int(len(spec["x_cols"])),
        "w_dim": int(len(spec["w_cols"])),
        "z_dim": int(len(spec["z_cols"])),
        "implemented_w_dim": int(build_feature_blocks(subset.iloc[:1], spec)[1].shape[1]),
        "implemented_z_dim": int(build_feature_blocks(subset.iloc[:1], spec)[2].shape[1]),
        "bootstrap_reps": int(args.bootstrap_reps),
    }
    summary.update(summarize_predictions(tau_hat))

    age_peak_idx = int(np.argmax(age_df["cate_days"].to_numpy(dtype=float)))
    summary["age_peak_year"] = float(age_df.iloc[age_peak_idx]["age_years"])
    summary["age_peak_cate_days"] = float(age_df.iloc[age_peak_idx]["cate_days"])

    if not blp_df.empty:
        age_only = blp_df.loc[(blp_df["design"] == "Age only") & (blp_df["term"] == "Age"), :]
        if not age_only.empty:
            row = age_only.iloc[0]
            summary["blp_age_only_coef"] = float(row["coef"])
            summary["blp_age_only_se"] = float(row["se"])

    with open(model_dir / "model_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    return summary, age_df, table4_df, blp_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "hiv_actg175_final_conditional",
        help="Directory where runner outputs will be written.",
    )
    parser.add_argument(
        "--cache-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "hiv_actg175_final_conditional" / "actg175_ucimlrepo_cache.csv",
        help="Optional cache path for the fetched UCI dataset.",
    )
    parser.add_argument(
        "--refresh-dataset",
        action="store_true",
        help="Refetch the dataset from ucimlrepo instead of reusing the cached CSV.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_NAMES,
        choices=[spec["model_name"] for spec in MODEL_SPECS],
        help="Which model specs to run.",
    )
    parser.add_argument("--num-trees", type=int, default=200, help="Trees in the Final Conditional forest.")
    parser.add_argument("--q-trees", type=int, default=300, help="Trees in the q learner.")
    parser.add_argument("--h-trees", type=int, default=600, help="Trees in the h learner.")
    parser.add_argument("--min-samples-leaf", type=int, default=20, help="Final forest minimum leaf size.")
    parser.add_argument("--q-min-samples-leaf", type=int, default=20, help="q learner minimum leaf size.")
    parser.add_argument("--h-min-samples-leaf", type=int, default=5, help="h learner minimum leaf size.")
    parser.add_argument("--cv", type=int, default=2, help="Cross-fitting folds for Final Conditional.")
    parser.add_argument("--grf-num-trees", type=int, default=2000, help="Trees in the R-CSF baseline.")
    parser.add_argument("--grf-min-node-size", type=int, default=5, help="Minimum node size for the R-CSF baseline.")
    parser.add_argument("--bootstrap-reps", type=int, default=0, help="Nonparametric bootstrap repetitions for Figure 4 bands and Table 4 SEs.")
    parser.add_argument("--bootstrap-seed", type=int, default=2026, help="Seed for bootstrap resampling.")
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05, help="Bootstrap central interval level, e.g. 0.05 for 95%% bands.")
    parser.add_argument("--random-state", type=int, default=42, help="Model random seed.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for the Table-4-style random 10 sample.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = fetch_actg175_dataframe(cache_csv=args.cache_csv, refresh=args.refresh_dataset)
    subset = prepare_actg175_paper_subset(raw_df)

    age_grid = np.arange(int(subset["age"].min()), int(subset["age"].max()) + 1, dtype=float)
    age_profile_frame = build_median_profile_frame(subset, age_grid=age_grid)
    sample_rng = np.random.default_rng(args.sample_seed)
    sample_idx = np.sort(sample_rng.choice(len(subset), size=min(10, len(subset)), replace=False))
    table4_frame = build_table4_sample_frame(subset, sample_idx)

    cohort_summary = {
        "dataset_name": "AIDS Clinical Trials Group Study 175",
        "dataset_id": DATASET_ID,
        "paper_target": "RMST",
        "paper_horizon_days": PAPER_HORIZON,
        "full_raw_n": int(len(raw_df)),
        "subset_n": int(len(subset)),
        "treated_label": "ZDV+ddI",
        "control_label": "ddI",
        "treated_n": int((subset["A"] == 1).sum()),
        "control_n": int((subset["A"] == 0).sum()),
        "event_rate": float(subset["event"].mean()),
        "median_followup_days": float(subset["time_obs"].median()),
        "paper_14_covariates": PAPER_14_COVARIATES,
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_intervals_method": "Percentile bootstrap on fixed age grid and fixed random 10 subjects." if args.bootstrap_reps > 0 else "disabled",
        "blp_method": "HC3-robust OLS on doubly robust scores, following paper Section 3.2.",
        "sample_seed": int(args.sample_seed),
        "table4_subject_ids": table4_frame["subject_id"].tolist(),
        "notes": [
            "The runner matches the ACTG175 ddI vs ZDV+ddI subset and the RMST@1000-day target used in the paper.",
            "For our Final Conditional X-only replication, W/Z are implemented as constant zero placeholder columns because EconML requires at least one column.",
            "Bootstrap bands and SEs are added as practical inference summaries; they are not the same CI construction as the grf paper's pointwise interval machinery.",
        ],
    }
    with open(output_dir / "cohort_summary.json", "w", encoding="utf-8") as fp:
        json.dump(cohort_summary, fp, indent=2)

    feature_split_df = pd.DataFrame(
        [
            {
                "model_name": spec["model_name"],
                "display_name": spec["display_name"],
                "model_kind": spec["model_kind"],
                "x_cols": ", ".join(spec["x_cols"]),
                "w_cols": ", ".join(spec["w_cols"]),
                "z_cols": ", ".join(spec["z_cols"]),
                "notes": spec["notes"],
            }
            for spec in MODEL_SPECS
        ]
    )
    feature_split_df.to_csv(output_dir / "feature_splits.csv", index=False)

    subset.to_csv(output_dir / "paper_subset_filtered.csv", index=False)
    age_profile_frame.to_csv(output_dir / "age_profile_reference_frame.csv", index=False)
    table4_frame.to_csv(output_dir / "table4_reference_subjects.csv", index=False)
    save_histogram(subset, output_dir / "followup_histogram.png")

    selected_specs = [spec for spec in MODEL_SPECS if spec["model_name"] in set(args.models)]
    summaries = []
    age_tables = []
    table4_tables = []
    blp_tables = []

    for spec in selected_specs:
        print(
            f"[fit] {spec['display_name']} | kind={spec['model_kind']} "
            f"| X={len(spec['x_cols'])} W={len(spec['w_cols'])} Z={len(spec['z_cols'])}"
        )
        summary, age_df, table4_df, blp_df = run_single_model(
            subset=subset,
            spec=spec,
            args=args,
            output_dir=output_dir,
            age_profile_frame=age_profile_frame,
            table4_frame=table4_frame,
        )
        summaries.append(summary)
        age_tables.append(age_df)
        table4_tables.append(table4_df)
        if not blp_df.empty:
            blp_tables.append(blp_df)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "model_summary.csv", index=False)

    age_comparison_df = pd.concat(age_tables, ignore_index=True)
    age_comparison_df.to_csv(output_dir / "age_profile_comparison.csv", index=False)
    save_age_profile_comparison_plot(age_comparison_df, output_dir / "age_profile_comparison.png")

    table4_comparison_df = pd.concat(table4_tables, ignore_index=True)
    table4_comparison_df.to_csv(output_dir / "table4_style_comparison.csv", index=False)

    if blp_tables:
        blp_comparison_df = pd.concat(blp_tables, ignore_index=True)
        blp_comparison_df.to_csv(output_dir / "table5_style_blp_comparison.csv", index=False)

    print(f"[done] wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
