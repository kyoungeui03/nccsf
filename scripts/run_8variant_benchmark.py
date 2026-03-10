#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_fn
from scipy.special import gammainc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.backends import NativeCausalSurvivalForest  # noqa: E402
from grf.core import compute_grf_orthogonal_scores, compute_risk_set_expectations  # noqa: E402
from grf.nuisance import CensoringModel, EventSurvivalModel  # noqa: E402
from grf.nuisance import (  # noqa: E402
    OutcomeProxyModel,
    TreatmentProxyModel,
    compute_ipcw_outcome,
)
from grf.synthetic import SynthConfig, add_ground_truth_cate, generate_synthetic_nc_cox  # noqa: E402
from grf.synthetic.scenarios import standardized_synthetic_scenarios  # noqa: E402


def _default_mtry(num_features: int) -> int:
    return int(min(np.ceil(np.sqrt(num_features) + 20), num_features))


def _weibull_scale(lam: float, k: float, eta: np.ndarray) -> np.ndarray:
    return lam * np.exp(-eta / k)


def mean_survival_given_eta(eta: np.ndarray, cfg: SynthConfig) -> np.ndarray:
    scale = _weibull_scale(cfg.lam_t, cfg.k_t, np.asarray(eta, dtype=float))
    return scale * math.gamma(1.0 + 1.0 / cfg.k_t)


def recover_dgp_internals(cfg: SynthConfig) -> dict[str, np.ndarray | float | None]:
    rng = np.random.default_rng(cfg.seed)
    n, p = cfg.n, cfg.p_x

    X = rng.normal(size=(n, p))
    U = rng.normal(size=n)
    b_z = rng.normal(scale=0.3, size=p)
    b_w = rng.normal(scale=0.3, size=p)
    _eps_z = rng.normal(scale=cfg.sigma_z, size=n)
    _eps_w = rng.normal(scale=cfg.sigma_w, size=n)

    alpha = rng.normal(scale=0.5, size=p)
    alpha_nonlinear = None
    if cfg.linear_treatment:
        linpred = X @ alpha + cfg.gamma_u_in_a * U
    else:
        x_nonlinear = np.column_stack(
            [1.0 / (1.0 + np.exp(-X[:, i])) for i in range(min(3, p))]
            + [X[:, i] * X[:, (i + 1) % p] for i in range(min(2, p))]
        )
        alpha_nonlinear = rng.normal(scale=0.3, size=x_nonlinear.shape[1])
        linpred = x_nonlinear @ alpha_nonlinear + cfg.gamma_u_in_a * (1.0 / (1.0 + np.exp(-U)))

    from grf.synthetic import calibrate_intercept_for_prevalence, sigmoid

    b0 = calibrate_intercept_for_prevalence(linpred, cfg.a_prevalence)
    p_a = sigmoid(b0 + linpred)
    A = rng.binomial(1, p_a, size=n).astype(int)

    beta_t = rng.normal(scale=0.4, size=p)
    u_t = rng.random(n)
    beta_squared = None
    beta_interact = None
    if cfg.linear_outcome:
        eta_t0 = X @ beta_t + cfg.beta_u_in_t * U
        eta_t1 = eta_t0 + cfg.tau_log_hr
    else:
        x_squared = X[:, : min(2, p)] ** 2
        x_interact = X[:, 0:1] * U.reshape(-1, 1)
        beta_squared = rng.normal(scale=0.2, size=x_squared.shape[1])
        beta_interact = rng.normal(scale=0.2, size=x_interact.shape[1])
        from grf.synthetic import sigmoid

        nonlinear_part = x_squared @ beta_squared + x_interact @ beta_interact + 0.5 * sigmoid(U)
        eta_t0 = X @ beta_t + cfg.beta_u_in_t * U + nonlinear_part
        eta_t1 = eta_t0 + cfg.tau_log_hr

    from grf.synthetic import weibull_ph_time_paper

    T0 = weibull_ph_time_paper(u_t, k=cfg.k_t, lam=cfg.lam_t, eta=eta_t0)
    T1 = weibull_ph_time_paper(u_t, k=cfg.k_t, lam=cfg.lam_t, eta=eta_t1)

    beta_c = rng.normal(scale=0.3, size=p)
    u_c = rng.random(n)
    eta_c = X @ beta_c + cfg.beta_u_in_c * U
    t_obs = np.where(A == 1, T1, T0)

    lam_c = cfg.lam_c
    if lam_c is None:
        lo, hi = float(cfg.censor_lam_lo), float(cfg.censor_lam_hi)
        for _ in range(cfg.max_censor_calib_iter):
            mid = 0.5 * (lo + hi)
            C_mid = weibull_ph_time_paper(u_c, k=cfg.k_c, lam=mid, eta=eta_c)
            if (C_mid < t_obs).mean() < cfg.target_censor_rate:
                hi = mid
            else:
                lo = mid
        lam_c = 0.5 * (lo + hi)

    return {
        "X": X,
        "U": U,
        "A": A,
        "alpha": alpha,
        "alpha_nonlinear": alpha_nonlinear,
        "b0": b0,
        "beta_t": beta_t,
        "beta_squared": beta_squared,
        "beta_interact": beta_interact,
        "beta_c": beta_c,
        "b_z": b_z,
        "b_w": b_w,
        "lam_c": lam_c,
    }


def _treatment_logit_score(
    X: np.ndarray,
    U: np.ndarray,
    cfg: SynthConfig,
    dgp: dict[str, np.ndarray | float | None],
) -> np.ndarray:
    from grf.synthetic import sigmoid

    U = np.asarray(U, dtype=float)
    base_linear = X @ np.asarray(dgp["alpha"], dtype=float)
    if cfg.linear_treatment:
        if U.ndim == 1:
            return base_linear + cfg.gamma_u_in_a * U
        return base_linear[:, np.newaxis] + cfg.gamma_u_in_a * U

    x_nonlinear = np.column_stack(
        [sigmoid(X[:, i]) for i in range(min(3, cfg.p_x))]
        + [X[:, i] * X[:, (i + 1) % cfg.p_x] for i in range(min(2, cfg.p_x))]
    )
    alpha_nonlinear = np.asarray(dgp["alpha_nonlinear"], dtype=float)
    base = x_nonlinear @ alpha_nonlinear
    if U.ndim == 1:
        return base + cfg.gamma_u_in_a * sigmoid(U)
    return base[:, np.newaxis] + cfg.gamma_u_in_a * sigmoid(U)


def _event_eta(
    X: np.ndarray,
    U: np.ndarray,
    A: np.ndarray | float,
    cfg: SynthConfig,
    dgp: dict[str, np.ndarray | float | None],
) -> np.ndarray:
    from grf.synthetic import sigmoid

    U = np.asarray(U, dtype=float)
    xb = X @ np.asarray(dgp["beta_t"], dtype=float)
    if U.ndim == 1:
        eta = xb + cfg.beta_u_in_t * U
        if not cfg.linear_outcome:
            x_squared = X[:, : min(2, cfg.p_x)] ** 2
            beta_squared = np.asarray(dgp["beta_squared"], dtype=float)
            beta_interact = np.asarray(dgp["beta_interact"], dtype=float)
            eta = eta + x_squared @ beta_squared + X[:, 0] * U * beta_interact[0] + 0.5 * sigmoid(U)
        return eta + cfg.tau_log_hr * np.asarray(A, dtype=float)

    eta = xb[:, np.newaxis] + cfg.beta_u_in_t * U
    if not cfg.linear_outcome:
        x_squared = X[:, : min(2, cfg.p_x)] ** 2
        beta_squared = np.asarray(dgp["beta_squared"], dtype=float)
        beta_interact = np.asarray(dgp["beta_interact"], dtype=float)
        eta = (
            eta
            + (x_squared @ beta_squared)[:, np.newaxis]
            + X[:, [0]] * U * beta_interact[0]
            + 0.5 * sigmoid(U)
        )
    return eta + cfg.tau_log_hr * np.asarray(A, dtype=float)[:, np.newaxis]


def _posterior_u_given_proxy(
    proxy: np.ndarray,
    X: np.ndarray,
    beta_proxy: np.ndarray,
    proxy_loading: float,
    noise_scale: float,
) -> tuple[np.ndarray, float]:
    proxy = np.asarray(proxy, dtype=float).reshape(-1)
    sigma2 = noise_scale**2
    a2 = proxy_loading**2
    posterior_var = sigma2 / (sigma2 + a2)
    posterior_mean = proxy_loading * (proxy - X @ beta_proxy) / (sigma2 + a2)
    return posterior_mean, posterior_var


def _gauss_hermite_expectation(
    posterior_mean: np.ndarray,
    posterior_var: float,
    fn,
    quadrature_nodes: int = 30,
) -> np.ndarray:
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_nodes)
    u_draws = posterior_mean[:, np.newaxis] + np.sqrt(2.0 * posterior_var) * nodes[np.newaxis, :]
    return np.sum(fn(u_draws) * weights[np.newaxis, :], axis=1) / np.sqrt(np.pi)


def true_censoring_on_grid(
    X: np.ndarray,
    U: np.ndarray,
    Y: np.ndarray,
    time_grid: np.ndarray,
    cfg: SynthConfig,
    beta_c: np.ndarray,
    lam_c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta_c = X @ beta_c + cfg.beta_u_in_c * U
    scale_c = _weibull_scale(lam_c, cfg.k_c, eta_c)
    t_over_s = time_grid[np.newaxis, :] / scale_c[:, np.newaxis]
    cum_haz = t_over_s ** cfg.k_c
    surv_c = np.exp(-cum_haz)

    hazard_jumps = np.zeros_like(cum_haz)
    hazard_jumps[:, 0] = cum_haz[:, 0]
    hazard_jumps[:, 1:] = np.diff(cum_haz, axis=1)
    hazard_jumps = np.clip(hazard_jumps, 0.0, None)

    s_c_y = np.exp(-(Y / scale_c) ** cfg.k_c)
    return surv_c, hazard_jumps, s_c_y


def true_event_survival_on_grid(
    X: np.ndarray,
    U: np.ndarray,
    A: np.ndarray,
    time_grid: np.ndarray,
    cfg: SynthConfig,
    dgp: dict[str, np.ndarray | float | None],
) -> np.ndarray:
    eta_t = _event_eta(X, U, A, cfg, dgp)
    scale_t = _weibull_scale(cfg.lam_t, cfg.k_t, eta_t)
    return np.exp(-((time_grid[np.newaxis, :] / scale_t[:, np.newaxis]) ** cfg.k_t))


def true_propensity_oracle(
    X: np.ndarray,
    U: np.ndarray,
    dgp: dict[str, np.ndarray | float | None],
    cfg: SynthConfig,
) -> np.ndarray:
    from grf.synthetic import sigmoid

    return sigmoid(float(dgp["b0"]) + _treatment_logit_score(X, U, cfg, dgp))


def true_propensity_nc(
    Z: np.ndarray,
    X: np.ndarray,
    dgp: dict[str, np.ndarray | float | None],
    cfg: SynthConfig,
) -> np.ndarray:
    from grf.synthetic import sigmoid

    posterior_mean, posterior_var = _posterior_u_given_proxy(
        Z,
        X,
        np.asarray(dgp["b_z"], dtype=float),
        cfg.aZ,
        cfg.sigma_z,
    )
    return _gauss_hermite_expectation(
        posterior_mean,
        posterior_var,
        lambda u_draws: sigmoid(float(dgp["b0"]) + _treatment_logit_score(X, u_draws, cfg, dgp)),
    )


def true_outcome_oracle(
    X: np.ndarray,
    U: np.ndarray,
    cfg: SynthConfig,
    dgp: dict[str, np.ndarray | float | None],
) -> tuple[np.ndarray, np.ndarray]:
    h0 = mean_survival_given_eta(_event_eta(X, U, np.zeros(X.shape[0]), cfg, dgp), cfg)
    h1 = mean_survival_given_eta(_event_eta(X, U, np.ones(X.shape[0]), cfg, dgp), cfg)
    return h0, h1


def true_outcome_nc(
    W: np.ndarray,
    X: np.ndarray,
    cfg: SynthConfig,
    dgp: dict[str, np.ndarray | float | None],
) -> tuple[np.ndarray, np.ndarray]:
    posterior_mean, posterior_var = _posterior_u_given_proxy(
        W,
        X,
        np.asarray(dgp["b_w"], dtype=float),
        cfg.aW,
        cfg.sigma_w,
    )
    h0 = _gauss_hermite_expectation(
        posterior_mean,
        posterior_var,
        lambda u_draws: mean_survival_given_eta(_event_eta(X, u_draws, np.zeros(X.shape[0]), cfg, dgp), cfg),
    )
    h1 = _gauss_hermite_expectation(
        posterior_mean,
        posterior_var,
        lambda u_draws: mean_survival_given_eta(_event_eta(X, u_draws, np.ones(X.shape[0]), cfg, dgp), cfg),
    )
    return h0, h1


@dataclass
class PseudoResponses:
    gamma: np.ndarray
    h: np.ndarray
    x_forest: np.ndarray
    q_hat: np.ndarray
    delta: np.ndarray


def build_pseudo_responses(
    X: np.ndarray,
    U: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    delta: np.ndarray,
    cfg: SynthConfig,
    dgp: dict[str, np.ndarray | float | None],
    *,
    mode: str,
    true_surv: bool,
    true_qr: bool,
    n_jobs: int,
) -> PseudoResponses:
    Y_star = Y.astype(float).copy()
    delta_star = delta.astype(float).copy()
    n_samples = X.shape[0]

    if mode == "oracle":
        x_forest = np.hstack([X, U.reshape(-1, 1)])
        v_surv = np.hstack([X, U.reshape(-1, 1), A.reshape(-1, 1)])
    elif mode == "nc":
        x_forest = X
        v_surv = np.hstack([X, A.reshape(-1, 1), W, Z])
    elif mode == "naive":
        x_forest = X
        v_surv = np.hstack([X, A.reshape(-1, 1)])
    elif mode == "augmented":
        x_forest = np.hstack([X, W, Z])
        v_surv = np.hstack([X, W, Z, A.reshape(-1, 1)])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    time_grid = np.sort(np.unique(Y_star))

    if true_surv:
        surv_c, hazard_c, s_c_y = true_censoring_on_grid(
            X,
            U,
            Y_star,
            time_grid,
            cfg,
            np.asarray(dgp["beta_c"], dtype=float),
            float(dgp["lam_c"]),
        )
        fail_times_c = time_grid
        surv_c = np.maximum(surv_c, 0.025)
        s_c_y = np.maximum(s_c_y, 0.025)
    else:
        censoring_model = CensoringModel(n_jobs=n_jobs)
        censoring_model.fit(v_surv, Y_star, delta_star)
        surv_c, hazard_c, fail_times_c = censoring_model.predict_surv_and_hazard(v_surv)
        s_c_y = np.array(
            [
                surv_c[index, max(0, np.searchsorted(fail_times_c, Y_star[index], side="right") - 1)]
                for index in range(n_samples)
            ]
        )

    if true_surv:
        surv_e = true_event_survival_on_grid(X, U, A, time_grid, cfg, dgp)
        fail_times_e = time_grid
    else:
        event_model = EventSurvivalModel(n_jobs=n_jobs)
        event_model.fit(v_surv, Y_star, delta_star)
        surv_e, fail_times_e = event_model.predict_survival(v_surv)

    e_t_given_gt_t = compute_risk_set_expectations(surv_e, fail_times_e, np.inf)

    if true_qr:
        from grf.synthetic import sigmoid

        if mode == "oracle":
            q_hat = true_propensity_oracle(X, U, dgp, cfg)
        elif mode == "nc":
            q_hat = true_propensity_nc(Z, X, dgp, cfg)
        else:
            g = cfg.gamma_u_in_a
            q_hat = sigmoid((float(dgp["b0"]) + X @ np.asarray(dgp["alpha"], dtype=float)) / np.sqrt(1.0 + np.pi / 8.0 * g**2))
    else:
        q_model = TreatmentProxyModel()
        if mode == "oracle":
            q_model.fit(None, x_forest, A)
            q_hat = q_model.predict_proba(None, x_forest)
        elif mode == "nc":
            q_model.fit(Z, X, A)
            q_hat = q_model.predict_proba(Z, X)
        else:
            q_model.fit(None, x_forest, A)
            q_hat = q_model.predict_proba(None, x_forest)

    s_c_y_safe = np.maximum(s_c_y, 1e-10)
    y_ipcw = compute_ipcw_outcome(Y_star, delta_star, s_c_y_safe)

    if true_qr:
        if mode == "oracle":
            h0_hat, h1_hat = true_outcome_oracle(X, U, cfg, dgp)
        elif mode == "nc":
            h0_hat, h1_hat = true_outcome_nc(W, X, cfg, dgp)
        else:
            if cfg.linear_outcome:
                G = math.gamma(1.0 + 1.0 / cfg.k_t)
                c = -cfg.beta_u_in_t / cfg.k_t
                mgf = np.exp(0.5 * c**2)
                xb = X @ np.asarray(dgp["beta_t"], dtype=float)
                h0_hat = cfg.lam_t * G * np.exp(-xb / cfg.k_t) * mgf
                h1_hat = cfg.lam_t * G * np.exp(-(xb + cfg.tau_log_hr) / cfg.k_t) * mgf
            else:
                h0_hat = _gauss_hermite_expectation(
                    np.zeros(n_samples, dtype=float),
                    1.0,
                    lambda u_draws: mean_survival_given_eta(
                        _event_eta(X, u_draws, np.zeros(X.shape[0]), cfg, dgp),
                        cfg,
                    ),
                )
                h1_hat = _gauss_hermite_expectation(
                    np.zeros(n_samples, dtype=float),
                    1.0,
                    lambda u_draws: mean_survival_given_eta(
                        _event_eta(X, u_draws, np.ones(X.shape[0]), cfg, dgp),
                        cfg,
                    ),
                )
    else:
        h_model = OutcomeProxyModel()
        if mode == "oracle":
            h_model.fit(None, x_forest, A, y_ipcw)
            h0_hat, h1_hat = h_model.predict(None, x_forest)
        elif mode == "nc":
            h_model.fit(W, X, A, y_ipcw)
            h0_hat, h1_hat = h_model.predict(W, X)
        else:
            h_model.fit(None, x_forest, A, y_ipcw)
            h0_hat, h1_hat = h_model.predict(None, x_forest)

    m_hat = q_hat * h1_hat + (1.0 - q_hat) * h0_hat
    treatment_residual = A - q_hat

    k_gamma = np.zeros_like(surv_c)
    k_h = np.zeros_like(surv_c)
    for index in range(n_samples):
        event_interp = np.interp(fail_times_c, fail_times_e, e_t_given_gt_t[index, :])
        k_gamma[index, :] = treatment_residual[index] * (event_interp - m_hat[index])
        k_h[index, :] = treatment_residual[index] ** 2

    gamma, h = compute_grf_orthogonal_scores(
        Y_star,
        A,
        treatment_residual,
        m_hat,
        delta_star,
        surv_c,
        hazard_c,
        k_gamma,
        k_h,
        fail_times_c,
    )
    return PseudoResponses(gamma=gamma, h=h, x_forest=x_forest, q_hat=q_hat, delta=delta_star)


def fit_local_grf_variant(
    x_forest: np.ndarray,
    A: np.ndarray,
    q_hat: np.ndarray,
    gamma: np.ndarray,
    h: np.ndarray,
    delta: np.ndarray,
    *,
    num_trees: int,
    num_threads: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    model = NativeCausalSurvivalForest.fit(
        x_forest,
        A - q_hat,
        gamma,
        np.maximum(h, 1e-10),
        delta,
        num_trees=num_trees,
        sample_fraction=0.5,
        mtry=_default_mtry(x_forest.shape[1]),
        min_node_size=5,
        honesty=True,
        honesty_fraction=0.5,
        honesty_prune_leaves=True,
        alpha=0.05,
        imbalance_penalty=0.0,
        stabilize_splits=True,
        ci_group_size=2,
        num_threads=num_threads,
        seed=seed,
    )
    try:
        predictions = model.predict(x_forest, estimate_variance=False)
    finally:
        model.close()
    return predictions, time.perf_counter() - start


def evaluate_predictions(
    variant_key: str,
    variant_label: str,
    predictions: np.ndarray,
    truth: np.ndarray,
    nuis_time: float,
    forest_time: float,
) -> dict[str, float | str]:
    return {
        "variant": variant_key,
        "variant_label": variant_label,
        "mean_prediction": float(np.mean(predictions)),
        "mean_true_cate": float(np.mean(truth)),
        "bias": float(np.mean(predictions - truth)),
        "rmse": float(np.sqrt(np.mean((predictions - truth) ** 2))),
        "mae": float(np.mean(np.abs(predictions - truth))),
        "corr": float(np.corrcoef(predictions, truth)[0, 1]),
        "nuis_time_sec": float(nuis_time),
        "forest_time_sec": float(forest_time),
        "total_time_sec": float(nuis_time + forest_time),
        "std_prediction": float(np.std(predictions)),
    }


def _confounding_label(gamma_u_in_a: float, beta_u_in_t: float, beta_u_in_c: float) -> str:
    level = max(abs(gamma_u_in_a), abs(beta_u_in_t), abs(beta_u_in_c))
    if level < 0.35:
        return "weak"
    if level < 0.8:
        return "moderate"
    return "strong"


def _effect_size_label(tau_log_hr: float) -> str:
    magnitude = abs(tau_log_hr)
    if magnitude <= 0.15:
        return "near-null"
    if magnitude <= 0.35:
        return "small"
    if magnitude <= 0.6:
        return "moderate"
    return "large"


def _proxy_label(aZ: float, sigma_z: float, aW: float, sigma_w: float) -> str:
    mean_corr = 0.5 * (abs(aZ) / np.sqrt(aZ**2 + sigma_z**2) + abs(aW) / np.sqrt(aW**2 + sigma_w**2))
    return "informative proxies" if mean_corr >= 0.65 else "non informative proxies"


def build_title(cfg: SynthConfig, censor_rate: float) -> tuple[str, str]:
    treatment_shape = "Linear" if cfg.linear_treatment else "Nonlinear"
    outcome_shape = "linear" if cfg.linear_outcome else "nonlinear"
    proxy_quality = _proxy_label(cfg.aZ, cfg.sigma_z, cfg.aW, cfg.sigma_w)
    confounding = _confounding_label(cfg.gamma_u_in_a, cfg.beta_u_in_t, cfg.beta_u_in_c)
    direction = "beneficial" if cfg.tau_log_hr < 0 else "harmful"
    effect_size = _effect_size_label(cfg.tau_log_hr)
    headline = (
        f"{treatment_shape} treatment / {outcome_shape} outcome DGP, {proxy_quality}, {confounding} confounding,\n"
        f"{effect_size} {direction} treatment effect, n={cfg.n}, p={cfg.p_x}, seed={cfg.seed}, censoring rate={int(round(100 * censor_rate))}%"
    )
    return headline, "8-variant benchmark"


def render_summary_table(metrics_df: pd.DataFrame, output_path: Path, title: str, subtitle: str) -> None:
    display = pd.DataFrame(
        {
            "Variant": metrics_df["variant_label"],
            "Pred CATE": metrics_df["mean_prediction"].map(lambda value: f"{value:.4f}"),
            "True CATE": metrics_df["mean_true_cate"].map(lambda value: f"{value:.4f}"),
            "Bias": metrics_df["bias"].map(lambda value: f"{value:.4f}"),
            "RMSE": metrics_df["rmse"].map(lambda value: f"{value:.4f}"),
            "MAE": metrics_df["mae"].map(lambda value: f"{value:.4f}"),
            "Corr": metrics_df["corr"].map(lambda value: f"{value:.4f}"),
            "Time": metrics_df["total_time_sec"].map(lambda value: f"{value:.1f}s"),
        }
    )

    fig, ax = plt.subplots(figsize=(18, 7.9), dpi=200)
    fig.patch.set_facecolor("#efefef")
    ax.axis("off")
    fig.suptitle(title, fontsize=18, fontweight="semibold", y=0.975)
    fig.text(0.5, 0.855, subtitle, ha="center", va="center", fontsize=12, color="#56606e")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        colLoc="center",
        cellLoc="center",
        colWidths=[0.41, 0.095, 0.095, 0.09, 0.08, 0.08, 0.08, 0.07],
        bbox=[0.01, 0.03, 0.98, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c7cbd3")
        if row == 0:
            cell.set_facecolor("#1c2a3f")
            cell.set_text_props(color="white", weight="bold")
            cell.set_linewidth(1.4)
            cell.set_height(0.095)
        else:
            cell.set_facecolor("#f7f7f8" if row % 2 == 1 else "#ececef")
            cell.set_height(0.088)
            if col == 0:
                cell.set_text_props(ha="left")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full 8-variant NC-CSF benchmark.")
    parser.add_argument("--scenario-slug", default="survival_default", help="Standardized synthetic scenario slug.")
    parser.add_argument("--num-trees", type=int, default=200, help="Number of trees in the local GRF backend.")
    parser.add_argument("--num-threads", type=int, default=1, help="Thread count for native and nuisance models.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark_8variant"), help="Directory for benchmark artifacts.")
    return parser.parse_args()


def _load_standardized_scenario(slug: str) -> SynthConfig:
    for scenario in standardized_synthetic_scenarios():
        if scenario.slug == slug:
            if scenario.family != "survival":
                raise ValueError(f"Scenario {slug} is not a survival-family scenario.")
            return SynthConfig(**scenario.config)
    raise ValueError(f"Unknown scenario slug: {slug}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_standardized_scenario(args.scenario_slug)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    observed_df, truth_df, params = generate_synthetic_nc_cox(cfg)
    observed_df, truth_df = add_ground_truth_cate(observed_df, truth_df, cfg, params)
    dgp = recover_dgp_internals(cfg)

    x_cols = [f"X{index}" for index in range(cfg.p_x)]
    X = observed_df[x_cols].to_numpy(dtype=float)
    W = observed_df["W"].to_numpy(dtype=float).reshape(-1, 1)
    Z = observed_df["Z"].to_numpy(dtype=float).reshape(-1, 1)
    A = observed_df["A"].to_numpy(dtype=float)
    Y = observed_df["time"].to_numpy(dtype=float)
    delta = observed_df["event"].to_numpy(dtype=float)
    U = truth_df["U"].to_numpy(dtype=float)
    eta_t0 = truth_df["eta_t0"].to_numpy(dtype=float)
    eta_t1 = truth_df["eta_t1"].to_numpy(dtype=float)
    true_cate = mean_survival_given_eta(eta_t1, cfg) - mean_survival_given_eta(eta_t0, cfg)

    variants = [
        ("A1", "A1  Oracle (all true)", dict(mode="oracle", true_surv=True, true_qr=True)),
        ("A2", "A2  Oracle (true surv, est q/r)", dict(mode="oracle", true_surv=True, true_qr=False)),
        ("A3", "A3  Oracle (all estimated)", dict(mode="oracle", true_surv=False, true_qr=False)),
        ("B1", "B1  R-CSF baseline (X only)", dict(mode="naive", true_surv=False, true_qr=False)),
        ("B2", "B2  R-CSF baseline (X+W+Z)", dict(mode="augmented", true_surv=False, true_qr=False)),
        ("C1", "C1  NC-CSF (all true)", dict(mode="nc", true_surv=True, true_qr=True)),
        ("C2", "C2  NC-CSF (true surv, est q/r)", dict(mode="nc", true_surv=True, true_qr=False)),
        ("C3", "C3  NC-CSF (all estimated)", dict(mode="nc", true_surv=False, true_qr=False)),
    ]

    metrics: list[dict[str, float | str]] = []
    prediction_columns: dict[str, np.ndarray] = {"true_cate": true_cate}

    for variant_key, variant_label, settings in variants:
        print(f"Running {variant_label} ...")
        start_nuis = time.perf_counter()
        pseudo = build_pseudo_responses(
            X,
            U,
            W,
            Z,
            A,
            Y,
            delta,
            cfg,
            dgp,
            n_jobs=args.num_threads,
            **settings,
        )
        nuis_time = time.perf_counter() - start_nuis

        predictions, forest_time = fit_local_grf_variant(
            pseudo.x_forest,
            A,
            pseudo.q_hat,
            pseudo.gamma,
            pseudo.h,
            pseudo.delta,
            num_trees=args.num_trees,
            num_threads=args.num_threads,
            seed=cfg.seed,
        )
        metrics.append(evaluate_predictions(variant_key, variant_label, predictions, true_cate, nuis_time, forest_time))
        prediction_columns[variant_key] = predictions

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.DataFrame(prediction_columns)

    observed_csv = output_dir / "benchmark_8variant_observed.csv"
    truth_csv = output_dir / "benchmark_8variant_truth.csv"
    metrics_csv = output_dir / "benchmark_8variant_metrics.csv"
    metrics_json = output_dir / "benchmark_8variant_metrics.json"
    predictions_csv = output_dir / "benchmark_8variant_predictions.csv"
    config_json = output_dir / "benchmark_8variant_config.json"
    table_png = output_dir / "benchmark_8variant_table.png"
    root_png = PROJECT_ROOT / "benchmark_8variant_table.png"

    observed_df.to_csv(observed_csv, index=False)
    truth_df.assign(true_mean_cate=true_cate).to_csv(truth_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config_json.write_text(
        json.dumps(
            {
                "scenario_slug": args.scenario_slug,
                "num_trees": args.num_trees,
                "num_threads": args.num_threads,
                "config": cfg.__dict__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    title, subtitle = build_title(cfg, cfg.target_censor_rate)
    render_summary_table(metrics_df, table_png, title, subtitle)
    shutil.copyfile(table_png, root_png)

    print(metrics_df[["variant_label", "mean_prediction", "mean_true_cate", "bias", "rmse", "mae", "corr", "total_time_sec"]].to_string(index=False))
    print(f"\nSaved metrics to {metrics_csv}")
    print(f"Saved predictions to {predictions_csv}")
    print(f"Saved PNG table to {table_png}")
    print(f"Copied PNG table to {root_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
