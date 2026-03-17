from __future__ import annotations

import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from econml._ortho_learner import _OrthoLearner
from econml.utilities import filter_none_kwargs
from sklearn.base import clone

from grf.methods import EconmlMildShrinkNCSurvivalForest
from grf.methods.econml_mild_shrink import (
    _clip_quantile,
    _compute_ipcw_3term_y_res,
    _compute_ipcw_pseudo_outcome,
    _compute_survival_probability_q_from_s,
    _compute_target_ipcw_3term_y_res,
    _compute_target_pseudo_outcome,
    _compute_q_from_s,
    _prepare_target_inputs,
    _ensure_2d,
    _fit_event_cox,
    _fit_kaplan_meier_censoring,
    _predict_s_on_grid,
)
from grf.synthetic import (
    SynthConfig,
    add_ground_truth_cate,
    calibrate_intercept_for_prevalence,
    generate_synthetic_nc_cox,
    sigmoid,
    weibull_ph_time_paper,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
R_CSF_SCRIPT = PROJECT_ROOT / "scripts" / "run_grf_csf_baseline.R"
TITLE_SUFFIX = "n=2000, p=5, seed=42, censoring rate=35%"

os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_CONFIG = dict(
    n=2000,
    p_x=5,
    seed=42,
    a_prevalence=0.5,
    k_t=1.5,
    lam_t=0.4,
    k_c=1.2,
    lam_c=None,
    beta_u_in_c=0.3,
    target_censor_rate=0.35,
    max_censor_calib_iter=60,
    censor_lam_lo=1e-8,
    censor_lam_hi=1e6,
    admin_censor_time=None,
    aZ=1.5,
    aW=1.5,
)

CASE_SPECS = [
    {
        "case_id": 1,
        "slug": "linear_linear_informative_strong_beneficial_large",
        "title": f"Linear treatment / linear outcome DGP, informative proxies, strong confounding, large beneficial treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.9, beta_u_in_t=1.1, tau_log_hr=-0.7, sigma_z=1.125, sigma_w=1.53, linear_treatment=True, linear_outcome=True),
    },
    {
        "case_id": 2,
        "slug": "linear_linear_informative_weak_harmful_small",
        "title": f"Linear treatment / linear outcome DGP, informative proxies, weak confounding, small harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.2, beta_u_in_t=0.2, tau_log_hr=0.25, sigma_z=1.125, sigma_w=1.53, linear_treatment=True, linear_outcome=True),
    },
    {
        "case_id": 3,
        "slug": "linear_linear_weakproxy_strong_harmful_moderate",
        "title": f"Linear treatment / linear outcome DGP, weak proxies, strong confounding, moderate harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.9, beta_u_in_t=1.1, tau_log_hr=0.5, sigma_z=7.35, sigma_w=4.77, linear_treatment=True, linear_outcome=True),
    },
    {
        "case_id": 4,
        "slug": "linear_linear_weakproxy_weak_nearnull_harmful",
        "title": f"Linear treatment / linear outcome DGP, weak proxies, weak confounding, near-null harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.25, beta_u_in_t=0.2, tau_log_hr=0.12, sigma_z=7.35, sigma_w=4.77, linear_treatment=True, linear_outcome=True),
    },
    {
        "case_id": 5,
        "slug": "linear_nonlinear_informative_strong_beneficial_large",
        "title": f"Linear treatment / nonlinear outcome DGP, informative proxies, strong confounding, large beneficial treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.9, beta_u_in_t=1.1, tau_log_hr=-0.7, sigma_z=1.125, sigma_w=1.53, linear_treatment=True, linear_outcome=False),
    },
    {
        "case_id": 6,
        "slug": "linear_nonlinear_informative_weak_harmful_small",
        "title": f"Linear treatment / nonlinear outcome DGP, informative proxies, weak confounding, small harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.2, beta_u_in_t=0.2, tau_log_hr=0.3, sigma_z=1.125, sigma_w=1.53, linear_treatment=True, linear_outcome=False),
    },
    {
        "case_id": 7,
        "slug": "linear_nonlinear_weakproxy_strong_harmful_moderate",
        "title": f"Linear treatment / nonlinear outcome DGP, weak proxies, strong confounding, moderate harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.9, beta_u_in_t=1.1, tau_log_hr=0.5, sigma_z=7.35, sigma_w=4.77, linear_treatment=True, linear_outcome=False),
    },
    {
        "case_id": 8,
        "slug": "linear_nonlinear_weakproxy_weak_nearnull_harmful",
        "title": f"Linear treatment / nonlinear outcome DGP, weak proxies, weak confounding, near-null harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.25, beta_u_in_t=0.2, tau_log_hr=0.12, sigma_z=7.35, sigma_w=4.77, linear_treatment=True, linear_outcome=False),
    },
    {
        "case_id": 9,
        "slug": "nonlinear_nonlinear_informative_strong_beneficial_large",
        "title": f"Nonlinear treatment / nonlinear outcome DGP, informative proxies, strong confounding, large beneficial treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.9, beta_u_in_t=1.1, tau_log_hr=-0.7, sigma_z=1.125, sigma_w=1.53, linear_treatment=False, linear_outcome=False),
    },
    {
        "case_id": 10,
        "slug": "nonlinear_nonlinear_informative_weak_harmful_small",
        "title": f"Nonlinear treatment / nonlinear outcome DGP, informative proxies, weak confounding, small harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.2, beta_u_in_t=0.2, tau_log_hr=0.3, sigma_z=1.125, sigma_w=1.53, linear_treatment=False, linear_outcome=False),
    },
    {
        "case_id": 11,
        "slug": "nonlinear_nonlinear_weakproxy_strong_harmful_moderate",
        "title": f"Nonlinear treatment / nonlinear outcome DGP, weak proxies, strong confounding, moderate harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.9, beta_u_in_t=1.1, tau_log_hr=0.5, sigma_z=7.35, sigma_w=4.77, linear_treatment=False, linear_outcome=False),
    },
    {
        "case_id": 12,
        "slug": "nonlinear_nonlinear_weakproxy_weak_nearnull_harmful",
        "title": f"Nonlinear treatment / nonlinear outcome DGP, weak proxies, weak confounding, near-null harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.25, beta_u_in_t=0.2, tau_log_hr=0.12, sigma_z=7.35, sigma_w=4.77, linear_treatment=False, linear_outcome=False),
    },
]

EIGHT_VARIANT_SPECS = [
    ("A1  Oracle (all true)", "oracle", dict(true_surv=True, true_qh=True)),
    ("A2  Oracle (true surv, est q/h)", "oracle", dict(true_surv=True, true_qh=False)),
    ("A3  Oracle (all estimated)", "oracle", dict(true_surv=False, true_qh=False)),
    ("B1  R-CSF baseline (X only)", "b1", {}),
    ("B2  R-CSF baseline (X+W+Z)", "b2", {}),
    ("C1  NC-CSF (all true)", "nc", dict(true_surv=True, true_qh=True)),
    ("C2  NC-CSF (true surv, est q/h)", "nc", dict(true_surv=True, true_qh=False)),
    ("C3  NC-CSF (all estimated)", "nc", dict(true_surv=False, true_qh=False)),
]

Q_CLIP = 0.02
Y_TILDE_CLIP_QUANTILE = 0.99
Y_RES_CLIP_PERCENTILES = (1.0, 99.0)
SURV_FLOOR = 0.025
MAX_GRID = 500


@dataclass
class CaseData:
    cfg: SynthConfig
    dgp: dict[str, np.ndarray | float | None]
    obs_df: pd.DataFrame
    truth_df: pd.DataFrame
    x_cols: list[str]
    X: np.ndarray
    W: np.ndarray
    Z: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    delta: np.ndarray
    U: np.ndarray
    true_cate: np.ndarray
    horizon: float


def build_case_cfg(case_spec: dict[str, object]) -> SynthConfig:
    cfg_kwargs = dict(BASE_CONFIG)
    cfg_kwargs.update(case_spec["cfg"])
    return SynthConfig(**cfg_kwargs)


def _weibull_scale(lam: float, k: float, eta: np.ndarray) -> np.ndarray:
    return lam * np.exp(-eta / k)


def mean_survival_given_eta(eta: np.ndarray, cfg: SynthConfig) -> np.ndarray:
    scale = _weibull_scale(cfg.lam_t, cfg.k_t, np.asarray(eta, dtype=float))
    return scale * math.gamma(1.0 + 1.0 / cfg.k_t)


def survival_probability_given_eta(eta: np.ndarray, horizon: float, cfg: SynthConfig) -> np.ndarray:
    scale = _weibull_scale(cfg.lam_t, cfg.k_t, np.asarray(eta, dtype=float))
    return np.exp(-((float(horizon) / scale) ** cfg.k_t))


def recover_dgp_internals(cfg: SynthConfig) -> dict[str, np.ndarray | float | None]:
    rng = np.random.default_rng(cfg.seed)
    n, p = cfg.n, cfg.p_x

    X = rng.normal(size=(n, p))
    U = rng.normal(size=n)
    b_z = rng.normal(scale=0.3, size=p)
    b_w = rng.normal(scale=0.3, size=p)
    rng.normal(scale=cfg.sigma_z, size=n)
    rng.normal(scale=cfg.sigma_w, size=n)

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
        nonlinear_part = x_squared @ beta_squared + x_interact @ beta_interact + 0.5 * sigmoid(U)
        eta_t0 = X @ beta_t + cfg.beta_u_in_t * U + nonlinear_part
        eta_t1 = eta_t0 + cfg.tau_log_hr

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
            c_mid = weibull_ph_time_paper(u_c, k=cfg.k_c, lam=mid, eta=eta_c)
            if (c_mid < t_obs).mean() < cfg.target_censor_rate:
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


def _treatment_logit_score(X: np.ndarray, U: np.ndarray, cfg: SynthConfig, dgp: dict[str, np.ndarray | float | None]):
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


def _event_eta(X: np.ndarray, U: np.ndarray, A: np.ndarray | float, cfg: SynthConfig, dgp: dict[str, np.ndarray | float | None]):
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


def _posterior_u_given_proxy(proxy: np.ndarray, X: np.ndarray, beta_proxy: np.ndarray, proxy_loading: float, noise_scale: float):
    proxy = np.asarray(proxy, dtype=float).reshape(-1)
    sigma2 = noise_scale**2
    a2 = proxy_loading**2
    posterior_var = sigma2 / (sigma2 + a2)
    posterior_mean = proxy_loading * (proxy - X @ beta_proxy) / (sigma2 + a2)
    return posterior_mean, posterior_var


def _gauss_hermite_expectation(posterior_mean: np.ndarray, posterior_var: float, fn, quadrature_nodes: int = 30):
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_nodes)
    u_draws = posterior_mean[:, np.newaxis] + np.sqrt(2.0 * posterior_var) * nodes[np.newaxis, :]
    return np.sum(fn(u_draws) * weights[np.newaxis, :], axis=1) / np.sqrt(np.pi)


def true_censoring_on_grid(X: np.ndarray, U: np.ndarray, Y: np.ndarray, time_grid: np.ndarray, cfg: SynthConfig, beta_c: np.ndarray, lam_c: float):
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


def true_event_surv_on_grid(X: np.ndarray, U: np.ndarray, A: np.ndarray, time_grid: np.ndarray, cfg: SynthConfig, dgp: dict[str, np.ndarray | float | None]):
    eta_t = _event_eta(X, U, A, cfg, dgp)
    scale_t = _weibull_scale(cfg.lam_t, cfg.k_t, eta_t)
    return np.exp(-((time_grid[np.newaxis, :] / scale_t[:, np.newaxis]) ** cfg.k_t))


def true_propensity_oracle(X: np.ndarray, U: np.ndarray, dgp: dict[str, np.ndarray | float | None], cfg: SynthConfig):
    return sigmoid(float(dgp["b0"]) + _treatment_logit_score(X, U, cfg, dgp))


def true_propensity_nc(Z: np.ndarray, X: np.ndarray, dgp: dict[str, np.ndarray | float | None], cfg: SynthConfig):
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
    *,
    target: str = "RMST",
    horizon: float | None = None,
):
    if target == "survival.probability":
        if horizon is None:
            raise ValueError("horizon is required for target='survival.probability'.")
        h0 = survival_probability_given_eta(_event_eta(X, U, np.zeros(X.shape[0]), cfg, dgp), horizon, cfg)
        h1 = survival_probability_given_eta(_event_eta(X, U, np.ones(X.shape[0]), cfg, dgp), horizon, cfg)
        return h0, h1
    h0 = mean_survival_given_eta(_event_eta(X, U, np.zeros(X.shape[0]), cfg, dgp), cfg)
    h1 = mean_survival_given_eta(_event_eta(X, U, np.ones(X.shape[0]), cfg, dgp), cfg)
    return h0, h1


def true_outcome_nc(
    W: np.ndarray,
    X: np.ndarray,
    cfg: SynthConfig,
    dgp: dict[str, np.ndarray | float | None],
    *,
    target: str = "RMST",
    horizon: float | None = None,
):
    posterior_mean, posterior_var = _posterior_u_given_proxy(
        W,
        X,
        np.asarray(dgp["b_w"], dtype=float),
        cfg.aW,
        cfg.sigma_w,
    )
    if target == "survival.probability":
        if horizon is None:
            raise ValueError("horizon is required for target='survival.probability'.")
        h0 = _gauss_hermite_expectation(
            posterior_mean,
            posterior_var,
            lambda u_draws: survival_probability_given_eta(
                _event_eta(X, u_draws, np.zeros(X.shape[0]), cfg, dgp),
                horizon,
                cfg,
            ),
        )
        h1 = _gauss_hermite_expectation(
            posterior_mean,
            posterior_var,
            lambda u_draws: survival_probability_given_eta(
                _event_eta(X, u_draws, np.ones(X.shape[0]), cfg, dgp),
                horizon,
                cfg,
            ),
        )
        return h0, h1
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


def _cap_time_grid(time_values):
    grid = np.sort(np.unique(np.asarray(time_values, dtype=float)))
    if grid.size > MAX_GRID:
        idx = np.linspace(0, grid.size - 1, MAX_GRID, dtype=int)
        grid = grid[idx]
    return grid


def _build_true_y_tilde(x_base, u_vec, y_time, delta, cfg, dgp, *, target="RMST", horizon=None):
    if target == "survival.probability":
        if horizon is None:
            raise ValueError("horizon is required for target='survival.probability'.")
        eval_time = np.minimum(np.asarray(y_time, dtype=float), float(horizon))
        dummy_grid = np.array([float(np.max(eval_time))], dtype=float)
        _, _, sc_at_y = true_censoring_on_grid(x_base, u_vec, eval_time, dummy_grid, cfg, dgp["beta_c"], dgp["lam_c"])
        sc_at_y = np.maximum(sc_at_y, SURV_FLOOR)
        return (np.asarray(y_time, dtype=float) > float(horizon)).astype(float) / sc_at_y

    dummy_grid = np.array([np.max(y_time)], dtype=float)
    _, _, sc_at_y = true_censoring_on_grid(x_base, u_vec, y_time, dummy_grid, cfg, dgp["beta_c"], dgp["lam_c"])
    sc_at_y = np.maximum(sc_at_y, SURV_FLOOR)
    return y_time * delta / sc_at_y


def _true_survival_components(x_base, u_vec, eval_time, grid_time, cfg, dgp):
    t_grid = _cap_time_grid(grid_time)
    surv_c, hazard_c, sc_at_y = true_censoring_on_grid(x_base, u_vec, eval_time, t_grid, cfg, dgp["beta_c"], dgp["lam_c"])
    surv_c = np.maximum(surv_c, SURV_FLOOR)
    sc_at_y = np.maximum(sc_at_y, SURV_FLOOR)
    return t_grid, surv_c, hazard_c, sc_at_y


def _compute_true_ipcw_3term_y_res(y_time, delta, m_pred, q_hat, t_grid, surv_c, hazard_c, sc_at_y, *, clip_percentiles):
    n, grid_size = q_hat.shape
    y_idx = np.searchsorted(t_grid, y_time, side="right") - 1
    y_idx = np.clip(y_idx, 0, grid_size - 1)
    q_at_y = q_hat[np.arange(n), y_idx]

    term1 = (delta * y_time + (1.0 - delta) * q_at_y) / np.maximum(sc_at_y, 1e-10)

    grid_weight = hazard_c / np.maximum(surv_c, 1e-10)
    integrand = grid_weight * q_hat
    mask = np.arange(grid_size)[None, :] <= y_idx[:, None]
    term2 = (integrand * mask).sum(axis=1)

    y_res = (term1 - term2) - m_pred
    lo, hi = np.percentile(y_res, clip_percentiles)
    return np.clip(y_res, lo, hi)


def _compute_true_target_ipcw_3term_y_res(
    f_y,
    eval_time,
    eval_delta,
    m_pred,
    q_hat,
    t_grid,
    surv_c,
    hazard_c,
    sc_at_eval,
    *,
    clip_percentiles,
):
    n, grid_size = q_hat.shape
    y_idx = np.searchsorted(t_grid, eval_time, side="right") - 1
    y_idx = np.clip(y_idx, 0, grid_size - 1)
    q_at_y = q_hat[np.arange(n), y_idx]

    term1 = (eval_delta * (f_y - m_pred) + (1.0 - eval_delta) * (q_at_y - m_pred)) / np.maximum(sc_at_eval, 1e-10)

    grid_weight = hazard_c / np.maximum(surv_c, 1e-10)
    integrand = grid_weight * (q_hat - m_pred[:, None])
    mask = np.arange(grid_size)[None, :] <= y_idx[:, None]
    term2 = (integrand * mask).sum(axis=1)

    y_res = term1 - term2
    lo, hi = np.percentile(y_res, clip_percentiles)
    return np.clip(y_res, lo, hi)


class _BenchmarkNCSurvivalNuisance:
    def __init__(
        self,
        cfg,
        dgp,
        p_x,
        z_proxy_dim,
        q_model,
        h_model,
        *,
        true_surv,
        true_qh,
        target,
        horizon,
        q_clip,
        y_tilde_clip_quantile,
        y_res_clip_percentiles,
    ):
        self._cfg = cfg
        self._dgp = dgp
        self._p_x = p_x
        self._z_proxy_dim = z_proxy_dim
        self._true_surv = true_surv
        self._true_qh = true_qh
        self._target = target
        self._horizon = horizon
        self._q_model_template = q_model
        self._h_model_template = h_model
        self._q_clip = q_clip
        self._y_tilde_clip_quantile = y_tilde_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._q_model = None
        self._h1_model = None
        self._h0_model = None
        self._km_times = None
        self._km_surv = None
        self._event_cox_1 = None
        self._event_cox_0 = None
        self._cox_col_names = None
        self._t_grid = None

    @staticmethod
    def _unpack_y(y):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] >= 2:
            return y[:, 0], y[:, 1]
        raise ValueError("Y must be a 2-column array [time, event].")

    def _split_inputs(self, x, w, z):
        x_full = np.asarray(x)
        x_base = x_full[:, : self._p_x]
        w_proxy = _ensure_2d(w)
        z_bundle = _ensure_2d(z)
        z_proxy = z_bundle[:, : self._z_proxy_dim]
        u_cols = z_bundle[:, self._z_proxy_dim :]
        u_vec = None if u_cols.shape[1] == 0 else u_cols[:, 0]
        return x_full, x_base, w_proxy, z_proxy, u_vec

    def train(self, is_selecting, folds, y, t, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_time, delta = self._unpack_y(y)
        a = np.asarray(t).ravel()
        x_full, x_base, w_proxy, z_proxy, u_vec = self._split_inputs(X, W, Z)
        target_inputs = _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )

        if self._true_surv:
            if u_vec is None:
                raise ValueError("True survival nuisances require U in the Z bundle.")
            y_tilde = _build_true_y_tilde(
                x_base,
                u_vec,
                y_time,
                delta,
                self._cfg,
                self._dgp,
                target=self._target,
                horizon=self._horizon,
            )
        else:
            self._km_times, self._km_surv = _fit_kaplan_meier_censoring(
                target_inputs["nuisance_time"],
                target_inputs["nuisance_delta"],
            )
            y_tilde = _compute_target_pseudo_outcome(
                y_time=y_time,
                delta=delta,
                target=self._target,
                horizon=self._horizon,
                nuisance_time=target_inputs["nuisance_time"],
                nuisance_delta=target_inputs["nuisance_delta"],
                km_times=self._km_times,
                km_surv=self._km_surv,
            )
        y_tilde = _clip_quantile(y_tilde, self._y_tilde_clip_quantile)

        if not self._true_qh:
            xz = np.column_stack([x_full, z_proxy])
            xw = np.column_stack([x_full, w_proxy])
            self._q_model = clone(self._q_model_template)
            self._q_model.fit(xz, a, **filter_none_kwargs(sample_weight=sample_weight))

            treated_mask = a == 1
            control_mask = a == 0
            self._h1_model = clone(self._h_model_template)
            self._h0_model = clone(self._h_model_template)

            if treated_mask.sum() > 10:
                self._h1_model.fit(xw[treated_mask], y_tilde[treated_mask], **filter_none_kwargs(sample_weight=None if sample_weight is None else sample_weight[treated_mask]))
            if control_mask.sum() > 10:
                self._h0_model.fit(xw[control_mask], y_tilde[control_mask], **filter_none_kwargs(sample_weight=None if sample_weight is None else sample_weight[control_mask]))

        if not self._true_surv:
            surv_features = np.column_stack([x_full, w_proxy, z_proxy])
            treated_mask = a == 1
            control_mask = a == 0
            self._event_cox_1, self._cox_col_names = _fit_event_cox(
                target_inputs["nuisance_time"][treated_mask],
                target_inputs["nuisance_delta"][treated_mask],
                surv_features[treated_mask],
            )
            self._event_cox_0, _ = _fit_event_cox(
                target_inputs["nuisance_time"][control_mask],
                target_inputs["nuisance_delta"][control_mask],
                surv_features[control_mask],
            )
            self._t_grid = _cap_time_grid(target_inputs["grid_time"])

        return self

    def predict(self, y, t, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_time, delta = self._unpack_y(y)
        a = np.asarray(t).ravel()
        x_full, x_base, w_proxy, z_proxy, u_vec = self._split_inputs(X, W, Z)
        target_inputs = _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )

        if self._true_qh:
            q_pred = true_propensity_nc(z_proxy, x_base, self._dgp, self._cfg)
            h0_pred, h1_pred = true_outcome_nc(
                w_proxy,
                x_base,
                self._cfg,
                self._dgp,
                target=self._target,
                horizon=self._horizon,
            )
        else:
            xz = np.column_stack([x_full, z_proxy])
            xw = np.column_stack([x_full, w_proxy])
            q_pred = self._q_model.predict_proba(xz)[:, 1]
            h1_pred = self._h1_model.predict(xw)
            h0_pred = self._h0_model.predict(xw)

        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        m_pred = q_pred * h1_pred + (1.0 - q_pred) * h0_pred

        if self._true_surv:
            if u_vec is None:
                raise ValueError("True survival nuisances require bundled U.")
            t_grid, surv_c, hazard_c, sc_at_y = _true_survival_components(
                x_base,
                u_vec,
                target_inputs["eval_time"],
                target_inputs["grid_time"],
                self._cfg,
                self._dgp,
            )
            s_hat_1 = true_event_surv_on_grid(x_base, u_vec, np.ones_like(a), t_grid, self._cfg, self._dgp)
            s_hat_0 = true_event_surv_on_grid(x_base, u_vec, np.zeros_like(a), t_grid, self._cfg, self._dgp)
            if self._target == "survival.probability":
                q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, t_grid, self._horizon)
                q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, t_grid, self._horizon)
            else:
                q_hat_1 = _compute_q_from_s(s_hat_1, t_grid)
                q_hat_0 = _compute_q_from_s(s_hat_0, t_grid)
            q_hat = np.where((a == 1)[:, None], q_hat_1, q_hat_0)
            if self._target == "RMST" and self._horizon is None:
                y_res = _compute_true_ipcw_3term_y_res(
                    y_time,
                    delta,
                    m_pred,
                    q_hat,
                    t_grid,
                    surv_c,
                    hazard_c,
                    sc_at_y,
                    clip_percentiles=self._y_res_clip_percentiles,
                )
            else:
                y_res = _compute_true_target_ipcw_3term_y_res(
                    target_inputs["f_y"],
                    target_inputs["eval_time"],
                    target_inputs["eval_delta"],
                    m_pred,
                    q_hat,
                    t_grid,
                    surv_c,
                    hazard_c,
                    sc_at_y,
                    clip_percentiles=self._y_res_clip_percentiles,
                )
        else:
            surv_features = np.column_stack([x_full, w_proxy, z_proxy])
            s_hat_1 = _predict_s_on_grid(self._event_cox_1, self._cox_col_names, surv_features, self._t_grid)
            s_hat_0 = _predict_s_on_grid(self._event_cox_0, self._cox_col_names, surv_features, self._t_grid)
            if self._target == "survival.probability":
                q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, self._t_grid, self._horizon)
                q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, self._t_grid, self._horizon)
            else:
                q_hat_1 = _compute_q_from_s(s_hat_1, self._t_grid)
                q_hat_0 = _compute_q_from_s(s_hat_0, self._t_grid)
            q_hat = np.where((a == 1)[:, None], q_hat_1, q_hat_0)
            if self._target == "RMST" and self._horizon is None:
                y_res = _compute_ipcw_3term_y_res(
                    y_time,
                    delta,
                    m_pred,
                    q_hat,
                    self._t_grid,
                    self._km_times,
                    self._km_surv,
                    clip_percentiles=self._y_res_clip_percentiles,
                )
            else:
                y_res = _compute_target_ipcw_3term_y_res(
                    target_inputs["f_y"],
                    target_inputs["eval_time"],
                    target_inputs["eval_delta"],
                    m_pred,
                    q_hat,
                    self._t_grid,
                    self._km_times,
                    self._km_surv,
                    clip_percentiles=self._y_res_clip_percentiles,
                )

        a_res = (a - q_pred).reshape(-1, 1)
        return y_res, a_res


class _BenchmarkOracleSurvivalNuisance:
    def __init__(
        self,
        cfg,
        dgp,
        p_x,
        q_model,
        h_model,
        *,
        true_surv,
        true_qh,
        target,
        horizon,
        q_clip,
        y_tilde_clip_quantile,
        y_res_clip_percentiles,
    ):
        self._cfg = cfg
        self._dgp = dgp
        self._p_x = p_x
        self._true_surv = true_surv
        self._true_qh = true_qh
        self._target = target
        self._horizon = horizon
        self._q_model_template = q_model
        self._h_model_template = h_model
        self._q_clip = q_clip
        self._y_tilde_clip_quantile = y_tilde_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._q_model = None
        self._h1_model = None
        self._h0_model = None
        self._km_times = None
        self._km_surv = None
        self._event_cox_1 = None
        self._event_cox_0 = None
        self._cox_col_names = None
        self._t_grid = None

    @staticmethod
    def _unpack_y(y):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] >= 2:
            return y[:, 0], y[:, 1]
        raise ValueError("Y must be a 2-column array [time, event].")

    def _split_inputs(self, x, w):
        x_final = np.asarray(x)
        x_base = x_final[:, : self._p_x]
        u_vec = _ensure_2d(w)[:, 0]
        return x_final, x_base, u_vec

    def train(self, is_selecting, folds, y, t, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_time, delta = self._unpack_y(y)
        a = np.asarray(t).ravel()
        x_final, x_base, u_vec = self._split_inputs(X, W)
        target_inputs = _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )

        if self._true_surv:
            y_tilde = _build_true_y_tilde(
                x_base,
                u_vec,
                y_time,
                delta,
                self._cfg,
                self._dgp,
                target=self._target,
                horizon=self._horizon,
            )
        else:
            self._km_times, self._km_surv = _fit_kaplan_meier_censoring(
                target_inputs["nuisance_time"],
                target_inputs["nuisance_delta"],
            )
            y_tilde = _compute_target_pseudo_outcome(
                y_time=y_time,
                delta=delta,
                target=self._target,
                horizon=self._horizon,
                nuisance_time=target_inputs["nuisance_time"],
                nuisance_delta=target_inputs["nuisance_delta"],
                km_times=self._km_times,
                km_surv=self._km_surv,
            )
        y_tilde = _clip_quantile(y_tilde, self._y_tilde_clip_quantile)

        if not self._true_qh:
            self._q_model = clone(self._q_model_template)
            self._q_model.fit(x_final, a, **filter_none_kwargs(sample_weight=sample_weight))

            treated_mask = a == 1
            control_mask = a == 0
            self._h1_model = clone(self._h_model_template)
            self._h0_model = clone(self._h_model_template)

            if treated_mask.sum() > 10:
                self._h1_model.fit(x_final[treated_mask], y_tilde[treated_mask], **filter_none_kwargs(sample_weight=None if sample_weight is None else sample_weight[treated_mask]))
            if control_mask.sum() > 10:
                self._h0_model.fit(x_final[control_mask], y_tilde[control_mask], **filter_none_kwargs(sample_weight=None if sample_weight is None else sample_weight[control_mask]))

        if not self._true_surv:
            treated_mask = a == 1
            control_mask = a == 0
            self._event_cox_1, self._cox_col_names = _fit_event_cox(
                target_inputs["nuisance_time"][treated_mask],
                target_inputs["nuisance_delta"][treated_mask],
                x_final[treated_mask],
            )
            self._event_cox_0, _ = _fit_event_cox(
                target_inputs["nuisance_time"][control_mask],
                target_inputs["nuisance_delta"][control_mask],
                x_final[control_mask],
            )
            self._t_grid = _cap_time_grid(target_inputs["grid_time"])

        return self

    def predict(self, y, t, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_time, delta = self._unpack_y(y)
        a = np.asarray(t).ravel()
        x_final, x_base, u_vec = self._split_inputs(X, W)
        target_inputs = _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )

        if self._true_qh:
            q_pred = true_propensity_oracle(x_base, u_vec, self._dgp, self._cfg)
            h0_pred, h1_pred = true_outcome_oracle(
                x_base,
                u_vec,
                self._cfg,
                self._dgp,
                target=self._target,
                horizon=self._horizon,
            )
        else:
            q_pred = self._q_model.predict_proba(x_final)[:, 1]
            h1_pred = self._h1_model.predict(x_final)
            h0_pred = self._h0_model.predict(x_final)

        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        m_pred = q_pred * h1_pred + (1.0 - q_pred) * h0_pred

        if self._true_surv:
            t_grid, surv_c, hazard_c, sc_at_y = _true_survival_components(
                x_base,
                u_vec,
                target_inputs["eval_time"],
                target_inputs["grid_time"],
                self._cfg,
                self._dgp,
            )
            s_hat_1 = true_event_surv_on_grid(x_base, u_vec, np.ones_like(a), t_grid, self._cfg, self._dgp)
            s_hat_0 = true_event_surv_on_grid(x_base, u_vec, np.zeros_like(a), t_grid, self._cfg, self._dgp)
            if self._target == "survival.probability":
                q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, t_grid, self._horizon)
                q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, t_grid, self._horizon)
            else:
                q_hat_1 = _compute_q_from_s(s_hat_1, t_grid)
                q_hat_0 = _compute_q_from_s(s_hat_0, t_grid)
            q_hat = np.where((a == 1)[:, None], q_hat_1, q_hat_0)
            if self._target == "RMST" and self._horizon is None:
                y_res = _compute_true_ipcw_3term_y_res(
                    y_time,
                    delta,
                    m_pred,
                    q_hat,
                    t_grid,
                    surv_c,
                    hazard_c,
                    sc_at_y,
                    clip_percentiles=self._y_res_clip_percentiles,
                )
            else:
                y_res = _compute_true_target_ipcw_3term_y_res(
                    target_inputs["f_y"],
                    target_inputs["eval_time"],
                    target_inputs["eval_delta"],
                    m_pred,
                    q_hat,
                    t_grid,
                    surv_c,
                    hazard_c,
                    sc_at_y,
                    clip_percentiles=self._y_res_clip_percentiles,
                )
        else:
            s_hat_1 = _predict_s_on_grid(self._event_cox_1, self._cox_col_names, x_final, self._t_grid)
            s_hat_0 = _predict_s_on_grid(self._event_cox_0, self._cox_col_names, x_final, self._t_grid)
            if self._target == "survival.probability":
                q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, self._t_grid, self._horizon)
                q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, self._t_grid, self._horizon)
            else:
                q_hat_1 = _compute_q_from_s(s_hat_1, self._t_grid)
                q_hat_0 = _compute_q_from_s(s_hat_0, self._t_grid)
            q_hat = np.where((a == 1)[:, None], q_hat_1, q_hat_0)
            if self._target == "RMST" and self._horizon is None:
                y_res = _compute_ipcw_3term_y_res(
                    y_time,
                    delta,
                    m_pred,
                    q_hat,
                    self._t_grid,
                    self._km_times,
                    self._km_surv,
                    clip_percentiles=self._y_res_clip_percentiles,
                )
            else:
                y_res = _compute_target_ipcw_3term_y_res(
                    target_inputs["f_y"],
                    target_inputs["eval_time"],
                    target_inputs["eval_delta"],
                    m_pred,
                    q_hat,
                    self._t_grid,
                    self._km_times,
                    self._km_surv,
                    clip_percentiles=self._y_res_clip_percentiles,
                )

        a_res = (a - q_pred).reshape(-1, 1)
        return y_res, a_res


class BenchmarkNCSurvivalForestDML(EconmlMildShrinkNCSurvivalForest):
    def __init__(self, cfg, dgp, p_x, *, true_surv, true_qh, z_proxy_dim=1, **kwargs):
        self._benchmark_cfg = cfg
        self._benchmark_dgp = dgp
        self._benchmark_p_x = p_x
        self._benchmark_true_surv = true_surv
        self._benchmark_true_qh = true_qh
        self._benchmark_z_proxy_dim = z_proxy_dim
        super().__init__(**kwargs)

    def _gen_ortho_learner_model_nuisance(self):
        return _BenchmarkNCSurvivalNuisance(
            cfg=self._benchmark_cfg,
            dgp=self._benchmark_dgp,
            p_x=self._benchmark_p_x,
            z_proxy_dim=self._benchmark_z_proxy_dim,
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            true_surv=self._benchmark_true_surv,
            true_qh=self._benchmark_true_qh,
            target=self._target,
            horizon=self._horizon,
            q_clip=self._custom_q_clip,
            y_tilde_clip_quantile=self._custom_y_tilde_clip_quantile,
            y_res_clip_percentiles=self._custom_y_res_clip_percentiles,
        )


class BenchmarkOracleSurvivalForestDML(EconmlMildShrinkNCSurvivalForest):
    def __init__(self, cfg, dgp, p_x, *, true_surv, true_qh, **kwargs):
        self._benchmark_cfg = cfg
        self._benchmark_dgp = dgp
        self._benchmark_p_x = p_x
        self._benchmark_true_surv = true_surv
        self._benchmark_true_qh = true_qh
        super().__init__(**kwargs)

    def _gen_ortho_learner_model_nuisance(self):
        return _BenchmarkOracleSurvivalNuisance(
            cfg=self._benchmark_cfg,
            dgp=self._benchmark_dgp,
            p_x=self._benchmark_p_x,
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            true_surv=self._benchmark_true_surv,
            true_qh=self._benchmark_true_qh,
            target=self._target,
            horizon=self._horizon,
            q_clip=self._custom_q_clip,
            y_tilde_clip_quantile=self._custom_y_tilde_clip_quantile,
            y_res_clip_percentiles=self._custom_y_res_clip_percentiles,
        )

    def fit_oracle(self, X, A, time, event, U, **kwargs):
        x = np.asarray(X, dtype=float)
        y = np.asarray(time, dtype=float).ravel()
        delta = np.asarray(event, dtype=float).ravel()
        u = _ensure_2d(U)
        y_packed = np.column_stack([y, delta])
        z_dummy = np.zeros((len(y), 1), dtype=float)
        return _OrthoLearner.fit(self, y_packed, A, X=x, W=u, Z=z_dummy, **kwargs)


def _make_forest_kwargs():
    return dict(
        n_estimators=200,
        min_samples_leaf=20,
        cv=3,
        random_state=42,
    )


def prepare_case(
    case_spec: dict[str, object],
    *,
    target: str = "RMST",
    horizon_quantile: float = 0.60,
) -> CaseData:
    cfg = build_case_cfg(case_spec)
    obs_df, truth_df, params = generate_synthetic_nc_cox(cfg)
    obs_df, truth_df = add_ground_truth_cate(obs_df, truth_df, cfg, params)
    dgp = recover_dgp_internals(cfg)
    x_cols = [f"X{j}" for j in range(cfg.p_x)]
    X = obs_df[x_cols].to_numpy()
    W = obs_df[["W"]].to_numpy()
    Z = obs_df[["Z"]].to_numpy()
    A = obs_df["A"].to_numpy()
    Y = obs_df["time"].to_numpy()
    delta = obs_df["event"].to_numpy()
    U = truth_df["U"].to_numpy()
    if target == "survival.probability":
        horizon = float(np.quantile(Y, horizon_quantile))
        eta0 = _event_eta(X, U, np.zeros(X.shape[0]), cfg, dgp)
        eta1 = _event_eta(X, U, np.ones(X.shape[0]), cfg, dgp)
        true_cate = survival_probability_given_eta(eta1, horizon, cfg) - survival_probability_given_eta(eta0, horizon, cfg)
    else:
        true_cate = truth_df["CATE_XU_eq7"].to_numpy()
        horizon = float(np.max(Y))
    return CaseData(cfg, dgp, obs_df, truth_df, x_cols, X, W, Z, A, Y, delta, U, true_cate, horizon)


def _evaluate_predictions(name, preds, true_cate, elapsed, backend):
    sign_acc = float(np.mean(np.sign(preds) == np.sign(true_cate)))
    pehe = float(np.sqrt(np.mean((preds - true_cate) ** 2)))
    return dict(
        name=name,
        mean_pred=float(np.mean(preds)),
        std_pred=float(np.std(preds)),
        mean_true_cate=float(np.mean(true_cate)),
        std_true_cate=float(np.std(true_cate)),
        bias=float(np.mean(preds - true_cate)),
        rmse=pehe,
        pehe=pehe,
        mae=float(np.mean(np.abs(preds - true_cate))),
        pearson=float(np.corrcoef(preds, true_cate)[0, 1]),
        sign_acc=sign_acc,
        total_time=float(elapsed),
        backend=backend,
    )


def evaluate_r_csf_variant(name, obs_df, feature_cols, true_cate, horizon, num_trees=200, *, target="RMST"):
    (PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs", prefix="benchmark8_r_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        input_path = tmp_dir / "input.csv"
        output_path = tmp_dir / "predictions.csv"
        obs_df.loc[:, ["time", "event", "A", *feature_cols]].to_csv(input_path, index=False)
        cmd = [
            "Rscript",
            str(R_CSF_SCRIPT),
            str(input_path),
            ",".join(feature_cols),
            str(horizon),
            str(num_trees),
            str(output_path),
            str(target),
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"Installed R grf baseline failed.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        preds = pd.read_csv(output_path)["prediction"].to_numpy(dtype=float)
    return _evaluate_predictions(name, preds, true_cate, elapsed, backend="installed R grf")


def _evaluate_oracle_variant(name, case: CaseData, *, true_surv: bool, true_qh: bool, target: str):
    x_oracle = np.column_stack([case.X, case.U])
    model = BenchmarkOracleSurvivalForestDML(
        cfg=case.cfg,
        dgp=case.dgp,
        p_x=case.X.shape[1],
        true_surv=true_surv,
        true_qh=true_qh,
        target=target,
        horizon=case.horizon,
        **_make_forest_kwargs(),
    )
    t0 = time.time()
    model.fit_oracle(x_oracle, case.A, case.Y, case.delta, case.U)
    preds = model.effect(x_oracle).ravel()
    elapsed = time.time() - t0
    return _evaluate_predictions(name, preds, case.true_cate, elapsed, backend="econml mild shrink")


def _evaluate_nc_variant(name, case: CaseData, *, true_surv: bool, true_qh: bool, target: str):
    x_full = np.hstack([case.X, case.W, case.Z])
    z_bundle = np.hstack([case.Z, _ensure_2d(case.U)])
    model = BenchmarkNCSurvivalForestDML(
        cfg=case.cfg,
        dgp=case.dgp,
        p_x=case.X.shape[1],
        z_proxy_dim=case.Z.shape[1],
        true_surv=true_surv,
        true_qh=true_qh,
        target=target,
        horizon=case.horizon,
        **_make_forest_kwargs(),
    )
    t0 = time.time()
    model.fit_survival(x_full, case.A, case.Y, case.delta, z_bundle, case.W)
    preds = model.effect(x_full).ravel()
    elapsed = time.time() - t0
    return _evaluate_predictions(name, preds, case.true_cate, elapsed, backend="econml mild shrink")


def run_case_benchmark(
    case_spec: dict[str, object],
    *,
    num_trees_b2: int = 200,
    verbose: bool = False,
    target: str = "RMST",
    horizon_quantile: float = 0.60,
) -> pd.DataFrame:
    case = prepare_case(case_spec, target=target, horizon_quantile=horizon_quantile)
    rows: list[dict[str, object]] = []
    for name, kind, kwargs in EIGHT_VARIANT_SPECS:
        if verbose:
            print(f"  - {name}", flush=True)
        if kind == "oracle":
            row = _evaluate_oracle_variant(name, case, target=target, **kwargs)
        elif kind == "nc":
            row = _evaluate_nc_variant(name, case, target=target, **kwargs)
        elif kind == "b1":
            row = evaluate_r_csf_variant(
                name,
                case.obs_df,
                case.x_cols,
                case.true_cate,
                case.horizon,
                num_trees_b2,
                target=target,
            )
        elif kind == "b2":
            row = evaluate_r_csf_variant(
                name,
                case.obs_df,
                case.x_cols + ["W", "Z"],
                case.true_cate,
                case.horizon,
                num_trees_b2,
                target=target,
            )
        else:
            raise ValueError(f"Unknown variant kind: {kind}")

        row.update(
            case_id=case_spec["case_id"],
            case_slug=case_spec["slug"],
            case_title=case_spec["title"],
            n=case.cfg.n,
            p_x=case.cfg.p_x,
            seed=case.cfg.seed,
            target=target,
            estimand_horizon=float(case.horizon),
            horizon_quantile=float(horizon_quantile) if target == "survival.probability" else None,
            target_censor_rate=case.cfg.target_censor_rate,
            actual_censor_rate=float(1.0 - case.delta.mean()),
            linear_treatment=case.cfg.linear_treatment,
            linear_outcome=case.cfg.linear_outcome,
            tau_log_hr=case.cfg.tau_log_hr,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_results(combined_df: pd.DataFrame):
    summary = (
        combined_df.groupby("name", as_index=False)
        .agg(
            avg_pred_cate=("mean_pred", "mean"),
            avg_true_cate=("mean_true_cate", "mean"),
            avg_acc=("sign_acc", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_pehe=("pehe", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
        )
        .sort_values(["avg_pehe", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    top5 = summary.head(5).copy()
    return summary, top5


def render_case_table_png(case_df: pd.DataFrame, output_path: Path):
    display = case_df.loc[
        :,
        ["name", "mean_pred", "mean_true_cate", "bias", "rmse", "pehe", "mae", "pearson", "total_time"],
    ].copy()
    display.columns = ["Variant", "Pred CATE", "True CATE", "Bias", "RMSE", "PEHE", "MAE", "Pearson", "Time"]
    for col in ["Pred CATE", "True CATE", "Bias", "RMSE", "PEHE", "MAE", "Pearson"]:
        display[col] = display[col].map(lambda value: f"{value:.4f}")
    display["Time"] = display["Time"].map(lambda value: f"{value:.1f}s")

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.axis("off")
    title = case_df["case_title"].iloc[0]
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)
    ax.set_title("8-variant benchmark", fontsize=13, color="#4b5563", pad=18)

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0.02, 1, 0.86],
        colWidths=[0.40, 0.095, 0.095, 0.09, 0.08, 0.08, 0.08, 0.09, 0.075],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.55)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("#d1d5db")
            cell.set_linewidth(1.5)
        else:
            cell.set_facecolor("#ffffff" if row % 2 == 1 else "#f3f4f6")
            cell.set_edgecolor("#d1d5db")
            cell.set_linewidth(0.8)
            if col == 0:
                cell.set_text_props(ha="left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_avg_summary_png(summary_df: pd.DataFrame, output_path: Path):
    display = summary_df.copy()
    display["Avg Pred CATE"] = display["avg_pred_cate"].map(lambda value: f"{value:.4f}")
    display["Avg True CATE"] = display["avg_true_cate"].map(lambda value: f"{value:.4f}")
    display["Avg Acc"] = display["avg_acc"].map(lambda value: f"{100.0 * value:.1f}%")
    display["Avg RMSE"] = display["avg_rmse"].map(lambda value: f"{value:.4f}")
    display["Avg PEHE"] = display["avg_pehe"].map(lambda value: f"{value:.4f}")
    display["Avg MAE"] = display["avg_mae"].map(lambda value: f"{value:.4f}")
    display["Avg Pearson"] = display["avg_pearson"].map(lambda value: f"{value:.4f}")
    display["Avg Bias"] = display["avg_bias"].map(lambda value: f"{value:+.4f}")
    display = display.loc[:, ["rank", "name", "Avg Pred CATE", "Avg True CATE", "Avg Acc", "Avg RMSE", "Avg PEHE", "Avg MAE", "Avg Pearson", "Avg Bias"]]
    display.columns = ["Rank", "Variant", "Avg Pred CATE", "Avg True CATE", "Avg Acc", "Avg RMSE", "Avg PEHE", "Avg MAE", "Avg Pearson", "Avg Bias"]

    fig, ax = plt.subplots(figsize=(24, 9))
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.03, 0.98, 0.94],
        colWidths=[0.05, 0.27, 0.09, 0.1, 0.08, 0.09, 0.09, 0.08, 0.1, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.7)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2b3440")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor("#19212c")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#0b1220")
            cell.set_text_props(color="white")
            if col == 1:
                cell.set_text_props(ha="left", color="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def render_top5_png(top5_df: pd.DataFrame, output_path: Path):
    display = top5_df.copy()
    display["Avg RMSE"] = display["avg_rmse"].map(lambda value: f"{value:.4f}")
    display["Avg PEHE"] = display["avg_pehe"].map(lambda value: f"{value:.4f}")
    display["Avg MAE"] = display["avg_mae"].map(lambda value: f"{value:.4f}")
    display["Avg Pearson"] = display["avg_pearson"].map(lambda value: f"{value:.4f}")
    display["Avg Bias"] = display["avg_bias"].map(lambda value: f"{value:+.4f}")
    display = display.loc[:, ["rank", "name", "Avg RMSE", "Avg PEHE", "Avg MAE", "Avg Pearson", "Avg Bias"]]
    display.columns = ["Rank", "Variant", "Avg RMSE", "Avg PEHE", "Avg MAE", "Avg Pearson", "Avg Bias"]

    fig, ax = plt.subplots(figsize=(18, 6))
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.05, 0.96, 0.9],
        colWidths=[0.07, 0.40, 0.12, 0.12, 0.12, 0.12, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.7)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2b3440")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor("#19212c")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#0b1220")
            cell.set_text_props(color="white")
            if col == 1:
                cell.set_text_props(ha="left", color="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def render_b2_vs_c3_png(combined_df: pd.DataFrame, output_path: Path):
    subset = combined_df[combined_df["name"].isin(["B2  R-CSF baseline (X+W+Z)", "C3  NC-CSF (all estimated)"])].copy()
    order = combined_df.drop_duplicates("case_id").sort_values("case_id")[["case_id", "case_slug"]]
    case_labels = {
        1: "lin / lin / info / strong / bene / large",
        2: "lin / lin / info / weak / harm / small",
        3: "lin / lin / weakproxy / strong / harm / mod",
        4: "lin / lin / weakproxy / weak / near0 / harm",
        5: "lin / nonlin / info / strong / bene / large",
        6: "lin / nonlin / info / weak / harm / small",
        7: "lin / nonlin / weakproxy / strong / harm / mod",
        8: "lin / nonlin / weakproxy / weak / near0 / harm",
        9: "nonlin / nonlin / info / strong / bene / large",
        10: "nonlin / nonlin / info / weak / harm / small",
        11: "nonlin / nonlin / weakproxy / strong / harm / mod",
        12: "nonlin / nonlin / weakproxy / weak / near0 / harm",
    }

    b2 = subset[subset["name"] == "B2  R-CSF baseline (X+W+Z)"].sort_values("case_id")
    c3 = subset[subset["name"] == "C3  NC-CSF (all estimated)"].sort_values("case_id")
    c3_wins = int((c3["pehe"].to_numpy() < b2["pehe"].to_numpy()).sum())
    b2_wins = int((b2["pehe"].to_numpy() < c3["pehe"].to_numpy()).sum())

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.2])
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    y = np.arange(len(order))
    height = 0.34
    ax.barh(y - height / 2, c3["pehe"], height=height, color="#2ca89b", label="C3  NC-CSF (all estimated)")
    ax.barh(y + height / 2, b2["pehe"], height=height, color="#e76f51", label="B2  R-CSF baseline (X+W+Z)")
    ax.set_yticks(y)
    ax.set_yticklabels([case_labels[i] for i in order["case_id"]], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("sqrt-PEHE")
    ax.set_title("Case-wise sqrt-PEHE comparison", fontsize=22, fontweight="bold", pad=16)
    ax.grid(axis="x", alpha=0.25)
    for idx, (c3_val, b2_val) in enumerate(zip(c3["pehe"], b2["pehe"])):
        winner = "C3 best" if c3_val < b2_val else "B2 best"
        xpos = min(c3_val, b2_val) + 0.01
        ax.text(xpos, idx, winner, va="center", ha="left", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")

    ax2.bar(["C3", "B2"], [c3_wins, b2_wins], color=["#2ca89b", "#e76f51"])
    ax2.set_ylim(0, len(order) + 2)
    ax2.set_ylabel("Number of cases won")
    ax2.set_title("Who wins more cases?", fontsize=22, fontweight="bold", pad=16)
    for idx, value in enumerate([c3_wins, b2_wins]):
        ax2.text(idx, value + 0.15, f"{value}", ha="center", va="bottom", fontsize=18, fontweight="bold")
    annotation = (
        "Primary criterion: sqrt-PEHE\n"
        "Standard CATE metric in synthetic studies\n\n"
        f"C3 avg sqrt-PEHE  {c3['pehe'].mean():.4f}\n"
        f"B2 avg sqrt-PEHE  {b2['pehe'].mean():.4f}\n"
        f"C3 avg Pearson    {c3['pearson'].mean():.4f}\n"
        f"B2 avg Pearson    {b2['pearson'].mean():.4f}"
    )
    ax2.text(
        0.06,
        0.18,
        annotation,
        transform=ax2.transAxes,
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#bbbbbb"),
    )

    fig.suptitle("C3 tuned vs B2 across all 12 cases\nsqrt-PEHE-based comparison", fontsize=30, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
