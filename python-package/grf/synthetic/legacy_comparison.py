from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .survival import calibrate_intercept_for_prevalence, sigmoid, weibull_ph_time_paper


@dataclass
class LegacyComparisonConfig:
    n: int = 200
    p_x: int = 5
    seed: int = 42
    w_prevalence: float = 0.5
    gamma_u_in_w: float = 1.0
    k_t: float = 1.5
    lam_t: float = 0.4
    tau_log_hr: float = -0.6
    beta_u_in_t: float = 0.8
    k_c: float = 1.2
    lam_c: Optional[float] = None
    beta_u_in_c: float = 0.3
    target_censor_rate: float = 0.35
    max_censor_calib_iter: int = 60
    censor_lam_lo: float = 1e-8
    censor_lam_hi: float = 1e6
    admin_censor_time: Optional[float] = None
    az: float = 1.0
    av: float = 1.0
    sigma_z: float = 0.8
    sigma_v: float = 0.8


@dataclass
class LegacyComparisonParams:
    b_z: np.ndarray
    b_v: np.ndarray
    beta_t: np.ndarray


def generate_legacy_comparison_nc_cox(
    cfg: LegacyComparisonConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, LegacyComparisonParams]:
    rng = np.random.default_rng(cfg.seed)
    n, p = cfg.n, cfg.p_x

    X = rng.normal(size=(n, p))
    U = rng.normal(size=n)

    b_z = rng.normal(scale=0.3, size=p)
    b_v = rng.normal(scale=0.3, size=p)
    Z = cfg.az * U + X @ b_z + rng.normal(scale=cfg.sigma_z, size=n)
    V = cfg.av * U + X @ b_v + rng.normal(scale=cfg.sigma_v, size=n)

    alpha = rng.normal(scale=0.5, size=p)
    linpred = X @ alpha + cfg.gamma_u_in_w * U
    b0 = calibrate_intercept_for_prevalence(linpred, cfg.w_prevalence)
    p_w = sigmoid(b0 + linpred)
    W = rng.binomial(1, p_w, size=n).astype(int)

    beta_t = rng.normal(scale=0.4, size=p)
    u_t = rng.random(n)
    eta_t0 = X @ beta_t + cfg.beta_u_in_t * U
    eta_t1 = eta_t0 + cfg.tau_log_hr
    T0 = weibull_ph_time_paper(u_t, k=cfg.k_t, lam=cfg.lam_t, eta=eta_t0)
    T1 = weibull_ph_time_paper(u_t, k=cfg.k_t, lam=cfg.lam_t, eta=eta_t1)

    beta_c = rng.normal(scale=0.3, size=p)
    u_c = rng.random(n)
    eta_c = X @ beta_c + cfg.beta_u_in_c * U

    T_obs_for_calib = np.where(W == 1, T1, T0)
    lam_c_used = cfg.lam_c
    if lam_c_used is None:
        lo, hi = float(cfg.censor_lam_lo), float(cfg.censor_lam_hi)
        for _ in range(cfg.max_censor_calib_iter):
            mid = 0.5 * (lo + hi)
            C_mid = weibull_ph_time_paper(u_c, k=cfg.k_c, lam=mid, eta=eta_c)
            censor_rate_mid = (C_mid < T_obs_for_calib).mean()
            if censor_rate_mid < cfg.target_censor_rate:
                hi = mid
            else:
                lo = mid
        lam_c_used = 0.5 * (lo + hi)

    C0 = weibull_ph_time_paper(u_c, k=cfg.k_c, lam=lam_c_used, eta=eta_c)
    C1 = weibull_ph_time_paper(u_c, k=cfg.k_c, lam=lam_c_used, eta=eta_c)

    T = np.where(W == 1, T1, T0)
    C = np.where(W == 1, C1, C0)
    time = np.minimum(T, C)
    event = (T <= C).astype(int)

    if cfg.admin_censor_time is not None:
        admin = float(cfg.admin_censor_time)
        cens_by_admin = admin < time
        time = np.where(cens_by_admin, admin, time)
        event = np.where(cens_by_admin, 0, event).astype(int)

    x_cols = {f"X{j}": X[:, j] for j in range(p)}
    observed_df = pd.DataFrame({"time": time, "event": event, "W": W, "A": W, "Z": Z, "V": V, **x_cols})

    truth_df = observed_df.copy()
    truth_df.insert(0, "U", U)
    truth_df["T0"] = T0
    truth_df["T1"] = T1
    truth_df["C0"] = C0
    truth_df["C1"] = C1
    truth_df["T"] = T
    truth_df["C"] = C
    truth_df.attrs["lam_c_used"] = lam_c_used

    params = LegacyComparisonParams(b_z=b_z, b_v=b_v, beta_t=beta_t)
    return observed_df, truth_df, params


def add_eq8_eq9_columns(
    observed_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    cfg: LegacyComparisonConfig,
    params: LegacyComparisonParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = observed_df.copy()
    tru = truth_df.copy()

    x_cols = sorted([column for column in obs.columns if column.startswith("X")], key=lambda value: int(value[1:]))
    X = obs[x_cols].to_numpy()
    Z = obs["Z"].to_numpy()
    V = obs["V"].to_numpy()

    tildeZ = Z - X @ params.b_z
    tildeV = V - X @ params.b_v

    az, av = float(cfg.az), float(cfg.av)
    sz2, sv2 = float(cfg.sigma_z) ** 2, float(cfg.sigma_v) ** 2
    denom = (az**2) * sv2 + (av**2) * sz2 + sz2 * sv2
    mu_post = (az * sv2 * tildeZ + av * sz2 * tildeV) / denom
    var_post = (sz2 * sv2) / denom

    k = float(cfg.k_t)
    lam = float(cfg.lam_t)
    tau = float(cfg.tau_log_hr)
    beta_u = float(cfg.beta_u_in_t)
    G = math.gamma(1.0 + 1.0 / k)
    xb = X @ params.beta_t

    obs["tildeZ"] = tildeZ
    obs["tildeV"] = tildeV
    obs["mu_U_post"] = mu_post
    obs["var_U_post"] = var_post
    obs["CATE_XZV_eq9"] = (
        lam
        * G
        * np.exp(-(1.0 / k) * xb - (beta_u / k) * mu_post + 0.5 * (beta_u**2) * var_post / (k**2))
        * (np.exp(-tau / k) - 1.0)
    )

    U = tru["U"].to_numpy()
    tru["CATE_XU_eq7"] = lam * G * np.exp(-(1.0 / k) * (xb + beta_u * U)) * (np.exp(-tau / k) - 1.0)
    tru["ITE_T1_minus_T0"] = tru["T1"].to_numpy() - tru["T0"].to_numpy()

    return obs, tru
