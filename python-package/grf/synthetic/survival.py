from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_intercept_for_prevalence(
    linpred_no_intercept: np.ndarray,
    target_prevalence: float,
    max_iter: int = 60,
) -> float:
    lo, hi = -20.0, 20.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = sigmoid(mid + linpred_no_intercept).mean()
        if p < target_prevalence:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def weibull_ph_time_paper(u01: np.ndarray, k: float, lam: float, eta: np.ndarray) -> np.ndarray:
    u01 = np.clip(u01, 1e-12, 1 - 1e-12)
    scale = lam * np.exp(-eta / k)
    return scale * (-np.log(u01)) ** (1.0 / k)


@dataclass
class SynthConfig:
    n: int = 200
    p_x: int = 5
    p_w: int = 1
    p_z: int = 1
    seed: int = 42
    a_prevalence: float = 0.5
    gamma_u_in_a: float = 1.0
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
    aZ: float = 1.0
    sigma_z: float = 0.8
    aW: float = 1.0
    sigma_w: float = 0.8
    linear_treatment: bool = True
    linear_outcome: bool = True


@dataclass
class SynthParams:
    b_z: np.ndarray
    b_w: np.ndarray
    beta_t: np.ndarray
    beta_squared: Optional[np.ndarray] = None
    beta_interact: Optional[np.ndarray] = None


def generate_synthetic_nc_cox(cfg: SynthConfig) -> tuple[pd.DataFrame, pd.DataFrame, SynthParams]:
    rng = np.random.default_rng(cfg.seed)
    n, p = cfg.n, cfg.p_x

    X = rng.normal(size=(n, p))
    U = rng.normal(size=n)

    b_z = rng.normal(scale=0.3, size=(p, cfg.p_z))
    b_w = rng.normal(scale=0.3, size=(p, cfg.p_w))

    Z = cfg.aZ * U[:, np.newaxis] + X @ b_z + rng.normal(scale=cfg.sigma_z, size=(n, cfg.p_z))
    W_nc = cfg.aW * U[:, np.newaxis] + X @ b_w + rng.normal(scale=cfg.sigma_w, size=(n, cfg.p_w))

    alpha = rng.normal(scale=0.5, size=p)
    if cfg.linear_treatment:
        linpred = X @ alpha + cfg.gamma_u_in_a * U
    else:
        x_nonlinear = np.column_stack(
            [sigmoid(X[:, i]) for i in range(min(3, p))]
            + [X[:, i] * X[:, (i + 1) % p] for i in range(min(2, p))]
        )
        alpha_nonlinear = rng.normal(scale=0.3, size=x_nonlinear.shape[1])
        linpred = x_nonlinear @ alpha_nonlinear + cfg.gamma_u_in_a * sigmoid(U)

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

    T_obs_for_calib = np.where(A == 1, T1, T0)
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

    T = np.where(A == 1, T1, T0)
    C = np.where(A == 1, C1, C0)
    time = np.minimum(T, C)
    event = (T <= C).astype(int)

    if cfg.admin_censor_time is not None:
        admin = float(cfg.admin_censor_time)
        cens_by_admin = admin < time
        time = np.where(cens_by_admin, admin, time)
        event = np.where(cens_by_admin, 0, event).astype(int)

    x_cols = {f"X{j}": X[:, j] for j in range(p)}
    if cfg.p_w == 1:
        w_cols = {"W": W_nc[:, 0]}
    else:
        w_cols = {f"W{j}": W_nc[:, j] for j in range(cfg.p_w)}
    if cfg.p_z == 1:
        z_cols = {"Z": Z[:, 0]}
    else:
        z_cols = {f"Z{j}": Z[:, j] for j in range(cfg.p_z)}
    observed_df = pd.DataFrame(
        {
            "time": time,
            "event": event,
            "A": A,
            **w_cols,
            **z_cols,
            **x_cols,
        }
    )

    truth_df = observed_df.copy()
    truth_df.insert(0, "U", U)
    truth_df["T0"] = T0
    truth_df["T1"] = T1
    truth_df["C0"] = C0
    truth_df["C1"] = C1
    truth_df["T"] = T
    truth_df["C"] = C
    truth_df["eta_t0"] = eta_t0
    truth_df["eta_t1"] = eta_t1
    truth_df.attrs["lam_c_used"] = lam_c_used

    params = SynthParams(
        b_z=b_z,
        b_w=b_w,
        beta_t=beta_t,
        beta_squared=beta_squared,
        beta_interact=beta_interact,
    )
    return observed_df, truth_df, params


def add_ground_truth_cate(
    observed_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    cfg: SynthConfig,
    params: SynthParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = observed_df.copy()
    tru = truth_df.copy()

    k = float(cfg.k_t)
    lam = float(cfg.lam_t)
    G = math.gamma(1.0 + 1.0 / k)

    tru["ITE_T1_minus_T0"] = tru["T1"].to_numpy() - tru["T0"].to_numpy()

    eta_t0 = tru["eta_t0"].to_numpy()
    eta_t1 = tru["eta_t1"].to_numpy()
    tru["CATE_XU_eq7"] = lam * G * (np.exp(-eta_t1 / k) - np.exp(-eta_t0 / k))

    return obs, tru
