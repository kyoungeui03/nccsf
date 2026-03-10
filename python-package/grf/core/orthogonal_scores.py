from __future__ import annotations

import numpy as np


def apply_rmst_truncation(Y: np.ndarray, delta: np.ndarray, L: float | None) -> tuple[np.ndarray, np.ndarray]:
    if L is None or np.isinf(L):
        return Y.copy(), delta.copy()

    Y_star = np.minimum(Y, L)
    delta_star = delta.copy()
    delta_star[Y >= L] = 1
    return Y_star, delta_star


def compute_risk_set_expectations(S_e_matrix: np.ndarray, failure_times: np.ndarray, L: float | None) -> np.ndarray:
    if L is None or np.isinf(L):
        bounded_times = failure_times
        L_cap = np.inf
    else:
        bounded_times = np.minimum(failure_times, L)
        L_cap = L

    dt = np.diff(bounded_times)
    S_dt = S_e_matrix[:, :-1] * dt[np.newaxis, :]
    integral_from_k = np.cumsum(S_dt[:, ::-1], axis=1)[:, ::-1]

    integral_matrix = np.zeros_like(S_e_matrix)
    integral_matrix[:, :-1] = integral_from_k

    S_safe = np.maximum(S_e_matrix, 1e-10)
    expectation = bounded_times[np.newaxis, :] + (integral_matrix / S_safe)

    if not np.isinf(L_cap):
        expectation = np.minimum(expectation, L_cap)
        mask_past_L = bounded_times >= L_cap
        if np.any(mask_past_L):
            expectation[:, mask_past_L] = L_cap

    return expectation


def compute_grf_orthogonal_scores(
    Y: np.ndarray,
    A: np.ndarray,
    D: np.ndarray,
    m_val: np.ndarray,
    delta: np.ndarray,
    surv_prob_C: np.ndarray,
    hazard_jumps_C: np.ndarray,
    K_gamma: np.ndarray,
    K_H: np.ndarray,
    failure_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    del A
    n_samples = Y.shape[0]

    uncensored_gamma = D * (Y - m_val)
    uncensored_H = D**2

    idx_Y = np.searchsorted(failure_times, Y, side="right") - 1
    idx_Y = np.clip(idx_Y, 0, len(failure_times) - 1)

    S_c_Y = surv_prob_C[np.arange(n_samples), idx_Y]
    S_c_Y = np.maximum(S_c_Y, 1e-10)

    K_gamma_Y = K_gamma[np.arange(n_samples), idx_Y]
    K_H_Y = K_H[np.arange(n_samples), idx_Y]

    time_mask = np.arange(len(failure_times))[np.newaxis, :] <= idx_Y[:, np.newaxis]
    S_c_safe = np.maximum(surv_prob_C, 1e-10)

    integrand_gamma = np.where(time_mask, (hazard_jumps_C / S_c_safe) * K_gamma, 0.0)
    integrand_H = np.where(time_mask, (hazard_jumps_C / S_c_safe) * K_H, 0.0)

    integral_gamma = np.sum(integrand_gamma, axis=1)
    integral_H = np.sum(integrand_H, axis=1)

    term1_gamma = (delta / S_c_Y) * uncensored_gamma
    term1_H = (delta / S_c_Y) * uncensored_H

    term2_gamma = ((1 - delta) / S_c_Y) * K_gamma_Y
    term2_H = ((1 - delta) / S_c_Y) * K_H_Y

    Gamma_array = term1_gamma + term2_gamma - integral_gamma
    H_array = term1_H + term2_H - integral_H
    return Gamma_array, H_array
