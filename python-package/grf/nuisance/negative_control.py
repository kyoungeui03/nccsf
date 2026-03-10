from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import apply_rmst_truncation, compute_grf_orthogonal_scores, compute_risk_set_expectations
from .event_survival import EventSurvivalModel
from .proxies import OutcomeProxyModel, TreatmentProxyModel, compute_ipcw_outcome
from .survival import CensoringModel


@dataclass
class EstimatedNCPseudoResponses:
    gamma: np.ndarray
    h: np.ndarray
    treatment_residual: np.ndarray
    q_hat: np.ndarray
    h0_hat: np.ndarray
    h1_hat: np.ndarray
    m_hat: np.ndarray
    y_star: np.ndarray
    delta_star: np.ndarray
    failure_times_c: np.ndarray
    surv_c: np.ndarray
    hazard_c: np.ndarray
    failure_times_e: np.ndarray
    surv_e: np.ndarray


def generate_estimated_nc_pseudo_responses(
    X: np.ndarray,
    W_proxy: np.ndarray | None,
    Z_proxy: np.ndarray | None,
    A: np.ndarray,
    Y: np.ndarray,
    delta: np.ndarray,
    *,
    horizon: float | None,
    n_jobs: int,
) -> EstimatedNCPseudoResponses:
    L = np.inf if horizon is None else float(horizon)
    Y_star, delta_star = apply_rmst_truncation(Y, delta, L)

    to_stack = [X, A.reshape(-1, 1)]
    if W_proxy is not None:
        to_stack.append(W_proxy)
    if Z_proxy is not None:
        to_stack.append(Z_proxy)
    V = np.hstack(tuple(to_stack))

    n_samples = X.shape[0]

    censoring_model = CensoringModel(n_jobs=n_jobs)
    censoring_model.fit(V, Y_star, delta_star)
    surv_C_matrix, hazard_C_matrix, fail_times_C = censoring_model.predict_surv_and_hazard(V)

    S_c_Y = np.zeros(n_samples)
    for i in range(n_samples):
        idx = np.searchsorted(fail_times_C, Y_star[i], side="right") - 1
        idx = max(0, idx)
        S_c_Y[i] = surv_C_matrix[i, idx]

    event_model = EventSurvivalModel(n_jobs=n_jobs)
    event_model.fit(V, Y_star, delta_star)
    surv_E_matrix, fail_times_E = event_model.predict_survival(V)

    q_model = TreatmentProxyModel()
    q_model.fit(Z_proxy, X, A)
    q_preds = q_model.predict_proba(Z_proxy, X)

    Y_ipcw = compute_ipcw_outcome(Y_star, delta_star, S_c_Y)
    h_model = OutcomeProxyModel()
    h_model.fit(W_proxy, X, A, Y_ipcw)
    h_0, h_1 = h_model.predict(W_proxy, X, A=None)

    m_val = q_preds * h_1 + (1 - q_preds) * h_0
    D_residual = A - q_preds

    E_T_given_gt_t = compute_risk_set_expectations(surv_E_matrix, fail_times_E, L)

    K_gamma = np.zeros_like(surv_C_matrix)
    K_H_matrix = np.zeros_like(surv_C_matrix)

    for i in range(n_samples):
        E_T_interp = np.interp(fail_times_C, fail_times_E, E_T_given_gt_t[i, :])
        K_gamma[i, :] = D_residual[i] * (E_T_interp - m_val[i])
        K_H_matrix[i, :] = D_residual[i] ** 2

    Gamma_i, H_i = compute_grf_orthogonal_scores(
        Y_star,
        A,
        D_residual,
        m_val,
        delta_star,
        surv_C_matrix,
        hazard_C_matrix,
        K_gamma,
        K_H_matrix,
        fail_times_C,
    )

    return EstimatedNCPseudoResponses(
        gamma=Gamma_i,
        h=H_i,
        treatment_residual=D_residual,
        q_hat=q_preds,
        h0_hat=h_0,
        h1_hat=h_1,
        m_hat=m_val,
        y_star=Y_star,
        delta_star=delta_star,
        failure_times_c=fail_times_C,
        surv_c=surv_C_matrix,
        hazard_c=hazard_C_matrix,
        failure_times_e=fail_times_E,
        surv_e=surv_E_matrix,
    )
