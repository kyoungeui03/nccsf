from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backends import NativeCausalSurvivalForest
from ..core import (
    CausalSurvivalForest,
    apply_rmst_truncation,
    build_train_frame,
    compute_grf_orthogonal_scores,
    compute_risk_set_expectations,
    default_mtry,
    validate_binary,
    validate_num_threads,
    validate_observations,
    validate_x,
)
from ..nuisance import generate_estimated_nc_pseudo_responses


def _validate_optional_matrix(
    values: np.ndarray | pd.DataFrame | None,
    X: np.ndarray,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None

    if isinstance(values, pd.DataFrame):
        array = values.to_numpy(dtype=float)
    else:
        array = np.asarray(values, dtype=float)

    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    if array.shape[0] != X.shape[0]:
        raise ValueError(f"{name} has incorrect number of rows.")
    return array


def _validate_monotone_times(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector.")
    if values.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty.")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    return values


@dataclass
class OracleNCNuisanceInputs:
    q_hat: np.ndarray
    h0_hat: np.ndarray
    h1_hat: np.ndarray
    surv_c: np.ndarray
    hazard_c: np.ndarray
    failure_times_c: np.ndarray
    surv_e: np.ndarray
    failure_times_e: np.ndarray


def _validate_oracle_inputs(oracle: OracleNCNuisanceInputs, X: np.ndarray) -> OracleNCNuisanceInputs:
    n_samples = X.shape[0]

    q_hat = validate_observations(oracle.q_hat, X, "oracle.q_hat")
    h0_hat = validate_observations(oracle.h0_hat, X, "oracle.h0_hat")
    h1_hat = validate_observations(oracle.h1_hat, X, "oracle.h1_hat")

    surv_c = np.asarray(oracle.surv_c, dtype=float)
    hazard_c = np.asarray(oracle.hazard_c, dtype=float)
    surv_e = np.asarray(oracle.surv_e, dtype=float)
    if surv_c.ndim != 2 or surv_c.shape[0] != n_samples:
        raise ValueError("oracle.surv_c must have shape (n_samples, n_times_c).")
    if hazard_c.shape != surv_c.shape:
        raise ValueError("oracle.hazard_c must match oracle.surv_c.")
    if surv_e.ndim != 2 or surv_e.shape[0] != n_samples:
        raise ValueError("oracle.surv_e must have shape (n_samples, n_times_e).")

    failure_times_c = _validate_monotone_times(oracle.failure_times_c, "oracle.failure_times_c")
    failure_times_e = _validate_monotone_times(oracle.failure_times_e, "oracle.failure_times_e")

    if surv_c.shape[1] != failure_times_c.shape[0]:
        raise ValueError("oracle.surv_c column count must match oracle.failure_times_c.")
    if surv_e.shape[1] != failure_times_e.shape[0]:
        raise ValueError("oracle.surv_e column count must match oracle.failure_times_e.")

    return OracleNCNuisanceInputs(
        q_hat=q_hat,
        h0_hat=h0_hat,
        h1_hat=h1_hat,
        surv_c=surv_c,
        hazard_c=hazard_c,
        failure_times_c=failure_times_c,
        surv_e=surv_e,
        failure_times_e=failure_times_e,
    )


def _fit_nc_forest_from_scores(
    X: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    D: np.ndarray,
    q_hat: np.ndarray,
    y_hat: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    horizon: float | None,
    num_trees: int,
    num_threads: int,
    seed: int,
    feature_names: list[str] | None,
    psi_payload: dict[str, np.ndarray] | None = None,
) -> CausalSurvivalForest:
    train_frame, feature_columns = build_train_frame(X, Y, A, D, feature_names)
    denominator_safe = np.maximum(np.asarray(denominator, dtype=float), 1e-10)
    treatment_residual = np.asarray(A, dtype=float) - np.asarray(q_hat, dtype=float)

    model = NativeCausalSurvivalForest.fit(
        X,
        treatment_residual,
        np.asarray(numerator, dtype=float),
        denominator_safe,
        np.asarray(D, dtype=float),
        num_trees=int(num_trees),
        sample_fraction=0.5,
        mtry=default_mtry(X.shape[1]),
        min_node_size=5,
        honesty=True,
        honesty_fraction=0.5,
        honesty_prune_leaves=True,
        alpha=0.05,
        imbalance_penalty=0.0,
        stabilize_splits=True,
        ci_group_size=2,
        num_threads=num_threads,
        seed=int(seed),
    )
    oob_predictions = model.predict()

    psi = {
        "numerator": np.asarray(numerator, dtype=float),
        "denominator": denominator_safe,
        "raw_denominator": np.asarray(denominator, dtype=float),
        "treatment_residual": treatment_residual,
    }
    if psi_payload is not None:
        psi.update(psi_payload)

    return CausalSurvivalForest(
        feature_columns=feature_columns,
        target="RMST",
        horizon=float(np.inf if horizon is None else horizon),
        num_trees=int(num_trees),
        num_threads=int(num_threads),
        seed=int(seed),
        train_frame=train_frame,
        train_x=X,
        W_hat=np.asarray(q_hat, dtype=float),
        Y_hat=np.asarray(y_hat, dtype=float),
        psi=psi,
        model=model,
        oob_predictions=oob_predictions,
    )


def nc_causal_survival_forest(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray,
    A: np.ndarray,
    D: np.ndarray,
    W_proxy: np.ndarray | pd.DataFrame | None,
    Z_proxy: np.ndarray | pd.DataFrame | None,
    *,
    horizon: float | None = None,
    num_trees: int = 2000,
    num_threads: int | None = None,
    seed: int = 42,
    feature_names: list[str] | None = None,
) -> CausalSurvivalForest:
    X = validate_x(X, allow_na=True)
    Y = validate_observations(Y, X, "Y")
    A = validate_binary(validate_observations(A, X, "A"), "A").astype(float)
    D = validate_binary(validate_observations(D, X, "D"), "D").astype(float)
    W_proxy = _validate_optional_matrix(W_proxy, X, "W_proxy")
    Z_proxy = _validate_optional_matrix(Z_proxy, X, "Z_proxy")
    num_threads = validate_num_threads(num_threads)

    if np.any(Y < 0):
        raise ValueError("The event times must be non-negative.")
    if np.sum(D) == 0:
        raise ValueError("All observations are censored.")
    if horizon is not None and float(horizon) < 0:
        raise ValueError("horizon must be non-negative.")

    pseudo = generate_estimated_nc_pseudo_responses(
        X,
        W_proxy,
        Z_proxy,
        A,
        Y,
        D,
        horizon=horizon,
        n_jobs=num_threads,
    )

    return _fit_nc_forest_from_scores(
        X,
        pseudo.y_star,
        A,
        pseudo.delta_star,
        pseudo.q_hat,
        pseudo.m_hat,
        pseudo.gamma,
        pseudo.h,
        horizon=horizon,
        num_trees=num_trees,
        num_threads=num_threads,
        seed=seed,
        feature_names=feature_names,
        psi_payload={
            "q_hat": pseudo.q_hat,
            "h0_hat": pseudo.h0_hat,
            "h1_hat": pseudo.h1_hat,
            "m_hat": pseudo.m_hat,
            "y_star": pseudo.y_star,
            "delta_star": pseudo.delta_star,
            "failure_times_c": pseudo.failure_times_c,
            "surv_c": pseudo.surv_c,
            "hazard_c": pseudo.hazard_c,
            "failure_times_e": pseudo.failure_times_e,
            "surv_e": pseudo.surv_e,
        },
    )


def nc_oracle_causal_survival_forest(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray,
    A: np.ndarray,
    D: np.ndarray,
    oracle: OracleNCNuisanceInputs,
    *,
    horizon: float | None = None,
    num_trees: int = 2000,
    num_threads: int | None = None,
    seed: int = 42,
    feature_names: list[str] | None = None,
) -> CausalSurvivalForest:
    X = validate_x(X, allow_na=True)
    Y = validate_observations(Y, X, "Y")
    A = validate_binary(validate_observations(A, X, "A"), "A").astype(float)
    D = validate_binary(validate_observations(D, X, "D"), "D").astype(float)
    oracle = _validate_oracle_inputs(oracle, X)
    num_threads = validate_num_threads(num_threads)

    if np.any(Y < 0):
        raise ValueError("The event times must be non-negative.")
    if np.sum(D) == 0:
        raise ValueError("All observations are censored.")
    if horizon is not None and float(horizon) < 0:
        raise ValueError("horizon must be non-negative.")

    y_star, delta_star = apply_rmst_truncation(Y, D, np.inf if horizon is None else float(horizon))
    m_hat = oracle.q_hat * oracle.h1_hat + (1.0 - oracle.q_hat) * oracle.h0_hat
    treatment_residual = A - oracle.q_hat
    expectation = compute_risk_set_expectations(
        oracle.surv_e,
        oracle.failure_times_e,
        np.inf if horizon is None else float(horizon),
    )

    k_gamma = np.zeros_like(oracle.surv_c)
    k_h = np.zeros_like(oracle.surv_c)
    for index in range(X.shape[0]):
        event_expectation = np.interp(oracle.failure_times_c, oracle.failure_times_e, expectation[index, :])
        k_gamma[index, :] = treatment_residual[index] * (event_expectation - m_hat[index])
        k_h[index, :] = treatment_residual[index] ** 2

    gamma, h = compute_grf_orthogonal_scores(
        y_star,
        A,
        treatment_residual,
        m_hat,
        delta_star,
        oracle.surv_c,
        oracle.hazard_c,
        k_gamma,
        k_h,
        oracle.failure_times_c,
    )

    return _fit_nc_forest_from_scores(
        X,
        y_star,
        A,
        delta_star,
        oracle.q_hat,
        m_hat,
        gamma,
        h,
        horizon=horizon,
        num_trees=num_trees,
        num_threads=num_threads,
        seed=seed,
        feature_names=feature_names,
        psi_payload={
            "q_hat": oracle.q_hat,
            "h0_hat": oracle.h0_hat,
            "h1_hat": oracle.h1_hat,
            "m_hat": m_hat,
            "y_star": y_star,
            "delta_star": delta_star,
            "failure_times_c": oracle.failure_times_c,
            "surv_c": oracle.surv_c,
            "hazard_c": oracle.hazard_c,
            "failure_times_e": oracle.failure_times_e,
            "surv_e": oracle.surv_e,
        },
    )
