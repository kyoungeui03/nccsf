from __future__ import annotations

import numpy as np

from ..backends import NativeCausalSurvivalForest, NativeRegressionForest, NativeSurvivalForest
from ..core import (
    CausalSurvivalForest,
    build_train_frame,
    compute_psi,
    default_mtry,
    expected_survival,
    find_interval,
    validate_binary,
    validate_num_threads,
    validate_observations,
    validate_x,
)


def causal_survival_forest(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    D: np.ndarray,
    W_hat: float | np.ndarray | None = None,
    target: str = "RMST",
    horizon: float | None = None,
    failure_times: np.ndarray | None = None,
    num_trees: int = 2000,
    num_threads: int | None = None,
    seed: int = 42,
    feature_names: list[str] | None = None,
) -> CausalSurvivalForest:
    if target not in {"RMST", "survival.probability"}:
        raise ValueError("target must be one of {'RMST', 'survival.probability'}.")
    if horizon is None:
        raise ValueError("The horizon argument defining the estimand is required.")

    X = validate_x(X, allow_na=True)
    Y = validate_observations(Y, X, "Y")
    W = validate_observations(W, X, "W")
    D = validate_binary(validate_observations(D, X, "D"), "D").astype(float)
    num_threads = validate_num_threads(num_threads)

    if np.any(Y < 0):
        raise ValueError("The event times must be non-negative.")
    if np.sum(D) == 0:
        raise ValueError("All observations are censored.")

    train_frame, feature_columns = build_train_frame(X, Y, W, D, feature_names)

    mtry = default_mtry(X.shape[1])
    nuisance_trees = int(max(50, min(num_trees / 4, 500)))
    w_hat_trees = int(max(50, num_trees / 4))

    Y_for_forest = Y.astype(float).copy()
    D_for_forest = D.astype(float).copy()
    if target == "RMST":
        truncated = Y_for_forest >= horizon
        D_for_forest[truncated] = 1.0
        Y_for_forest[truncated] = float(horizon)
        fY = Y_for_forest.copy()
    else:
        fY = (Y > horizon).astype(float)

    if failure_times is None:
        Y_grid = np.sort(np.unique(Y_for_forest))
    else:
        Y_grid = np.asarray(failure_times, dtype=float)
        if np.min(Y_for_forest) < np.min(Y_grid):
            raise ValueError("If provided, failure_times should start on or before min(Y).")
    if Y_grid.shape[0] <= 2:
        raise ValueError("The number of distinct event times should be more than 2.")
    if horizon < float(np.min(Y_grid)):
        raise ValueError("horizon cannot be before the first event.")

    if W_hat is None:
        forest_w = NativeRegressionForest.fit(
            X,
            W,
            num_trees=w_hat_trees,
            sample_fraction=0.5,
            mtry=mtry,
            min_node_size=5,
            honesty=True,
            honesty_fraction=0.5,
            honesty_prune_leaves=True,
            alpha=0.05,
            imbalance_penalty=0.0,
            ci_group_size=1,
            compute_oob_predictions=True,
            num_threads=num_threads,
            seed=seed,
        )
        try:
            W_hat = forest_w.predict()
        finally:
            forest_w.close()
    elif np.isscalar(W_hat):
        W_hat = np.full(X.shape[0], float(W_hat), dtype=float)
    else:
        W_hat = validate_observations(W_hat, X, "W_hat")

    W_hat = np.asarray(W_hat, dtype=float)
    W_centered = W - W_hat

    nuisance_handles: list[NativeSurvivalForest] = []
    sf_y: NativeSurvivalForest | None = None
    try:
        X_with_w = np.column_stack([X, W])
        sf_survival = NativeSurvivalForest.fit(
            X_with_w,
            Y_for_forest,
            D_for_forest,
            failure_times=failure_times,
            num_trees=nuisance_trees,
            sample_fraction=0.5,
            mtry=mtry,
            min_node_size=15,
            honesty=True,
            honesty_fraction=0.5,
            honesty_prune_leaves=True,
            alpha=0.05,
            prediction_type=1,
            fast_logrank=False,
            compute_oob_predictions=True,
            num_threads=num_threads,
            seed=seed,
        )
        nuisance_handles.append(sf_survival)

        binary_w = np.all(np.isin(W, [0.0, 1.0]))
        if binary_w:
            X_w1 = np.column_stack([X, np.ones(X.shape[0], dtype=float)])
            X_w0 = np.column_stack([X, np.zeros(X.shape[0], dtype=float)])
            S1_hat = sf_survival.predict_oob(X_w1)
            S0_hat = sf_survival.predict_oob(X_w0)
            if target == "RMST":
                Y_hat = W_hat * expected_survival(S1_hat, sf_survival.failure_times) + (
                    1.0 - W_hat
                ) * expected_survival(S0_hat, sf_survival.failure_times)
            else:
                horizon_s_index = find_interval(np.array([horizon]), sf_survival.failure_times)[0]
                if horizon_s_index == 0:
                    Y_hat = np.ones(X.shape[0], dtype=float)
                else:
                    Y_hat = W_hat * S1_hat[:, horizon_s_index - 1] + (1.0 - W_hat) * S0_hat[:, horizon_s_index - 1]
        else:
            sf_y = NativeSurvivalForest.fit(
                X,
                Y_for_forest,
                D_for_forest,
                failure_times=failure_times,
                num_trees=nuisance_trees,
                sample_fraction=0.5,
                mtry=mtry,
                min_node_size=15,
                honesty=True,
                honesty_fraction=0.5,
                honesty_prune_leaves=True,
                alpha=0.05,
                prediction_type=1,
                fast_logrank=False,
                compute_oob_predictions=True,
                num_threads=num_threads,
                seed=seed,
            )
            nuisance_handles.append(sf_y)
            S_Y_hat = sf_y.predict_oob()
            if target == "RMST":
                Y_hat = expected_survival(S_Y_hat, sf_y.failure_times)
            else:
                horizon_s_index = find_interval(np.array([horizon]), sf_y.failure_times)[0]
                if horizon_s_index == 0:
                    Y_hat = np.ones(X.shape[0], dtype=float)
                else:
                    Y_hat = S_Y_hat[:, horizon_s_index - 1]

        S_hat = sf_survival.predict_oob(failure_times=Y_grid)

        sf_censor = NativeSurvivalForest.fit(
            X_with_w,
            Y_for_forest,
            1.0 - D_for_forest,
            failure_times=failure_times,
            num_trees=nuisance_trees,
            sample_fraction=0.5,
            mtry=mtry,
            min_node_size=15,
            honesty=True,
            honesty_fraction=0.5,
            honesty_prune_leaves=True,
            alpha=0.05,
            prediction_type=1,
            fast_logrank=False,
            compute_oob_predictions=True,
            num_threads=num_threads,
            seed=seed,
        )
        nuisance_handles.append(sf_censor)
        C_hat = sf_censor.predict_oob(failure_times=Y_grid)

        Y_for_psi = Y_for_forest.copy()
        D_for_psi = D_for_forest.copy()
        if target == "survival.probability":
            truncated = Y_for_psi > horizon
            D_for_psi[truncated] = 1.0
            Y_for_psi[truncated] = float(horizon)

        Y_index = find_interval(Y_for_psi, Y_grid)
        C_Y_hat = C_hat[np.arange(Y_index.shape[0]), Y_index - 1]

        psi = compute_psi(
            S_hat=S_hat,
            C_hat=C_hat,
            C_Y_hat=C_Y_hat,
            Y_hat=Y_hat,
            W_centered=W_centered,
            D=D_for_psi,
            fY=fY,
            Y_index=Y_index,
            Y_grid=Y_grid,
            target=target,
            horizon=float(horizon),
        )

        model = NativeCausalSurvivalForest.fit(
            X,
            W_centered,
            psi["numerator"],
            psi["denominator"],
            D_for_psi,
            num_trees=int(num_trees),
            sample_fraction=0.5,
            mtry=mtry,
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
        oob_predictions = model.predict()
    finally:
        for handle in nuisance_handles:
            handle.close()

    return CausalSurvivalForest(
        feature_columns=feature_columns,
        target=target,
        horizon=float(horizon),
        num_trees=int(num_trees),
        num_threads=num_threads,
        seed=int(seed),
        train_frame=train_frame,
        train_x=X,
        W_hat=W_hat,
        Y_hat=Y_hat,
        psi=psi,
        model=model,
        oob_predictions=oob_predictions,
    )
