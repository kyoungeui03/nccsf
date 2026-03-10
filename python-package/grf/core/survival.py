from __future__ import annotations

import numpy as np


def expected_survival(S_hat: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    S_hat = np.asarray(S_hat, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    grid_diff = np.diff(np.concatenate(([0.0], y_grid, [float(np.max(y_grid))])))
    return np.c_[np.ones(S_hat.shape[0]), S_hat] @ grid_diff


def compute_psi(
    S_hat: np.ndarray,
    C_hat: np.ndarray,
    C_Y_hat: np.ndarray,
    Y_hat: np.ndarray,
    W_centered: np.ndarray,
    D: np.ndarray,
    fY: np.ndarray,
    Y_index: np.ndarray,
    Y_grid: np.ndarray,
    target: str,
    horizon: float,
) -> dict[str, np.ndarray]:
    S_hat = np.asarray(S_hat, dtype=float)
    C_hat = np.asarray(C_hat, dtype=float)
    C_Y_hat = np.asarray(C_Y_hat, dtype=float)
    Y_hat = np.asarray(Y_hat, dtype=float)
    W_centered = np.asarray(W_centered, dtype=float)
    D = np.asarray(D, dtype=float)
    fY = np.asarray(fY, dtype=float)
    Y_index = np.asarray(Y_index, dtype=int)
    Y_grid = np.asarray(Y_grid, dtype=float)

    if target == "RMST":
        y_diff = np.diff(np.concatenate(([0.0], Y_grid)))
        q_hat = np.full_like(S_hat, np.nan, dtype=float)
        dot_products = S_hat[:, :-1] * y_diff[1:]
        q_hat[:, 0] = np.sum(dot_products, axis=1)
        for idx in range(1, q_hat.shape[1] - 1):
            q_hat[:, idx] = q_hat[:, idx - 1] - dot_products[:, idx - 1]
        q_hat = q_hat / S_hat
        q_hat = q_hat + Y_grid
        q_hat[:, -1] = float(np.max(Y_grid))
    else:
        horizon_index = np.searchsorted(Y_grid, horizon, side="right")
        if horizon_index == 0:
            raise ValueError("horizon cannot be before the first event.")
        q_hat = (S_hat[:, [horizon_index - 1]] / S_hat)
        q_hat[:, horizon_index - 1 :] = 1.0

    q_Y_hat = q_hat[np.arange(len(Y_index)), Y_index - 1]
    numerator_one = (D * (fY - Y_hat) + (1.0 - D) * (q_Y_hat - Y_hat)) * W_centered / C_Y_hat

    log_surv_c = -np.log(np.c_[np.ones(C_hat.shape[0]), C_hat])
    dlambda_c_hat = log_surv_c[:, 1:] - log_surv_c[:, :-1]
    integrand = dlambda_c_hat / C_hat * (q_hat - Y_hat[:, None])

    numerator_two = np.zeros(len(Y_index), dtype=float)
    for sample_index, yi_index in enumerate(Y_index):
        numerator_two[sample_index] = np.sum(integrand[sample_index, :yi_index]) * W_centered[sample_index]

    numerator = numerator_one - numerator_two
    denominator = W_centered**2
    return {
        "numerator": numerator,
        "denominator": denominator,
        "C_Y_hat": C_Y_hat,
    }
