from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import make_feature_columns


def default_mtry(num_features: int) -> int:
    return int(min(np.ceil(np.sqrt(num_features) + 20), num_features))


def find_interval(values: np.ndarray, sorted_grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_grid, values, side="right")


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def observation_weights(num_rows: int) -> np.ndarray:
    return np.full(num_rows, 1.0 / num_rows, dtype=float)


def build_train_frame(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    D: np.ndarray,
    feature_names: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = feature_names or make_feature_columns(X.shape[1])
    frame = pd.DataFrame(X, columns=feature_columns)
    frame["Y"] = Y
    frame["W"] = W
    frame["D"] = D
    return frame, feature_columns
