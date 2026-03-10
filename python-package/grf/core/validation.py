from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def validate_x(X: np.ndarray | pd.DataFrame, allow_na: bool = False) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy(dtype=float)
    else:
        X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must have positive dimensions.")
    if not allow_na and np.isnan(X).any():
        raise ValueError("X contains missing values.")
    return X


def validate_observations(values: Iterable[float], X: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector.")
    if values.shape[0] != X.shape[0]:
        raise ValueError(f"{name} has incorrect length.")
    return values


def validate_binary(values: np.ndarray, name: str) -> np.ndarray:
    unique = np.unique(values)
    if not np.all(np.isin(unique, [0.0, 1.0])):
        raise ValueError(f"{name} can only contain 0/1 values.")
    return values.astype(int)


def validate_num_threads(num_threads: int | None) -> int:
    if num_threads is None:
        return 1
    num_threads = int(num_threads)
    if num_threads <= 0:
        raise ValueError("num_threads must be positive.")
    return num_threads


def validate_newdata(newdata: np.ndarray | pd.DataFrame, train_x: np.ndarray) -> np.ndarray:
    newdata = validate_x(newdata, allow_na=True)
    if newdata.shape[1] != train_x.shape[1]:
        raise ValueError("newdata has incorrect number of columns.")
    return newdata


def make_feature_columns(num_features: int) -> list[str]:
    return [f"X{i}" for i in range(num_features)]
