"""Single-file reference implementation of the non-censored strict baseline.

This file intentionally avoids repo-internal imports so the baseline can be
read and run in isolation. It reproduces the current default non-censored
strict/EconML baseline path:

    - final input: [X, W, Z]
    - learner: EconML CausalForestDML
    - no custom bridge residualization
    - internal nuisance learning handled by EconML

Public API mirrors the current project style:

    model.fit_components(X, A, Y, Z, W)
    tau_hat = model.effect_from_components(X, W, Z)
"""

from __future__ import annotations

import numpy as np
from econml.dml import CausalForestDML


def _ensure_2d(array):
    """Return a 2D float array."""

    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _stack_final_features(*arrays):
    """Concatenate the raw feature blocks used by the strict baseline."""

    parts = [_ensure_2d(np.asarray(arr, dtype=float)) for arr in arrays]
    return np.hstack(parts)


class BaseNoncensoredModel:
    """Single-file version of the non-censored strict/EconML baseline."""

    def __init__(
        self,
        *,
        model_y="auto",
        model_t="auto",
        cv=2,
        discrete_treatment=True,
        criterion="het",
        random_state=42,
        **kwargs,
    ):
        self._model_y = model_y
        self._model_t = model_t
        self._cv = cv
        self._discrete_treatment = bool(discrete_treatment)
        self._criterion = criterion
        self._random_state = random_state
        self._extra_kwargs = dict(kwargs)
        self._model = None

    def fit_components(self, X, A, Y, Z, W):
        """Fit the strict baseline on raw [X, W, Z] features."""

        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_full = _stack_final_features(x, raw_w, raw_z)

        self._model = CausalForestDML(
            model_y=self._model_y,
            model_t=self._model_t,
            cv=self._cv,
            discrete_treatment=self._discrete_treatment,
            criterion=self._criterion,
            random_state=self._random_state,
            **self._extra_kwargs,
        )
        self._model.fit(
            Y=np.asarray(Y, dtype=float).ravel(),
            T=np.asarray(A, dtype=float).ravel(),
            X=x_full,
        )
        return self

    def effect_from_components(self, X, W, Z):
        """Predict CATE on raw [X, W, Z] features."""

        if self._model is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")

        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_full = _stack_final_features(x, raw_w, raw_z)
        return self._model.effect(x_full)


# Familiar alias so callers can use the existing project-style model name.
StrictEconmlXWZNCCausalForest = BaseNoncensoredModel


__all__ = ["BaseNoncensoredModel", "StrictEconmlXWZNCCausalForest"]
