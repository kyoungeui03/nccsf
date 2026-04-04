"""Single-file reference implementation of StrictEconmlXWZCensoredSurvivalForest.

This file keeps the current project behavior but removes project-local imports
so the full baseline can be read in one place. Only external libraries such as
NumPy, pandas, lifelines, and EconML are imported.
"""

from __future__ import annotations

import numpy as np
from econml.dml import CausalForestDML
from lifelines import KaplanMeierFitter, NelsonAalenFitter


def _ensure_2d(array):
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _prepare_target_inputs(y_time, delta, *, target, horizon):
    """Apply the same horizon/target preprocessing as the project baseline."""

    y_time = np.asarray(y_time, dtype=float)
    delta = np.asarray(delta, dtype=float)

    if target not in {"RMST", "survival.probability"}:
        raise ValueError("target must be one of {'RMST', 'survival.probability'}.")

    if target == "survival.probability":
        if horizon is None:
            raise ValueError("horizon is required when target='survival.probability'.")
        nuisance_time = y_time
        nuisance_delta = delta
        eval_time = np.minimum(y_time, float(horizon))
        eval_delta = delta.copy()
        eval_delta[y_time > float(horizon)] = 1.0
        f_y = (y_time > float(horizon)).astype(float)
        grid_time = y_time
    elif horizon is None:
        nuisance_time = y_time
        nuisance_delta = delta
        eval_time = y_time
        eval_delta = delta
        f_y = y_time
        grid_time = y_time
    else:
        nuisance_time = np.minimum(y_time, float(horizon))
        nuisance_delta = delta.copy()
        nuisance_delta[y_time >= float(horizon)] = 1.0
        eval_time = nuisance_time
        eval_delta = nuisance_delta
        f_y = nuisance_time
        grid_time = nuisance_time

    return {
        "nuisance_time": nuisance_time,
        "nuisance_delta": nuisance_delta,
        "eval_time": eval_time,
        "eval_delta": eval_delta,
        "f_y": f_y,
        "grid_time": grid_time,
    }


def _fit_kaplan_meier_censoring(time, event):
    censor_indicator = 1 - np.asarray(event).astype(int)
    kmf = KaplanMeierFitter()
    kmf.fit(np.asarray(time, dtype=float), event_observed=censor_indicator)
    return (
        np.asarray(kmf.survival_function_.index, dtype=float),
        np.asarray(kmf.survival_function_["KM_estimate"], dtype=float),
    )


def _fit_nelson_aalen_censoring(time, event):
    censor_indicator = 1 - np.asarray(event).astype(int)
    naf = NelsonAalenFitter()
    naf.fit(np.asarray(time, dtype=float), event_observed=censor_indicator)
    times = np.asarray(naf.cumulative_hazard_.index, dtype=float)
    cum_hazard = np.asarray(naf.cumulative_hazard_["NA_estimate"], dtype=float)
    surv = np.exp(-cum_hazard)
    return times, surv


def _fit_censoring_survival(time, event, estimator="nelson-aalen"):
    if estimator in {"aalen", "nelson-aalen"}:
        return _fit_nelson_aalen_censoring(time, event)
    if estimator in {"kaplan-meier", "km"}:
        return _fit_kaplan_meier_censoring(time, event)
    raise ValueError(
        "This single-file strict baseline supports only marginal censoring "
        "estimators {'nelson-aalen', 'kaplan-meier'}."
    )


def _fit_censoring_model(time, event, estimator="nelson-aalen"):
    times, surv = _fit_censoring_survival(time, event, estimator=estimator)
    return {
        "kind": "marginal",
        "times": times,
        "surv": surv,
    }


def _evaluate_sc(values, km_times, km_surv, clip_min=0.01):
    idx = np.searchsorted(km_times, np.asarray(values, dtype=float), side="right") - 1
    idx = np.clip(idx, 0, len(km_surv) - 1)
    return np.clip(km_surv[idx], clip_min, 1.0)


def _predict_censoring_survival_at_values(censor_model, values, *, clip_min=0.01):
    if censor_model["kind"] != "marginal":
        raise ValueError("Only marginal censoring models are supported in this file.")
    return _evaluate_sc(values, censor_model["times"], censor_model["surv"], clip_min=clip_min)


def _compute_target_pseudo_outcome_from_sc(
    *,
    y_time,
    horizon,
    target,
    nuisance_time,
    nuisance_delta,
    sc_at_eval,
):
    """Build the IPCW pseudo-outcome used by the strict baseline."""

    if target == "survival.probability":
        return (np.asarray(y_time, dtype=float) > float(horizon)).astype(float) / np.maximum(sc_at_eval, 1e-10)
    return np.asarray(nuisance_time, dtype=float) * np.asarray(nuisance_delta, dtype=float) / np.maximum(sc_at_eval, 1e-10)


class StrictEconmlXWZCensoredSurvivalForest:
    """Strict censored EconML baseline on raw [X, W, Z] with IPCW pseudo-outcomes only."""

    def __init__(
        self,
        *,
        target="RMST",
        horizon=None,
        n_estimators=200,
        min_samples_leaf=20,
        cv=2,
        model_y="auto",
        model_t="auto",
        discrete_treatment=True,
        criterion="het",
        random_state=42,
        censoring_estimator="nelson-aalen",
        **kwargs,
    ):
        self._target = target
        self._horizon = horizon
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._cv = int(cv)
        self._model_y = model_y
        self._model_t = model_t
        self._discrete_treatment = bool(discrete_treatment)
        self._criterion = criterion
        self._random_state = int(random_state)
        self._censoring_estimator = censoring_estimator
        self._extra_kwargs = dict(kwargs)
        self._model = None

    @staticmethod
    def stack_final_features(*arrays):
        parts = [_ensure_2d(np.asarray(arr, dtype=float)) for arr in arrays]
        return np.hstack(parts)

    def fit_components(self, X, A, time, event, Z, W):
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_full = self.stack_final_features(x, raw_w, raw_z)
        y_time = np.asarray(time, dtype=float).ravel()
        delta = np.asarray(event, dtype=float).ravel()
        a = np.asarray(A, dtype=float).ravel()

        target_inputs = _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )
        censor_model = _fit_censoring_model(
            target_inputs["nuisance_time"],
            target_inputs["nuisance_delta"],
            estimator=self._censoring_estimator,
        )
        y_tilde_eval_time = (
            target_inputs["eval_time"]
            if self._target == "survival.probability"
            else target_inputs["nuisance_time"]
        )
        sc_at_eval = _predict_censoring_survival_at_values(
            censor_model,
            y_tilde_eval_time,
            clip_min=1e-10,
        )
        y_tilde = _compute_target_pseudo_outcome_from_sc(
            y_time=y_time,
            target=self._target,
            horizon=self._horizon,
            nuisance_time=target_inputs["nuisance_time"],
            nuisance_delta=target_inputs["nuisance_delta"],
            sc_at_eval=sc_at_eval,
        )

        self._model = CausalForestDML(
            model_y=self._model_y,
            model_t=self._model_t,
            cv=self._cv,
            discrete_treatment=self._discrete_treatment,
            criterion=self._criterion,
            n_estimators=self._n_estimators,
            min_samples_leaf=self._min_samples_leaf,
            random_state=self._random_state,
            **self._extra_kwargs,
        )
        self._model.fit(
            Y=np.asarray(y_tilde, dtype=float).ravel(),
            T=a,
            X=x_full,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._model is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_full = self.stack_final_features(x, raw_w, raw_z)
        return self._model.effect(x_full)


StrictCensoredBaseline = StrictEconmlXWZCensoredSurvivalForest


__all__ = [
    "StrictEconmlXWZCensoredSurvivalForest",
    "StrictCensoredBaseline",
]
