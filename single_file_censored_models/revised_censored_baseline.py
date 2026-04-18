"""Single-file reference implementation of the Revised Censored Baseline.

This model is intended to be the closest observed-data baseline to the current
censored Final Model while removing PCI-style bridge construction entirely.

Design
------
The revised baseline keeps the same broad training backbone that matters for a
fair censored comparison:

    - explicit orthogonal residual construction (y_res, a_res)
    - marginal censoring correction with IPCW
    - arm-specific event-survival nuisances
    - final EconML CausalForestDML learner

What it removes relative to the Final Model:

    - no PCI bridge nuisances q / h1 / h0 / m
    - no proxy duplication in nuisance features
    - no outcome / propensity percentile clipping
    - final learner sees only raw [X, W, Z]

Instead of a bridge treatment nuisance q(X, Z), this model uses an ordinary
observed-data propensity model e(X, W, Z). Likewise, the mean nuisance is built
from arm-specific survival predictions:

    mu_hat(x) = e_hat(x) * mu1_hat(x) + (1 - e_hat(x)) * mu0_hat(x)

where mu1_hat / mu0_hat are derived from the treated/control Cox survival
models for the requested estimand.

Only external libraries are imported. All project-local helpers are inlined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from econml._ortho_learner import _OrthoLearner
from econml.dml import CausalForestDML
from econml.utilities import filter_none_kwargs
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

try:  # pragma: no cover
    from .gpu_backends import make_xgb_classifier
except ImportError:  # pragma: no cover
    from single_file_censored_models.gpu_backends import make_xgb_classifier  # type: ignore


def _ensure_2d(array):
    """Convert a 1D vector into a 2D column matrix."""

    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _stack_observed_features(x, w, z):
    """Build the raw observed feature stack `[X, W, Z]`."""

    return np.hstack(
        [
            _ensure_2d(x).astype(float),
            _ensure_2d(w).astype(float),
            _ensure_2d(z).astype(float),
        ]
    )


def _maybe_clip_percentiles(values, clip_percentiles):
    """Optionally clip values to empirical percentile bounds.

    The revised baseline defaults to `None`, meaning no residual clipping. The
    helper still exists so the model can be stress-tested with optional
    stabilization.
    """

    values = np.asarray(values, dtype=float)
    if clip_percentiles is None:
        return values
    lo, hi = np.percentile(values, clip_percentiles)
    return np.clip(values, lo, hi)


def _prepare_target_inputs(y_time, delta, *, target, horizon):
    """Build the target-specific time/event representation.

    Returns a dictionary whose keys are used throughout the code:
    - `nuisance_time`, `nuisance_delta`: time/event pair used to fit nuisance
      models
    - `eval_time`, `eval_delta`: time/event pair used inside the IPCW score
    - `f_y`: target-specific transformed outcome
    - `grid_time`: time grid source for Cox/survival summaries

    For finite-horizon RMST, the grid is extended to include the requested
    horizon so the Cox-based q-hat summaries target RMST up to that horizon
    rather than stopping at the largest observed follow-up time.
    """

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
        grid_time = np.concatenate([nuisance_time, np.asarray([float(horizon)], dtype=float)])

    return {
        "nuisance_time": nuisance_time,
        "nuisance_delta": nuisance_delta,
        "eval_time": eval_time,
        "eval_delta": eval_delta,
        "f_y": f_y,
        "grid_time": grid_time,
    }


def make_propensity_model(
    kind="logit",
    *,
    random_state=42,
    n_estimators=300,
    min_samples_leaf=20,
    poly_degree=2,
):
    """Create the observed-data treatment model `e(X, W, Z)`."""

    if kind == "poly2":
        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
                ("logit", LogisticRegression(max_iter=10000)),
            ]
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    if kind == "xgb":
        return make_xgb_classifier(
            random_state=random_state,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=1,
            device="cpu",
        )
    if kind == "xgb_gpu":
        return make_xgb_classifier(
            random_state=random_state,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=1,
            device="cuda",
        )
    return LogisticRegression(max_iter=10000)


def _fit_kaplan_meier_censoring(time, event):
    """Fit a marginal censoring survival curve with Kaplan-Meier."""

    censor_indicator = 1 - np.asarray(event).astype(int)
    kmf = KaplanMeierFitter()
    kmf.fit(np.asarray(time, dtype=float), event_observed=censor_indicator)
    return (
        np.asarray(kmf.survival_function_.index, dtype=float),
        np.asarray(kmf.survival_function_["KM_estimate"], dtype=float),
    )


def _fit_nelson_aalen_censoring(time, event):
    """Fit a marginal censoring survival curve via Nelson-Aalen."""

    censor_indicator = 1 - np.asarray(event).astype(int)
    naf = NelsonAalenFitter()
    naf.fit(np.asarray(time, dtype=float), event_observed=censor_indicator)
    times = np.asarray(naf.cumulative_hazard_.index, dtype=float)
    cum_hazard = np.asarray(naf.cumulative_hazard_["NA_estimate"], dtype=float)
    surv = np.exp(-cum_hazard)
    return times, surv


def _fit_censoring_survival(time, event, estimator="nelson-aalen"):
    """Dispatch to the requested marginal censoring estimator."""

    if estimator in {"aalen", "nelson-aalen"}:
        return _fit_nelson_aalen_censoring(time, event)
    if estimator in {"kaplan-meier", "km"}:
        return _fit_kaplan_meier_censoring(time, event)
    raise ValueError(
        "This single-file revised baseline supports only marginal censoring "
        "estimators {'nelson-aalen', 'kaplan-meier'}."
    )


def _fit_censoring_model(time, event, estimator="nelson-aalen"):
    """Wrap the marginal censoring fit in a dictionary interface."""

    times, surv = _fit_censoring_survival(time, event, estimator=estimator)
    return {
        "kind": "marginal",
        "times": times,
        "surv": surv,
    }


def _fit_conditional_censoring_model(time, event, features, *, estimator="cox"):
    """Fit conditional censoring `S_C(t | X, W, Z, A)` when requested."""

    if estimator == "cox":
        cox, col_names, keep_mask = _fit_event_cox(
            time,
            1.0 - np.asarray(event, dtype=float),
            features,
        )
        return {
            "kind": "cox",
            "model": cox,
            "col_names": col_names,
            "keep_mask": keep_mask,
        }
    return _fit_censoring_model(time, event, estimator=estimator)


def _evaluate_sc(values, km_times, km_surv, clip_min=0.01):
    """Evaluate a stepwise survival curve at arbitrary time points."""

    idx = np.searchsorted(km_times, np.asarray(values, dtype=float), side="right") - 1
    idx = np.clip(idx, 0, len(km_surv) - 1)
    return np.clip(km_surv[idx], clip_min, 1.0)


def _predict_censoring_survival_at_values(censor_model, values, *, clip_min=0.01):
    """Evaluate marginal censoring survival at one time per observation."""

    if censor_model["kind"] != "marginal":
        raise ValueError("Only marginal censoring models are supported in this file.")
    return _evaluate_sc(values, censor_model["times"], censor_model["surv"], clip_min=clip_min)


def _predict_censoring_survival_on_grid(censor_model, t_grid, n_rows, *, clip_min=0.01):
    """Evaluate marginal censoring survival on a shared time grid."""

    if censor_model["kind"] != "marginal":
        raise ValueError("Only marginal censoring models are supported in this file.")
    surv = _evaluate_sc(np.asarray(t_grid, dtype=float), censor_model["times"], censor_model["surv"], clip_min=clip_min)
    return np.broadcast_to(surv[None, :], (int(n_rows), len(t_grid)))


def _predict_conditional_censoring_survival_at_values(censor_model, features, values, *, clip_min=0.01):
    """Evaluate conditional censoring survival at one time per observation."""

    values = np.asarray(values, dtype=float)
    if censor_model["kind"] == "marginal":
        return _evaluate_sc(values, censor_model["times"], censor_model["surv"], clip_min=clip_min)

    eval_times = np.sort(np.unique(values))
    surv = _predict_s_on_grid(
        censor_model["model"],
        censor_model["col_names"],
        np.asarray(features, dtype=float),
        eval_times,
        censor_model["keep_mask"],
    )
    idx = np.searchsorted(eval_times, values, side="right") - 1
    idx = np.clip(idx, 0, len(eval_times) - 1)
    return np.clip(surv[np.arange(len(values)), idx], clip_min, 1.0)


def _predict_conditional_censoring_survival_on_grid(censor_model, features, t_grid, *, clip_min=0.01):
    """Evaluate conditional censoring survival on a shared time grid."""

    t_grid = np.asarray(t_grid, dtype=float)
    if censor_model["kind"] == "marginal":
        surv = _evaluate_sc(t_grid, censor_model["times"], censor_model["surv"], clip_min=clip_min)
        return np.broadcast_to(surv[None, :], (len(features), len(t_grid)))

    surv = _predict_s_on_grid(
        censor_model["model"],
        censor_model["col_names"],
        np.asarray(features, dtype=float),
        t_grid,
        censor_model["keep_mask"],
    )
    return np.clip(surv, clip_min, 1.0)


def _fit_event_cox(y_time, delta, features, penalizer=0.01):
    """Fit a Cox model on a feature block.

    The revised baseline uses this helper for the treated event model, the
    control event model, and the conditional censoring model.
    """

    features = np.asarray(features, dtype=float)
    variance = np.nanvar(features, axis=0)
    keep_mask = variance > 1e-10
    if not np.any(keep_mask):
        keep_mask = np.ones(features.shape[1], dtype=bool)

    filtered = features[:, keep_mask]
    col_names = [f"cxf{j}" for j in range(filtered.shape[1])]
    train_df = pd.DataFrame(filtered, columns=col_names)
    train_df["_duration"] = np.asarray(y_time, dtype=float)
    train_df["_event"] = np.asarray(delta, dtype=float)

    last_error = None
    for penalizer_try in (penalizer, 0.05, 0.1, 0.5, 1.0, 5.0):
        try:
            cox = CoxPHFitter(penalizer=penalizer_try)
            cox.fit(train_df, duration_col="_duration", event_col="_event")
            return cox, col_names, keep_mask
        except Exception as exc:
            last_error = exc
    raise last_error


def _predict_s_on_grid(cox, col_names, features, t_grid, keep_mask):
    """Evaluate a fitted Cox model on the supplied time grid."""

    pred_df = pd.DataFrame(np.asarray(features, dtype=float)[:, keep_mask], columns=col_names)
    surv = cox.predict_survival_function(pred_df, times=np.asarray(t_grid, dtype=float))
    return np.clip(surv.values.T, 1e-10, 1.0)


def _compute_q_from_s(s_hat, t_grid):
    """Convert RMST survival curves into the continuation term `q_hat(t)`."""

    s_hat = np.asarray(s_hat, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    n, g = s_hat.shape
    if g < 2:
        return np.full((n, g), t_grid[0] if g == 1 else 0.0)

    dt = np.diff(np.concatenate([[0.0], t_grid]))
    dot_products = s_hat[:, :-1] * dt[1:]

    cum = np.zeros((n, g), dtype=float)
    cum[:, 0] = dot_products.sum(axis=1)
    for k in range(1, g - 1):
        cum[:, k] = cum[:, k - 1] - dot_products[:, k - 1]

    q_hat = t_grid[None, :] + cum / np.maximum(s_hat, 1e-10)
    q_hat[:, -1] = t_grid[-1]
    return q_hat


def _compute_survival_probability_q_from_s(s_hat, t_grid, horizon):
    """Convert survival curves into the finite-horizon continuation term."""

    t_grid = np.asarray(t_grid, dtype=float).ravel()
    horizon_index = np.searchsorted(t_grid, float(horizon), side="right")
    if horizon_index == 0:
        raise ValueError("horizon cannot be before the first event.")
    q_hat = s_hat[:, [horizon_index - 1]] / np.maximum(s_hat, 1e-10)
    q_hat[:, horizon_index - 1 :] = 1.0
    return q_hat


def _compute_survival_based_yhat(s_hat_1, s_hat_0, e_hat, t_grid, *, target, horizon):
    """Build the revised baseline mean nuisance.

    - For RMST: integrate each arm-specific survival curve.
    - For survival probability: read the arm-specific survival probability at
      the requested horizon.
    - Then average the two arm-specific predictions using the observed-data
      propensity `e_hat`.
    """

    e_hat = np.asarray(e_hat, dtype=float).ravel()
    t_grid = np.asarray(t_grid, dtype=float).ravel()

    if target == "RMST":
        interval_widths = np.diff(np.concatenate([[0.0], t_grid]))
        y_hat_1 = np.c_[np.ones(s_hat_1.shape[0]), s_hat_1[:, :-1]] @ interval_widths
        y_hat_0 = np.c_[np.ones(s_hat_0.shape[0]), s_hat_0[:, :-1]] @ interval_widths
    else:
        horizon_index = np.searchsorted(t_grid, float(horizon), side="right")
        if horizon_index == 0:
            y_hat_1 = np.ones(s_hat_1.shape[0], dtype=float)
            y_hat_0 = np.ones(s_hat_0.shape[0], dtype=float)
        else:
            y_hat_1 = s_hat_1[:, horizon_index - 1]
            y_hat_0 = s_hat_0[:, horizon_index - 1]

    return e_hat * y_hat_1 + (1.0 - e_hat) * y_hat_0


def _compute_ipcw_3term_y_res_from_survival(
    y_time,
    delta,
    m_pred,
    q_hat,
    t_grid,
    sc_at_y,
    sc_grid,
    *,
    clip_percentiles,
):
    """Compute the RMST orthogonal residual with IPCW correction."""

    n = len(y_time)
    g = len(t_grid)
    sc_grid = np.asarray(sc_grid, dtype=float)
    if sc_grid.ndim == 1:
        sc_grid = np.broadcast_to(sc_grid[None, :], (n, g))

    y_idx = np.searchsorted(t_grid, y_time, side="right") - 1
    y_idx = np.clip(y_idx, 0, g - 1)
    q_at_y = q_hat[np.arange(n), y_idx]

    numerator = delta * y_time + (1.0 - delta) * q_at_y
    term1 = numerator / np.maximum(sc_at_y, 1e-10)

    log_sc = -np.log(np.maximum(sc_grid, 1e-10))
    d_lambda_c = np.diff(np.concatenate([np.zeros((n, 1)), log_sc], axis=1), axis=1)
    grid_weight = d_lambda_c / np.maximum(sc_grid, 1e-10)
    integrand = grid_weight * q_hat
    mask = np.arange(g)[None, :] <= y_idx[:, None]
    term2 = (integrand * mask).sum(axis=1)

    y_res = (term1 - term2) - m_pred
    return _maybe_clip_percentiles(y_res, clip_percentiles)


def _compute_target_ipcw_3term_y_res_from_survival(
    f_y,
    y_time,
    delta,
    m_pred,
    q_hat,
    t_grid,
    sc_at_y,
    sc_grid,
    *,
    clip_percentiles,
):
    """Compute the finite-horizon orthogonal residual with IPCW correction."""

    n = len(y_time)
    g = len(t_grid)
    sc_grid = np.asarray(sc_grid, dtype=float)
    if sc_grid.ndim == 1:
        sc_grid = np.broadcast_to(sc_grid[None, :], (n, g))

    y_idx = np.searchsorted(t_grid, y_time, side="right") - 1
    y_idx = np.clip(y_idx, 0, g - 1)
    q_at_y = q_hat[np.arange(n), y_idx]

    term1 = (delta * (f_y - m_pred) + (1.0 - delta) * (q_at_y - m_pred)) / np.maximum(sc_at_y, 1e-10)

    log_sc = -np.log(np.maximum(sc_grid, 1e-10))
    d_lambda_c = np.diff(np.concatenate([np.zeros((n, 1)), log_sc], axis=1), axis=1)
    grid_weight = d_lambda_c / np.maximum(sc_grid, 1e-10)
    integrand = grid_weight * (q_hat - m_pred[:, None])
    mask = np.arange(g)[None, :] <= y_idx[:, None]
    term2 = (integrand * mask).sum(axis=1)

    y_res = term1 - term2
    return _maybe_clip_percentiles(y_res, clip_percentiles)


class _RevisedBaselineNuisance:
    """Observed-data nuisance layer for the revised censored baseline."""

    max_grid = 500

    def __init__(
        self,
        *,
        propensity_model,
        target,
        horizon,
        censoring_estimator,
        event_survival_estimator="cox",
        y_res_clip_percentiles=None,
        censoring_clip_min=0.01,
    ):
        self._propensity_model_template = propensity_model
        self._target = target
        self._horizon = horizon
        self._censoring_estimator = censoring_estimator
        self._event_survival_estimator = event_survival_estimator
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._censoring_clip_min = float(censoring_clip_min)

        self._propensity_model = None
        self._censor_model = None
        self._event_cox_1 = None
        self._event_cox_0 = None
        self._cox_col_names_1 = None
        self._cox_col_names_0 = None
        self._cox_keep_mask_1 = None
        self._cox_keep_mask_0 = None
        self._t_grid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_propensity_model"] = None
        state["_censor_model"] = None
        state["_event_cox_1"] = None
        state["_event_cox_0"] = None
        state["_cox_col_names_1"] = None
        state["_cox_col_names_0"] = None
        state["_cox_keep_mask_1"] = None
        state["_cox_keep_mask_0"] = None
        state["_t_grid"] = None
        return state

    @staticmethod
    def _unpack_y(y):
        y = np.asarray(y, dtype=float)
        if y.ndim == 2 and y.shape[1] >= 2:
            return y[:, 0], y[:, 1]
        raise ValueError("Y must be a 2-column array [time, event].")

    def _fit_event_survival_models(self, nuisance_time, nuisance_delta, x_full, a):
        if self._event_survival_estimator != "cox":
            raise ValueError("This single-file revised baseline supports event_survival_estimator='cox' only.")

        treated_mask = a == 1
        control_mask = a == 0
        if treated_mask.sum() <= 10 or control_mask.sum() <= 10:
            raise ValueError("Each treatment arm must have more than 10 samples per fold for this model.")

        self._event_cox_1, self._cox_col_names_1, self._cox_keep_mask_1 = _fit_event_cox(
            nuisance_time[treated_mask],
            nuisance_delta[treated_mask],
            x_full[treated_mask],
        )
        self._event_cox_0, self._cox_col_names_0, self._cox_keep_mask_0 = _fit_event_cox(
            nuisance_time[control_mask],
            nuisance_delta[control_mask],
            x_full[control_mask],
        )

    def _predict_event_survival_curves(self, x_full):
        return (
            _predict_s_on_grid(
                self._event_cox_1,
                self._cox_col_names_1,
                x_full,
                self._t_grid,
                self._cox_keep_mask_1,
            ),
            _predict_s_on_grid(
                self._event_cox_0,
                self._cox_col_names_0,
                x_full,
                self._t_grid,
                self._cox_keep_mask_0,
            ),
        )

    def train(self, is_selecting, folds, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        """Fit nuisance models on one training fold.

        Variable meanings:
        - `x_full = [X, W, Z]`: observed feature stack
        - `a`: treatment indicator
        - `y_time`: observed follow-up time
        - `delta`: event indicator
        - `censor_features = [X, W, Z, A]`: only used when conditional
          censoring is requested
        - `self._t_grid`: shared time grid used by Cox-based summaries
        """

        del is_selecting, folds, W, Z, groups

        y_time, delta = self._unpack_y(Y)
        a = np.asarray(T, dtype=float).ravel()
        x_full = _ensure_2d(X).astype(float)
        censor_features = np.column_stack([x_full, a.reshape(-1, 1)])

        target_inputs = _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )

        all_times = np.sort(np.unique(target_inputs["grid_time"]))
        if len(all_times) > self.max_grid:
            idx = np.linspace(0, len(all_times) - 1, self.max_grid, dtype=int)
            all_times = all_times[idx]
        self._t_grid = all_times

        if self._censoring_estimator == "cox":
            self._censor_model = _fit_conditional_censoring_model(
                target_inputs["nuisance_time"],
                target_inputs["nuisance_delta"],
                censor_features,
                estimator=self._censoring_estimator,
            )
        else:
            self._censor_model = _fit_censoring_model(
                target_inputs["nuisance_time"],
                target_inputs["nuisance_delta"],
                estimator=self._censoring_estimator,
            )

        self._propensity_model = clone(self._propensity_model_template)
        self._propensity_model.fit(x_full, a, **filter_none_kwargs(sample_weight=sample_weight))

        self._fit_event_survival_models(
            target_inputs["nuisance_time"],
            target_inputs["nuisance_delta"],
            x_full,
            a,
        )
        return self

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        """Return the orthogonal residual pair `(y_res, a_res)`.

        This is the key difference from the strict baseline. The revised model
        explicitly residualizes both treatment and outcome before the final
        causal forest is fit.
        """

        del W, Z, sample_weight, groups

        y_time, delta = self._unpack_y(Y)
        a = np.asarray(T, dtype=float).ravel()
        x_full = _ensure_2d(X).astype(float)

        target_inputs = _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )

        # `e_pred` is the observed-data propensity estimate P(A=1 | X, W, Z).
        e_pred = self._propensity_model.predict_proba(x_full)[:, 1]
        censor_features = np.column_stack([x_full, a.reshape(-1, 1)])

        # `s_hat_1` and `s_hat_0` are the arm-specific event survival curves.
        s_hat_1, s_hat_0 = self._predict_event_survival_curves(x_full)

        # `m_pred` is the mean nuisance implied by the arm-specific survival
        # predictions and the observed-data propensity.
        m_pred = _compute_survival_based_yhat(
            s_hat_1,
            s_hat_0,
            e_pred,
            self._t_grid,
            target=self._target,
            horizon=self._horizon,
        )

        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, self._t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, self._t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, self._t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, self._t_grid)

        q_hat = np.where((a == 1)[:, None], q_hat_1, q_hat_0)
        if self._censoring_estimator == "cox":
            sc_grid = _predict_conditional_censoring_survival_on_grid(
                self._censor_model,
                censor_features,
                self._t_grid,
                clip_min=self._censoring_clip_min,
            )
        else:
            sc_grid = _predict_censoring_survival_on_grid(
                self._censor_model,
                self._t_grid,
                len(x_full),
                clip_min=self._censoring_clip_min,
            )

        if self._target == "RMST" and self._horizon is None:
            if self._censoring_estimator == "cox":
                sc_at_eval = _predict_conditional_censoring_survival_at_values(
                    self._censor_model,
                    censor_features,
                    y_time,
                    clip_min=self._censoring_clip_min,
                )
            else:
                sc_at_eval = _predict_censoring_survival_at_values(
                    self._censor_model,
                    y_time,
                    clip_min=self._censoring_clip_min,
                )
            y_res = _compute_ipcw_3term_y_res_from_survival(
                y_time,
                delta,
                m_pred,
                q_hat,
                self._t_grid,
                sc_at_eval,
                sc_grid,
                clip_percentiles=self._y_res_clip_percentiles,
            )
        else:
            if self._censoring_estimator == "cox":
                sc_at_eval = _predict_conditional_censoring_survival_at_values(
                    self._censor_model,
                    censor_features,
                    target_inputs["eval_time"],
                    clip_min=self._censoring_clip_min,
                )
            else:
                sc_at_eval = _predict_censoring_survival_at_values(
                    self._censor_model,
                    target_inputs["eval_time"],
                    clip_min=self._censoring_clip_min,
                )
            y_res = _compute_target_ipcw_3term_y_res_from_survival(
                target_inputs["f_y"],
                target_inputs["eval_time"],
                target_inputs["eval_delta"],
                m_pred,
                q_hat,
                self._t_grid,
                sc_at_eval,
                sc_grid,
                clip_percentiles=self._y_res_clip_percentiles,
            )

        # Treatment residual used by the final causal forest.
        a_res = (a - e_pred).reshape(-1, 1)
        return y_res, a_res


class _RevisedBaselineCensoredDML(CausalForestDML):
    """CausalForestDML subclass for the revised censored baseline."""

    def __init__(
        self,
        *,
        target="RMST",
        horizon=None,
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=42,
        propensity_kind="logit",
        propensity_trees=300,
        propensity_min_samples_leaf=20,
        propensity_poly_degree=2,
        censoring_estimator="nelson-aalen",
        event_survival_estimator="cox",
        y_res_clip_percentiles=None,
        censoring_clip_min=0.01,
    ):
        self._propensity_model_template = make_propensity_model(
            propensity_kind,
            random_state=random_state,
            n_estimators=propensity_trees,
            min_samples_leaf=propensity_min_samples_leaf,
            poly_degree=propensity_poly_degree,
        )
        self._target = target
        self._horizon = horizon
        self._censoring_estimator = censoring_estimator
        self._event_survival_estimator = event_survival_estimator
        self._custom_y_res_clip_percentiles = y_res_clip_percentiles
        self._censoring_clip_min = float(censoring_clip_min)

        super().__init__(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            cv=cv,
            random_state=random_state,
            discrete_treatment=True,
            criterion="het",
        )

    def _gen_ortho_learner_model_nuisance(self):
        """Tell EconML which custom nuisance object to cross-fit."""

        return _RevisedBaselineNuisance(
            propensity_model=self._propensity_model_template,
            target=self._target,
            horizon=self._horizon,
            censoring_estimator=self._censoring_estimator,
            event_survival_estimator=self._event_survival_estimator,
            y_res_clip_percentiles=self._custom_y_res_clip_percentiles,
            censoring_clip_min=self._censoring_clip_min,
        )

    def fit_survival(self, X, A, time, event, **kwargs):
        """Pack `(time, event)` into EconML's expected outcome container."""

        x_full = np.asarray(X, dtype=float)
        y = np.asarray(time, dtype=float).ravel()
        delta = np.asarray(event, dtype=float).ravel()
        y_packed = np.column_stack([y, delta])
        return _OrthoLearner.fit(self, y_packed, A, X=x_full, **kwargs)


class RevisedBaselineCensoredSurvivalForest:
    """Single-file revised censored baseline with explicit residualization."""

    def __init__(
        self,
        *,
        target="RMST",
        horizon=None,
        cv=5,
        random_state=42,
        propensity_kind="logit",
        propensity_trees=300,
        propensity_min_samples_leaf=20,
        propensity_poly_degree=2,
        y_res_clip_percentiles=None,
        n_estimators=200,
        min_samples_leaf=20,
        censoring_estimator="nelson-aalen",
        event_survival_estimator="cox",
        censoring_clip_min=0.01,
    ):
        if event_survival_estimator != "cox":
            raise ValueError("This single-file revised baseline supports event_survival_estimator='cox' only.")

        self._target = target
        self._horizon = horizon
        self._cv = int(cv)
        self._random_state = int(random_state)
        self._propensity_kind = propensity_kind
        self._propensity_trees = int(propensity_trees)
        self._propensity_min_samples_leaf = int(propensity_min_samples_leaf)
        self._propensity_poly_degree = int(propensity_poly_degree)
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._censoring_estimator = censoring_estimator
        self._event_survival_estimator = event_survival_estimator
        self._censoring_clip_min = float(censoring_clip_min)

        self._model = None

    def fit_components(self, X, A, time, event, Z, W):
        """Fit the public revised baseline model.

        Public notation:
        - `X`: baseline covariates
        - `W`, `Z`: observed proxy blocks, treated here as ordinary observed
          covariates rather than bridge inputs
        - `A`: treatment indicator
        - `time`, `event`: observed censored survival outcome
        """

        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)

        # The final forest only sees the raw observed feature stack.
        x_full = _stack_observed_features(x, raw_w, raw_z)

        self._model = _RevisedBaselineCensoredDML(
            target=self._target,
            horizon=self._horizon,
            n_estimators=self._n_estimators,
            min_samples_leaf=self._min_samples_leaf,
            cv=self._cv,
            random_state=self._random_state,
            propensity_kind=self._propensity_kind,
            propensity_trees=self._propensity_trees,
            propensity_min_samples_leaf=self._propensity_min_samples_leaf,
            propensity_poly_degree=self._propensity_poly_degree,
            censoring_estimator=self._censoring_estimator,
            event_survival_estimator=self._event_survival_estimator,
            y_res_clip_percentiles=self._y_res_clip_percentiles,
            censoring_clip_min=self._censoring_clip_min,
        )
        self._model.fit_survival(x_full, A, time, event)
        return self

    def effect_from_components(self, X, W, Z):
        """Predict CATE on the raw observed feature stack `[X, W, Z]`."""

        if self._model is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_full = _stack_observed_features(x, raw_w, raw_z)
        return self._model.effect(x_full)


RevisedCensoredBaseline = RevisedBaselineCensoredSurvivalForest


__all__ = [
    "RevisedBaselineCensoredSurvivalForest",
    "RevisedCensoredBaseline",
]
