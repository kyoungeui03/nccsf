"""Single-file reference implementation of the current censored Final Model (full).

This file intentionally inlines the helper functions that are otherwise spread
across multiple project files. The goal is readability: someone should be able
to follow the full pipeline here without jumping between local modules.

Scope
-----
This file reproduces the *current default* censored final model used in the
benchmark pipeline:

    - final feature mode: "full"
    - raw final backbone: [X, W, Z]
    - bridge summaries passed to the final learner:
        [q_pred, h1_pred, h0_pred, m_pred, surv1_pred, surv0_pred, surv_diff_pred]
    - prediction nuisance mode: "full_refit"
    - event survival estimator: Cox
    - censoring estimator: Nelson-Aalen (default), with Kaplan-Meier also supported
    - nuisance feature mode: broad_dup
    - final learner: EconML CausalForestDML with a custom final-feature wrapper

The implementation below is deliberately focused on the default path that we
actually benchmark. It does not try to preserve every experimental ablation
branch from the original research files.

Notation
--------
Throughout this file we use the following names consistently:

    - `x`: baseline covariates X
    - `w`: outcome-side proxy block W
    - `z`: treatment-side proxy block Z
    - `a`: binary treatment A
    - `y_time`: observed follow-up time min(T, C)
    - `delta`: event indicator 1{T <= C}
    - `q_pred`: bridge-style treatment nuisance
    - `h1_pred`, `h0_pred`: arm-specific bridge regressions
    - `m_pred`: bridge mixture q*h1 + (1-q)*h0
    - `y_res`, `a_res`: orthogonal residuals fed to the final forest
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from econml._ortho_learner import _OrthoLearner
from econml.dml import CausalForestDML
from econml.utilities import filter_none_kwargs
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# ---------------------------------------------------------------------------
# Generic array helpers
# ---------------------------------------------------------------------------


def _ensure_2d(array):
    """Return a 2D float array.

    Many nuisance components accept both 1D and 2D inputs. Normalizing here
    keeps later code simpler and mirrors the current project behavior.
    """

    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _clip_quantile(values, q):
    """Winsorize values to the symmetric quantile range used by the model."""

    values = np.asarray(values, dtype=float)
    if q is None:
        return values
    lo = float(np.quantile(values, 1.0 - q))
    hi = float(np.quantile(values, q))
    return np.clip(values, lo, hi)


def _pairwise_products(left, right):
    """Compute all pairwise feature interactions between two matrices."""

    left = _ensure_2d(left)
    right = _ensure_2d(right)
    if left.shape[1] == 0 or right.shape[1] == 0:
        return np.empty((left.shape[0], 0), dtype=float)
    return (left[:, :, None] * right[:, None, :]).reshape(left.shape[0], -1)


# ---------------------------------------------------------------------------
# Target / horizon preprocessing
# ---------------------------------------------------------------------------


def _prepare_target_inputs(y_time, delta, *, target, horizon):
    """Apply the same target/horizon convention as the current project code.

    For finite-horizon RMST, observations past the horizon are truncated to the
    horizon and treated as event-observed at that truncation point. That is the
    convention used consistently across our current Final Model and benchmarks.
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
        grid_time = nuisance_time

    return {
        "nuisance_time": nuisance_time,
        "nuisance_delta": nuisance_delta,
        "eval_time": eval_time,
        "eval_delta": eval_delta,
        "f_y": f_y,
        "grid_time": grid_time,
    }


# ---------------------------------------------------------------------------
# Base ML models used inside the nuisance layer
# ---------------------------------------------------------------------------


def make_q_model(
    kind="logit",
    *,
    random_state=42,
    n_estimators=300,
    min_samples_leaf=20,
    poly_degree=2,
):
    """Create the treatment nuisance model q(X, Z)."""

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
    return LogisticRegression(max_iter=10000)


def make_h_model(
    kind="extra",
    *,
    random_state=42,
    n_estimators=600,
    min_samples_leaf=5,
    n_jobs=1,
):
    """Create the bridge regression model h(X, W)."""

    if kind == "extra":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )


# ---------------------------------------------------------------------------
# Censoring helpers
# ---------------------------------------------------------------------------


def _fit_kaplan_meier_censoring(time, event):
    """Fit marginal censoring survival using Kaplan-Meier."""

    censor_indicator = 1 - np.asarray(event).astype(int)
    kmf = KaplanMeierFitter()
    kmf.fit(np.asarray(time, dtype=float), event_observed=censor_indicator)
    return (
        np.asarray(kmf.survival_function_.index, dtype=float),
        np.asarray(kmf.survival_function_["KM_estimate"], dtype=float),
    )


def _fit_nelson_aalen_censoring(time, event):
    """Fit marginal censoring survival using Nelson-Aalen -> exp(-cumhaz)."""

    censor_indicator = 1 - np.asarray(event).astype(int)
    naf = NelsonAalenFitter()
    naf.fit(np.asarray(time, dtype=float), event_observed=censor_indicator)
    times = np.asarray(naf.cumulative_hazard_.index, dtype=float)
    cum_hazard = np.asarray(naf.cumulative_hazard_["NA_estimate"], dtype=float)
    surv = np.exp(-cum_hazard)
    return times, surv


def _fit_censoring_survival(time, event, estimator="nelson-aalen"):
    """Fit the marginal censoring survival curve used in the default path."""

    if estimator in {"aalen", "nelson-aalen"}:
        return _fit_nelson_aalen_censoring(time, event)
    if estimator in {"kaplan-meier", "km"}:
        return _fit_kaplan_meier_censoring(time, event)
    raise ValueError(
        "This single-file implementation supports only marginal censoring "
        "estimators {'nelson-aalen', 'kaplan-meier'}."
    )


def _fit_censoring_model(time, event, estimator="nelson-aalen"):
    """Wrap the censoring fit in the same dictionary-style interface as the project."""

    times, surv = _fit_censoring_survival(time, event, estimator=estimator)
    return {
        "kind": "marginal",
        "times": times,
        "surv": surv,
    }


def _evaluate_sc(values, km_times, km_surv, clip_min=0.01):
    """Evaluate a marginal survival curve at arbitrary values."""

    idx = np.searchsorted(km_times, np.asarray(values, dtype=float), side="right") - 1
    idx = np.clip(idx, 0, len(km_surv) - 1)
    return np.clip(km_surv[idx], clip_min, 1.0)


def _predict_censoring_survival_at_values(censor_model, values, *, clip_min=0.01):
    """Evaluate S_C(y) for each observation."""

    if censor_model["kind"] != "marginal":
        raise ValueError("Only marginal censoring models are supported in this file.")
    values = np.asarray(values, dtype=float)
    return _evaluate_sc(values, censor_model["times"], censor_model["surv"], clip_min=clip_min)


def _predict_censoring_survival_on_grid(censor_model, t_grid, n_rows, *, clip_min=0.01):
    """Evaluate S_C(t) on the common time grid for every observation."""

    if censor_model["kind"] != "marginal":
        raise ValueError("Only marginal censoring models are supported in this file.")
    surv = _evaluate_sc(np.asarray(t_grid, dtype=float), censor_model["times"], censor_model["surv"], clip_min=clip_min)
    return np.broadcast_to(surv[None, :], (int(n_rows), len(t_grid)))


# ---------------------------------------------------------------------------
# Event survival helpers (Cox only for the default Final Model path)
# ---------------------------------------------------------------------------


def _fit_event_cox(y_time, delta, features, penalizer=0.01):
    """Fit an arm-specific Cox model on survival features."""

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
    """Predict arm-specific survival curves on the shared time grid."""

    pred_df = pd.DataFrame(np.asarray(features, dtype=float)[:, keep_mask], columns=col_names)
    surv = cox.predict_survival_function(pred_df, times=np.asarray(t_grid, dtype=float))
    return np.clip(surv.values.T, 1e-10, 1.0)


# ---------------------------------------------------------------------------
# Survival math
# ---------------------------------------------------------------------------


def _compute_q_from_s(s_hat, t_grid):
    """Convert survival curves into the q-hat curve used by the orthogonal score.

    In the RMST setting, q-hat(t) is the remaining restricted mean beyond each
    grid time, normalized by the survival level at that time. This is exactly
    the same transformation used in the current project implementation.
    """

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
    """q-hat counterpart for survival.probability targets."""

    t_grid = np.asarray(t_grid, dtype=float).ravel()
    horizon_index = np.searchsorted(t_grid, float(horizon), side="right")
    if horizon_index == 0:
        raise ValueError("horizon cannot be before the first event.")
    q_hat = s_hat[:, [horizon_index - 1]] / np.maximum(s_hat, 1e-10)
    q_hat[:, horizon_index - 1 :] = 1.0
    return q_hat


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def _build_nuisance_features(raw_x, w, z, feature_mode):
    """Build branch-specific nuisance features.

    The current Final Model defaults to `broad_dup`, which mirrors the
    non-censored broad-dup logic:

      - start from [X, W, Z]
      - duplicate Z once more for q
      - duplicate W once more for h
      - duplicate both W and Z once more for the survival branch
    """

    raw_x = _ensure_2d(raw_x)
    w = _ensure_2d(w)
    z = _ensure_2d(z)
    base = np.column_stack([raw_x, w, z])

    if feature_mode == "broad_dup":
        q_features = np.column_stack([base, z])
        h_features = np.column_stack([base, w])
        surv_features = np.column_stack([base, w, z])
    elif feature_mode == "dup":
        q_features = np.column_stack([raw_x, z])
        h_features = np.column_stack([raw_x, w])
        surv_features = np.column_stack([raw_x, w, z])
    elif feature_mode == "interact":
        q_features = np.column_stack([raw_x, z])
        h_features = np.column_stack([raw_x, w])
        surv_features = np.column_stack([raw_x, w, z])
        extra = np.column_stack(
            [
                _pairwise_products(raw_x, w),
                _pairwise_products(raw_x, z),
                _pairwise_products(w, z),
            ]
        )
        q_features = np.column_stack([q_features, extra])
        h_features = np.column_stack([h_features, extra])
        surv_features = np.column_stack([surv_features, extra])
    else:
        raise ValueError(f"Unsupported nuisance_feature_mode: {feature_mode}")

    return q_features, h_features, surv_features


def _build_final_features_full(x, raw_w, raw_z, bridge):
    """Build the current default final-stage feature block.

    Current full features are exactly:

        [X, W, Z, q_pred, h1_pred, h0_pred, m_pred, surv1_pred, surv0_pred, surv_diff_pred]
    """

    return np.hstack(
        [
            _ensure_2d(x).astype(float),
            _ensure_2d(raw_w).astype(float),
            _ensure_2d(raw_z).astype(float),
            np.asarray(bridge["q_pred"], dtype=float).reshape(-1, 1),
            np.asarray(bridge["h1_pred"], dtype=float).reshape(-1, 1),
            np.asarray(bridge["h0_pred"], dtype=float).reshape(-1, 1),
            np.asarray(bridge["m_pred"], dtype=float).reshape(-1, 1),
            np.asarray(bridge["surv1_pred"], dtype=float).reshape(-1, 1),
            np.asarray(bridge["surv0_pred"], dtype=float).reshape(-1, 1),
            np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1),
        ]
    )


# ---------------------------------------------------------------------------
# Pseudo-outcome and residual calculations
# ---------------------------------------------------------------------------


def _compute_target_pseudo_outcome_from_sc(
    *,
    y_time,
    horizon,
    target,
    nuisance_time,
    nuisance_delta,
    sc_at_eval,
):
    """Create the IPCW pseudo-outcome used to fit the h nuisance models."""

    if target == "survival.probability":
        return (np.asarray(y_time, dtype=float) > float(horizon)).astype(float) / np.maximum(sc_at_eval, 1e-10)
    return np.asarray(nuisance_time, dtype=float) * np.asarray(nuisance_delta, dtype=float) / np.maximum(sc_at_eval, 1e-10)


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
    """Default RMST-without-horizon residual used by the orthogonal score."""

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
    lo, hi = np.percentile(y_res, clip_percentiles)
    return np.clip(y_res, lo, hi)


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
    """Finite-horizon residual used by the current censored Final Model."""

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
    lo, hi = np.percentile(y_res, clip_percentiles)
    return np.clip(y_res, lo, hi)


# ---------------------------------------------------------------------------
# Nuisance model
# ---------------------------------------------------------------------------


class _FinalCensoredNuisance:
    """All survival / censoring / bridge nuisance logic for the default model.

    This object is cloned by EconML across folds. It fits:

      1. a censoring survival model for IPCW
      2. a treatment model q(X, Z)
      3. bridge regressions h1(X, W), h0(X, W)
      4. arm-specific Cox survival models

    It then exposes:

      - predict(...): explicit (y_res, a_res)
      - predict_bridge_outputs(...): bridge/survival summaries for x_final
    """

    max_grid = 500

    def __init__(
        self,
        *,
        q_model,
        h_model,
        target,
        horizon,
        nuisance_feature_mode,
        censoring_estimator,
        q_clip,
        y_tilde_clip_quantile,
        y_res_clip_percentiles,
        event_survival_estimator="cox",
        m_pred_mode="bridge",
    ):
        self._q_model_template = q_model
        self._h_model_template = h_model
        self._target = target
        self._horizon = horizon
        self._nuisance_feature_mode = nuisance_feature_mode
        self._censoring_estimator = censoring_estimator
        self._q_clip = float(q_clip)
        self._y_tilde_clip_quantile = y_tilde_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._event_survival_estimator = event_survival_estimator
        self._m_pred_mode = m_pred_mode

        self._q_model = None
        self._h1_model = None
        self._h0_model = None
        self._censor_model = None
        self._event_cox_1 = None
        self._event_cox_0 = None
        self._cox_col_names_1 = None
        self._cox_col_names_0 = None
        self._cox_keep_mask_1 = None
        self._cox_keep_mask_0 = None
        self._t_grid = None

    def __getstate__(self):
        """Drop fitted state when EconML clones the nuisance object."""

        state = dict(self.__dict__)
        state["_q_model"] = None
        state["_h1_model"] = None
        state["_h0_model"] = None
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

    def _fit_event_survival_models(self, nuisance_time, nuisance_delta, surv_features, a):
        if self._event_survival_estimator != "cox":
            raise ValueError("This single-file implementation supports event_survival_estimator='cox' only.")

        treated_mask = a == 1
        control_mask = a == 0
        self._event_cox_1, self._cox_col_names_1, self._cox_keep_mask_1 = _fit_event_cox(
            nuisance_time[treated_mask],
            nuisance_delta[treated_mask],
            surv_features[treated_mask],
        )
        self._event_cox_0, self._cox_col_names_0, self._cox_keep_mask_0 = _fit_event_cox(
            nuisance_time[control_mask],
            nuisance_delta[control_mask],
            surv_features[control_mask],
        )

    def _predict_event_survival_curves(self, surv_features):
        return (
            _predict_s_on_grid(
                self._event_cox_1,
                self._cox_col_names_1,
                surv_features,
                self._t_grid,
                self._cox_keep_mask_1,
            ),
            _predict_s_on_grid(
                self._event_cox_0,
                self._cox_col_names_0,
                surv_features,
                self._t_grid,
                self._cox_keep_mask_0,
            ),
        )

    def _compute_m_pred(self, q_pred, h1_pred, h0_pred):
        if self._m_pred_mode != "bridge":
            raise ValueError("This single-file implementation supports m_pred_mode='bridge' only.")
        return q_pred * h1_pred + (1.0 - q_pred) * h0_pred

    def train(self, is_selecting, folds, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        """Fit all nuisance components on one fold's training slice."""

        del is_selecting, folds, groups  # Unused but kept for EconML compatibility.

        y_time, delta = self._unpack_y(Y)
        a = np.asarray(T, dtype=float).ravel()
        x = _ensure_2d(X).astype(float)
        w = _ensure_2d(W).astype(float)
        z = _ensure_2d(Z).astype(float)

        target_inputs = _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )
        xq, xh, surv_features = _build_nuisance_features(x, w, z, self._nuisance_feature_mode)

        all_times = np.sort(np.unique(target_inputs["grid_time"]))
        if len(all_times) > self.max_grid:
            idx = np.linspace(0, len(all_times) - 1, self.max_grid, dtype=int)
            all_times = all_times[idx]
        self._t_grid = all_times

        # Step 1: fit the censoring survival used for IPCW.
        self._censor_model = _fit_censoring_model(
            target_inputs["nuisance_time"],
            target_inputs["nuisance_delta"],
            estimator=self._censoring_estimator,
        )

        # Step 2: build the IPCW pseudo-outcome y_tilde used to fit h1 / h0.
        y_tilde_eval_time = (
            target_inputs["eval_time"]
            if self._target == "survival.probability"
            else target_inputs["nuisance_time"]
        )
        sc_for_y_tilde = _predict_censoring_survival_at_values(
            self._censor_model,
            y_tilde_eval_time,
        )
        y_tilde = _compute_target_pseudo_outcome_from_sc(
            y_time=y_time,
            target=self._target,
            horizon=self._horizon,
            nuisance_time=target_inputs["nuisance_time"],
            nuisance_delta=target_inputs["nuisance_delta"],
            sc_at_eval=sc_for_y_tilde,
        )
        y_tilde = _clip_quantile(y_tilde, self._y_tilde_clip_quantile)

        # Step 3: fit q(X, Z), h1(X, W), h0(X, W).
        self._q_model = clone(self._q_model_template)
        self._q_model.fit(xq, a, **filter_none_kwargs(sample_weight=sample_weight))

        treated_mask = a == 1
        control_mask = a == 0

        self._h1_model = clone(self._h_model_template)
        self._h0_model = clone(self._h_model_template)

        if treated_mask.sum() <= 10 or control_mask.sum() <= 10:
            raise ValueError("Each treatment arm must have more than 10 samples per fold for this model.")

        self._h1_model.fit(
            xh[treated_mask],
            y_tilde[treated_mask],
            **filter_none_kwargs(sample_weight=None if sample_weight is None else sample_weight[treated_mask]),
        )
        self._h0_model.fit(
            xh[control_mask],
            y_tilde[control_mask],
            **filter_none_kwargs(sample_weight=None if sample_weight is None else sample_weight[control_mask]),
        )

        # Step 4: fit arm-specific event survival curves.
        self._fit_event_survival_models(
            target_inputs["nuisance_time"],
            target_inputs["nuisance_delta"],
            surv_features,
            a,
        )
        return self

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        """Return the explicit orthogonalized residual pair (y_res, a_res)."""

        del sample_weight, groups  # Interface compatibility.

        y_time, delta = self._unpack_y(Y)
        a = np.asarray(T, dtype=float).ravel()
        x = _ensure_2d(X).astype(float)
        w = _ensure_2d(W).astype(float)
        z = _ensure_2d(Z).astype(float)

        target_inputs = _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )
        xq, xh, surv_features = _build_nuisance_features(x, w, z, self._nuisance_feature_mode)

        q_pred = self._q_model.predict_proba(xq)[:, 1]
        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        h1_pred = self._h1_model.predict(xh)
        h0_pred = self._h0_model.predict(xh)
        s_hat_1, s_hat_0 = self._predict_event_survival_curves(surv_features)
        m_pred = self._compute_m_pred(q_pred, h1_pred, h0_pred)

        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, self._t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, self._t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, self._t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, self._t_grid)

        q_hat = np.where((a == 1)[:, None], q_hat_1, q_hat_0)
        sc_grid = _predict_censoring_survival_on_grid(self._censor_model, self._t_grid, len(x))

        if self._target == "RMST" and self._horizon is None:
            sc_at_eval = _predict_censoring_survival_at_values(self._censor_model, y_time)
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
            sc_at_eval = _predict_censoring_survival_at_values(self._censor_model, target_inputs["eval_time"])
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

        a_res = (a - q_pred).reshape(-1, 1)
        return y_res, a_res

    def predict_bridge_outputs(self, X=None, W=None, Z=None):
        """Return the bridge/survival summaries needed to build x_final."""

        x = _ensure_2d(X).astype(float)
        w = _ensure_2d(W).astype(float)
        z = _ensure_2d(Z).astype(float)
        xq, xh, surv_features = _build_nuisance_features(x, w, z, self._nuisance_feature_mode)

        q_pred = self._q_model.predict_proba(xq)[:, 1]
        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        h1_pred = self._h1_model.predict(xh)
        h0_pred = self._h0_model.predict(xh)
        s_hat_1, s_hat_0 = self._predict_event_survival_curves(surv_features)
        m_pred = self._compute_m_pred(q_pred, h1_pred, h0_pred)

        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, self._t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, self._t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, self._t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, self._t_grid)

        surv1_pred = q_hat_1[:, 0]
        surv0_pred = q_hat_0[:, 0]
        return {
            "q_pred": q_pred,
            "h1_pred": h1_pred,
            "h0_pred": h0_pred,
            "m_pred": m_pred,
            "surv1_pred": surv1_pred,
            "surv0_pred": surv0_pred,
            "surv_diff_pred": surv1_pred - surv0_pred,
        }


class _BridgeOutputNuisance(_FinalCensoredNuisance):
    """Nuisance variant whose predict() returns everything needed for x_final."""

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_res, a_res = super().predict(Y, T, X=X, W=W, Z=Z, sample_weight=sample_weight, groups=groups)
        bridge = super().predict_bridge_outputs(X=X, W=W, Z=Z)
        return (
            y_res,
            a_res,
            bridge["q_pred"],
            bridge["h1_pred"],
            bridge["h0_pred"],
            bridge["m_pred"],
            bridge["surv1_pred"],
            bridge["surv0_pred"],
            bridge["surv_diff_pred"],
        )


# ---------------------------------------------------------------------------
# Custom final-model wrapper for CausalForestDML
# ---------------------------------------------------------------------------


class _BridgeFeatureFinalModel:
    """Translate nuisance outputs into x_final before calling EconML's final stage."""

    def __init__(self, base_model_final, *, raw_proxy_supplier=None):
        self._base_model_final = base_model_final
        self._raw_proxy_supplier = raw_proxy_supplier
        self._train_x_final = None

    def _transform(self, X, W, Z, nuisances):
        (
            y_res,
            a_res,
            q_pred,
            h1_pred,
            h0_pred,
            m_pred,
            surv1_pred,
            surv0_pred,
            surv_diff_pred,
        ) = nuisances

        bridge = {
            "q_pred": q_pred,
            "h1_pred": h1_pred,
            "h0_pred": h0_pred,
            "m_pred": m_pred,
            "surv1_pred": surv1_pred,
            "surv0_pred": surv0_pred,
            "surv_diff_pred": surv_diff_pred,
        }

        w_for_final = W
        z_for_final = Z
        if self._raw_proxy_supplier is not None:
            supplied = self._raw_proxy_supplier(X=X, W=W, Z=Z)
            if supplied is not None:
                w_for_final, z_for_final = supplied

        x_final = _build_final_features_full(X, w_for_final, z_for_final, bridge)
        return x_final, (y_res, a_res)

    def fit(
        self,
        Y,
        T,
        X=None,
        W=None,
        Z=None,
        nuisances=None,
        sample_weight=None,
        freq_weight=None,
        sample_var=None,
        groups=None,
    ):
        x_final, core_nuisances = self._transform(X, W, Z, nuisances)
        self._train_x_final = np.asarray(x_final, dtype=float).copy()
        self._base_model_final.fit(
            Y,
            T,
            X=x_final,
            W=W,
            Z=Z,
            nuisances=core_nuisances,
            sample_weight=sample_weight,
            freq_weight=freq_weight,
            sample_var=sample_var,
            groups=groups,
        )
        return self

    def score(
        self,
        Y,
        T,
        X=None,
        W=None,
        Z=None,
        nuisances=None,
        sample_weight=None,
        groups=None,
        scoring="mean_squared_error",
    ):
        x_final, core_nuisances = self._transform(X, W, Z, nuisances)
        return self._base_model_final.score(
            Y,
            T,
            X=x_final,
            W=W,
            Z=Z,
            nuisances=core_nuisances,
            sample_weight=sample_weight,
            groups=groups,
            scoring=scoring,
        )

    def predict(self, X=None):
        return self._base_model_final.predict(X)

    def training_x_final(self):
        if self._train_x_final is None:
            return None
        return np.asarray(self._train_x_final, dtype=float)


# ---------------------------------------------------------------------------
# Custom CausalForestDML subclass
# ---------------------------------------------------------------------------


class _SinglePassBridgeFeatureCensoredSurvivalForest(CausalForestDML):
    """CausalForestDML whose nuisance and final stages follow our Final Model."""

    def __init__(
        self,
        *,
        target="RMST",
        horizon=None,
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=42,
        q_kind="logit",
        q_trees=300,
        q_min_samples_leaf=20,
        q_poly_degree=2,
        q_clip=0.03,
        y_tilde_clip_quantile=0.98,
        y_res_clip_percentiles=(2.0, 98.0),
        h_kind="extra",
        h_n_estimators=600,
        h_min_samples_leaf=5,
        censoring_estimator="nelson-aalen",
        event_survival_estimator="cox",
        m_pred_mode="bridge",
        nuisance_feature_mode="broad_dup",
    ):
        self._q_model_template = make_q_model(
            q_kind,
            random_state=random_state,
            n_estimators=q_trees,
            min_samples_leaf=q_min_samples_leaf,
            poly_degree=q_poly_degree,
        )
        self._h_model_template = make_h_model(
            h_kind,
            random_state=random_state,
            n_estimators=h_n_estimators,
            min_samples_leaf=h_min_samples_leaf,
            n_jobs=1,
        )
        self._target = target
        self._horizon = horizon
        self._censoring_estimator = censoring_estimator
        self._event_survival_estimator = event_survival_estimator
        self._m_pred_mode = m_pred_mode
        self._nuisance_feature_mode = nuisance_feature_mode
        self._custom_q_clip = float(q_clip)
        self._custom_y_tilde_clip_quantile = y_tilde_clip_quantile
        self._custom_y_res_clip_percentiles = y_res_clip_percentiles
        self._raw_w_for_final = None
        self._raw_z_for_final = None

        super().__init__(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            cv=cv,
            random_state=random_state,
            discrete_treatment=True,
            criterion="het",
        )

    def _raw_proxy_for_final(self, *, X=None, W=None, Z=None):
        del W, Z
        if X is None:
            return None
        if self._raw_w_for_final is None or self._raw_z_for_final is None:
            return None
        if len(np.asarray(X, dtype=float)) != len(self._raw_w_for_final):
            return None
        return self._raw_w_for_final, self._raw_z_for_final

    def _gen_ortho_learner_model_nuisance(self):
        return _BridgeOutputNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            target=self._target,
            horizon=self._horizon,
            nuisance_feature_mode=self._nuisance_feature_mode,
            censoring_estimator=self._censoring_estimator,
            q_clip=self._custom_q_clip,
            y_tilde_clip_quantile=self._custom_y_tilde_clip_quantile,
            y_res_clip_percentiles=self._custom_y_res_clip_percentiles,
            event_survival_estimator=self._event_survival_estimator,
            m_pred_mode=self._m_pred_mode,
        )

    def _gen_ortho_learner_model_final(self):
        return _BridgeFeatureFinalModel(
            super()._gen_ortho_learner_model_final(),
            raw_proxy_supplier=self._raw_proxy_for_final,
        )

    def fit_survival(self, X, A, time, event, Z, W, **kwargs):
        """Pack `(time, event)` and delegate fitting to EconML's OrthoLearner."""

        x = np.asarray(X, dtype=float)
        y = np.asarray(time, dtype=float).ravel()
        delta = np.asarray(event, dtype=float).ravel()
        z = _ensure_2d(Z)
        w = _ensure_2d(W)
        y_packed = np.column_stack([y, delta])
        return _OrthoLearner.fit(self, y_packed, A, X=x, W=w, Z=z, **kwargs)

    def effect_on_final_features(self, x_final):
        """Evaluate the final causal forest on already-constructed `x_final`."""

        return np.asarray(self._ortho_learner_model_final.predict(x_final), dtype=float)

    def training_x_final(self):
        """Expose the cached training-time final feature matrix for debugging."""

        return self._ortho_learner_model_final.training_x_final()


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------


class FinalCensoredModel:
    """Single-file version of the current censored Final Model (full).

    Public API mirrors the current project style:

        model.fit_components(X, A, time, event, Z, W)
        tau_hat = model.effect_from_components(X, W, Z)
    """

    def __init__(
        self,
        *,
        target="RMST",
        horizon=None,
        cv=5,
        random_state=42,
        q_kind="logit",
        q_trees=300,
        q_min_samples_leaf=20,
        q_poly_degree=2,
        q_clip=0.03,
        h_kind="extra",
        h_n_estimators=600,
        h_min_samples_leaf=5,
        y_tilde_clip_quantile=0.98,
        y_res_clip_percentiles=(2.0, 98.0),
        n_estimators=200,
        min_samples_leaf=20,
        censoring_estimator="nelson-aalen",
        event_survival_estimator="cox",
        m_pred_mode="bridge",
        nuisance_feature_mode="broad_dup",
        prediction_nuisance_mode="full_refit",
        surv_scalar_mode="full",
        observed_only=False,
    ):
        if surv_scalar_mode != "full":
            raise ValueError("This single-file implementation reproduces only surv_scalar_mode='full'.")
        if prediction_nuisance_mode != "full_refit":
            raise ValueError("This single-file implementation supports prediction_nuisance_mode='full_refit' only.")
        if event_survival_estimator != "cox":
            raise ValueError("This single-file implementation supports event_survival_estimator='cox' only.")
        if m_pred_mode != "bridge":
            raise ValueError("This single-file implementation supports m_pred_mode='bridge' only.")

        self._target = target
        self._horizon = horizon
        self._cv = int(cv)
        self._random_state = int(random_state)
        self._q_kind = q_kind
        self._q_trees = int(q_trees)
        self._q_min_samples_leaf = int(q_min_samples_leaf)
        self._q_poly_degree = int(q_poly_degree)
        self._q_clip = float(q_clip)
        self._h_kind = h_kind
        self._h_n_estimators = int(h_n_estimators)
        self._h_min_samples_leaf = int(h_min_samples_leaf)
        self._y_tilde_clip_quantile = y_tilde_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._censoring_estimator = censoring_estimator
        self._event_survival_estimator = event_survival_estimator
        self._m_pred_mode = m_pred_mode
        self._nuisance_feature_mode = nuisance_feature_mode
        self._prediction_nuisance_mode = prediction_nuisance_mode
        self._observed_only = bool(observed_only)

        self._q_model_template = make_q_model(
            q_kind,
            random_state=random_state,
            n_estimators=q_trees,
            min_samples_leaf=q_min_samples_leaf,
            poly_degree=q_poly_degree,
        )
        self._h_model_template = make_h_model(
            h_kind,
            random_state=random_state,
            n_estimators=h_n_estimators,
            min_samples_leaf=h_min_samples_leaf,
            n_jobs=1,
        )

        self._dml_model = None
        self._feature_nuisance = None
        self._train_x = None
        self._train_w = None
        self._train_z = None
        self._train_x_final = None

    def _prepare_nuisance_inputs(self, W, Z):
        """Optionally zero proxies for no-PCI ablations.

        The current default Final Model keeps the observed proxies, so this is
        a no-op by default. We keep the hook because it is conceptually part of
        the current code path.
        """

        w = _ensure_2d(W).astype(float)
        z = _ensure_2d(Z).astype(float)
        if self._observed_only:
            w = np.zeros_like(w)
            z = np.zeros_like(z)
        return w, z

    def _make_feature_nuisance(self):
        """Create the full-data nuisance object used at prediction time."""

        return _FinalCensoredNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            target=self._target,
            horizon=self._horizon,
            nuisance_feature_mode=self._nuisance_feature_mode,
            censoring_estimator=self._censoring_estimator,
            q_clip=self._q_clip,
            y_tilde_clip_quantile=self._y_tilde_clip_quantile,
            y_res_clip_percentiles=self._y_res_clip_percentiles,
            event_survival_estimator=self._event_survival_estimator,
            m_pred_mode=self._m_pred_mode,
        )

    def fit_components(self, X, A, time, event, Z, W):
        """Fit the full model in two stages.

        Stage 1: use a custom CausalForestDML subclass to cross-fit nuisance
        models, compute x_final internally, and fit the final heterogeneity
        learner.

        Stage 2: refit the nuisance models on the full training data so we can
        reconstruct x_final consistently at prediction time.
        """

        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)

        # `raw_w` and `raw_z` are the observed proxy blocks used in the final
        # feature representation `[X, W, Z, q, h1, h0, m, surv1, surv0, diff]`.
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)

        self._dml_model = _SinglePassBridgeFeatureCensoredSurvivalForest(
            target=self._target,
            horizon=self._horizon,
            n_estimators=self._n_estimators,
            min_samples_leaf=self._min_samples_leaf,
            cv=self._cv,
            random_state=self._random_state,
            q_kind=self._q_kind,
            q_trees=self._q_trees,
            q_min_samples_leaf=self._q_min_samples_leaf,
            q_poly_degree=self._q_poly_degree,
            q_clip=self._q_clip,
            y_tilde_clip_quantile=self._y_tilde_clip_quantile,
            y_res_clip_percentiles=self._y_res_clip_percentiles,
            h_kind=self._h_kind,
            h_n_estimators=self._h_n_estimators,
            h_min_samples_leaf=self._h_min_samples_leaf,
            censoring_estimator=self._censoring_estimator,
            event_survival_estimator=self._event_survival_estimator,
            m_pred_mode=self._m_pred_mode,
            nuisance_feature_mode=self._nuisance_feature_mode,
        )
        # Save the raw proxy blocks so the custom final-model wrapper can
        # rebuild `x_final` inside EconML after cross-fitting the nuisances.
        self._dml_model._raw_w_for_final = raw_w
        self._dml_model._raw_z_for_final = raw_z
        self._dml_model.fit_survival(x, A, time, event, z_nuis, w_nuis)

        self._train_x = np.asarray(x, dtype=float).copy()
        self._train_w = np.asarray(raw_w, dtype=float).copy()
        self._train_z = np.asarray(raw_z, dtype=float).copy()
        self._train_x_final = self._dml_model.training_x_final()

        # Refit nuisances on the full training sample for prediction-time
        # reconstruction of `x_final`.
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        self._feature_nuisance = self._make_feature_nuisance()
        self._feature_nuisance.train(
            False,
            None,
            y_packed,
            np.asarray(A, dtype=float).ravel(),
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def effect_from_components(self, X, W, Z):
        """Predict treatment effects by rebuilding x_final from full-data nuisances."""

        if self._dml_model is None or self._feature_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")

        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)

        # Recompute the nuisance summaries at the new point, then rebuild the
        # same `x_final` representation used during training.
        bridge = self._feature_nuisance.predict_bridge_outputs(
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        x_final = _build_final_features_full(x, raw_w, raw_z, bridge)
        return self._dml_model.effect_on_final_features(x_final)


# Backward-friendly alias. The file name distinguishes the single-file version,
# so reusing the familiar class name makes comparisons less awkward.
FinalModelCensoredSurvivalForest = FinalCensoredModel


__all__ = [
    "FinalCensoredModel",
    "FinalModelCensoredSurvivalForest",
]
