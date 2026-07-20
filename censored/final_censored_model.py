"""Single-file reference implementation of the current censored Final Model (full).

This file intentionally inlines the helper functions that are otherwise spread
across multiple project files. The goal is readability: someone should be able
to follow the full pipeline here without jumping between local modules.

Scope
-----
This file reproduces the *current default* censored final model used in the
benchmark pipeline, and also exposes the final-stage ablations that keep the
PCI residualization fixed while changing only the final forest input:

    - final feature mode: "full" (default)
    - optional ablation modes:
        - "surv_only" -> [X, W, Z, surv1_pred, surv0_pred, surv_diff_pred]
        - "x_only" -> [X]
    - raw final backbone: [X, W, Z]
    - bridge summaries passed to the final learner in the default mode:
        [q_pred, h1_pred, h0_pred, m_pred, surv1_pred, surv0_pred, surv_diff_pred]
    - prediction nuisance mode: "full_refit"
    - event survival estimator: Cox
    - censoring mode: marginal (default) or conditional
    - marginal censoring estimator: Nelson-Aalen (default), with Kaplan-Meier also supported
    - conditional censoring estimator: Cox on [X, W, Z, A]
    - nuisance feature mode: broad_dup
    - final learner: EconML CausalForestDML with a custom final-feature wrapper

The implementation below is deliberately focused on the default path that we
actually benchmark plus the final-stage ablations that are most useful for
separating PCI-residualization effects from split-feature effects.

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
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

try:  # pragma: no cover
    from .gpu_backends import make_xgb_classifier, make_xgb_regressor
except ImportError:  # pragma: no cover
    from censored.gpu_backends import make_xgb_classifier, make_xgb_regressor  # type: ignore


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


def _clip_quantile_by_arm(values, q, treated_mask, control_mask):
    """Winsorize WITHIN each treatment arm rather than pooled.

    The treated and control arms have different follow-up, censoring and IPCW-weight
    distributions, so a single POOLED cutoff removes a different FRACTION of each arm.
    That shrinks the two arm-specific quantities by different amounts and therefore
    biases their DIFFERENCE -- which is exactly the estimand. Clipping per arm makes
    each arm lose the same tail fraction, so the truncation is effect-neutral.
    """
    values = np.asarray(values, dtype=float)
    if q is None:
        return values
    out = values.copy()
    for mask in (treated_mask, control_mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            out[mask] = _clip_quantile(values[mask], q)
    return out


def _clip_percentile_by_arm(values, percentiles, a):
    """Percentile-winsorize WITHIN each treatment arm (see _clip_quantile_by_arm)."""

    values = np.asarray(values, dtype=float)
    if percentiles is None:
        return values
    a = np.asarray(a, dtype=float).ravel()
    out = values.copy()
    for mask in (a == 1, a == 0):
        if mask.any():
            lo, hi = np.percentile(values[mask], percentiles)
            out[mask] = np.clip(values[mask], lo, hi)
    return out


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
    horizon and treated as event-observed at that truncation point. The shared
    survival grid is also extended to include the requested horizon so the
    Cox-based q-hat summaries target RMST up to that horizon even when it lies
    beyond the largest observed follow-up time.
    """

    y_time = np.asarray(y_time, dtype=float)
    delta = np.asarray(delta, dtype=float)

    if target not in {"RMST", "survival.probability"}:
        raise ValueError("target must be one of {'RMST', 'survival.probability'}.")

    if target == "survival.probability":
        if horizon is None:
            raise ValueError("horizon is required when target='survival.probability'.")
        # A subject "survives to h" iff they REACH the horizon without an event.
        # The pipeline SQL caps follow-up at LEAST(<event/censor date>, index+horizon),
        # so y_time can never EXCEED the horizon: the previous strict `y_time > horizon`
        # made this indicator identically 0 for every patient and silently killed the
        # entire survival-probability estimand. Mirror the RMST branch's `>=` (:154).
        survived = (y_time > float(horizon)) | ((y_time >= float(horizon)) & (delta == 0))
        # Horizon-completers are ADMINISTRATIVELY complete, not informatively censored.
        # Previously `nuisance_delta = delta` handed every event-free completer to the
        # censoring model as a censoring at exactly t=h, which cliffs the censoring
        # curve and pins the IPCW weights at the 1/clip_min (=100x) cap.
        nuisance_time = np.minimum(y_time, float(horizon))
        nuisance_delta = delta.copy()
        nuisance_delta[y_time >= float(horizon)] = 1.0
        eval_time = nuisance_time
        eval_delta = nuisance_delta
        f_y = survived.astype(float)
        grid_time = nuisance_time
        # EVENT-model delta (separate from the censoring-relabeled nuisance_delta): the
        # arm-specific event Cox must treat horizon-reachers as CENSORED at h (event-free),
        # NOT as events. Reusing nuisance_delta (=1 for reachers) injects a spurious event
        # mass at t=h that collapses the fitted S(h) nuisance toward 0. The =1 relabel is
        # correct ONLY for the censoring model (admin completion is not informative censoring).
        event_time = nuisance_time
        event_delta = delta.copy()
        event_delta[y_time >= float(horizon)] = 0.0
    elif horizon is None:
        nuisance_time = y_time
        nuisance_delta = delta
        eval_time = y_time
        eval_delta = delta
        f_y = y_time
        grid_time = y_time
        event_time = y_time
        event_delta = delta
    else:
        nuisance_time = np.minimum(y_time, float(horizon))
        nuisance_delta = delta.copy()
        nuisance_delta[y_time >= float(horizon)] = 1.0
        eval_time = nuisance_time
        eval_delta = nuisance_delta
        f_y = nuisance_time
        grid_time = np.concatenate([nuisance_time, np.asarray([float(horizon)], dtype=float)])
        event_time = nuisance_time
        event_delta = delta.copy()
        event_delta[y_time >= float(horizon)] = 0.0

    return {
        "nuisance_time": nuisance_time,
        "nuisance_delta": nuisance_delta,
        "eval_time": eval_time,
        "eval_delta": eval_delta,
        "f_y": f_y,
        "grid_time": grid_time,
        "event_time": event_time,
        "event_delta": event_delta,
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
    n_jobs=1,
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
    if kind == "xgb":
        return make_xgb_classifier(
            random_state=random_state,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            device="cpu",
        )
    if kind == "xgb_gpu":
        return make_xgb_classifier(
            random_state=random_state,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            device="cuda",
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
    if kind == "xgb":
        return make_xgb_regressor(
            random_state=random_state,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            device="cpu",
        )
    if kind == "xgb_gpu":
        return make_xgb_regressor(
            random_state=random_state,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            device="cuda",
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


def _fit_conditional_censoring_model(time, event, features, *, estimator="cox"):
    """Fit S_C(t | X, W, Z, A) for conditional censoring adjustment."""

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


def _predict_conditional_censoring_survival_at_values(censor_model, features, values, *, clip_min=0.01):
    """Evaluate S_C(y | X, W, Z, A) for each observation."""

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
    """Evaluate S_C(t | X, W, Z, A) on a common time grid."""

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

    The current Final Model defaults to `broad_dup`, which composes the
    proximal bridge inputs per the identification requirement:

      - q (treatment bridge) = [X, Z]   (NC-treatment proxy Z only)
      - h (outcome bridge)   = [X, W]   (NC-outcome proxy W only)
      - survival nuisance    = [X, W, Z] (not a bridge; may use both proxies)

    (Historically this branch duplicated/cross-included proxies, giving
    q=[X,W,Z,Z] and h=[X,W,Z,W], which violated the bridge exclusion
    structure; that has been corrected.)
    """

    raw_x = _ensure_2d(raw_x)
    w = _ensure_2d(w)
    z = _ensure_2d(z)

    if feature_mode == "broad_dup":
        # Proximal identification requires q = q(X, Z) and h = h(X, W): the
        # treatment bridge q may see ONLY the NC-treatment proxy Z (not the
        # NC-outcome proxy W), and the outcome bridge h ONLY W (not Z). The
        # earlier `[base, z]`/`[base, w]` leaked the wrong proxy into each
        # bridge (q=[X,W,Z,Z], h=[X,W,Z,W]) AND duplicated one proxy, which
        # violated the bridge exclusion structure. Compose them correctly.
        q_features = np.column_stack([raw_x, z])
        h_features = np.column_stack([raw_x, w])
        # surv is an arm-specific event-survival nuisance (NOT a proximal
        # bridge), so it may legitimately condition on both observed proxies;
        # use each once ([X, W, Z]) to avoid literal duplicate columns that
        # cause collinearity in the Cox event-survival fit.
        surv_features = np.column_stack([raw_x, w, z])
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


SUPPORTED_FINAL_FEATURE_MODES = ("full", "surv_only", "x_only", "raw_only")


def _build_final_features_surv_only(x, raw_w, raw_z, bridge):
    """Build the survival-summary-only final-stage feature block."""

    return np.hstack(
        [
            _ensure_2d(x).astype(float),
            _ensure_2d(raw_w).astype(float),
            _ensure_2d(raw_z).astype(float),
            np.asarray(bridge["surv1_pred"], dtype=float).reshape(-1, 1),
            np.asarray(bridge["surv0_pred"], dtype=float).reshape(-1, 1),
            np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1),
        ]
    )


def _build_final_features_x_only(x, raw_w, raw_z, bridge):
    """Build the X-only final-stage feature block."""

    del raw_w, raw_z, bridge
    return _ensure_2d(x).astype(float)


def _build_final_features_raw_only(x, raw_w, raw_z, bridge):
    """Build the raw observed final-stage feature block [X, W, Z]."""

    del bridge
    return np.hstack(
        [
            _ensure_2d(x).astype(float),
            _ensure_2d(raw_w).astype(float),
            _ensure_2d(raw_z).astype(float),
        ]
    )


def _build_final_features(mode, x, raw_w, raw_z, bridge):
    """Dispatch to the requested final-stage feature construction."""

    if mode == "full":
        return _build_final_features_full(x, raw_w, raw_z, bridge)
    if mode == "surv_only":
        return _build_final_features_surv_only(x, raw_w, raw_z, bridge)
    if mode == "x_only":
        return _build_final_features_x_only(x, raw_w, raw_z, bridge)
    if mode == "raw_only":
        return _build_final_features_raw_only(x, raw_w, raw_z, bridge)
    raise ValueError(
        f"Unsupported final_feature_mode={mode!r}. "
        f"Expected one of {SUPPORTED_FINAL_FEATURE_MODES}."
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
    f_y=None,
):
    """Create the IPCW pseudo-outcome used to fit the h nuisance models."""

    if target == "survival.probability":
        # Consume the survivor indicator prepared upstream. Re-deriving it here as
        # `y_time > horizon` repeated the boundary bug: pipeline follow-up is capped
        # at the horizon, so that comparison is never true and the pseudo-outcome
        # collapsed to 0 for every patient.
        if f_y is None:
            raise ValueError(
                "survival.probability requires the upstream survivor indicator f_y; "
                "re-deriving it from y_time re-introduces the horizon-boundary bug."
            )
        return np.asarray(f_y, dtype=float) / np.maximum(sc_at_eval, 1e-10)
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
    a,
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
    # Winsorize WITHIN arm: a pooled cutoff truncates a different fraction of each
    # arm (different follow-up/censoring/IPCW tails), which attenuates the R-learner
    # moment E[y_res * a_res] asymmetrically and biases the effect.
    return _clip_percentile_by_arm(y_res, clip_percentiles, a)


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
    a,
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
    # Winsorize WITHIN arm: a pooled cutoff truncates a different fraction of each
    # arm (different follow-up/censoring/IPCW tails), which attenuates the R-learner
    # moment E[y_res * a_res] asymmetrically and biases the effect.
    return _clip_percentile_by_arm(y_res, clip_percentiles, a)


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
        censoring_mode,
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
        if censoring_mode not in {"marginal", "conditional"}:
            raise ValueError("censoring_mode must be either 'marginal' or 'conditional'.")
        self._censoring_mode = censoring_mode
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

    def _bound_h_predictions(self, h1_pred, h0_pred):
        """Enforce the range constraint the outcome bridges must satisfy.

        For target='survival.probability' the h-bridges estimate S(h | X, W) -- a
        PROBABILITY, so their predictions must lie in [0, 1]. This clip enforces that
        known range constraint on the nuisance, and bounds the downstream CATE.

        NOTE ON THE ORIGINAL DIAGNOSIS: the out-of-range estimates first observed
        (-4.6, -6.4, -17.7) were NOT merely "a regression on a heavy-tailed target".
        The root cause was a horizon-boundary bug -- the survivor indicator used a
        strict `y_time > horizon` while the pipeline SQL caps follow-up AT the
        horizon, so the indicator was identically 0 and the censoring model was fed
        every completer as a censoring at t=h (pinning IPCW at the 100x cap). That
        is fixed upstream now (see _prepare_target_inputs / the survival.probability
        branch). This clip remains as a cheap range guard, not as the fix.

        RMST targets live on the day scale (bounded only by the horizon), so they are
        deliberately left untouched here.
        """
        if self._target == "survival.probability":
            return np.clip(h1_pred, 0.0, 1.0), np.clip(h0_pred, 0.0, 1.0)
        return h1_pred, h0_pred

    def _compute_m_pred(self, q_pred, h1_pred, h0_pred):
        if self._m_pred_mode != "bridge":
            raise ValueError("This single-file implementation supports m_pred_mode='bridge' only.")
        return q_pred * h1_pred + (1.0 - q_pred) * h0_pred

    @staticmethod
    def _build_censor_features(x, w, z):
        return np.column_stack([_ensure_2d(x), _ensure_2d(w), _ensure_2d(z)])

    @staticmethod
    def _augment_censor_features_with_treatment(features, a):
        return np.column_stack([np.asarray(features, dtype=float), np.asarray(a, dtype=float).reshape(-1, 1)])

    def _fit_censoring_nuisance(self, time, event, censor_features_obs):
        if self._censoring_mode == "conditional":
            return _fit_conditional_censoring_model(
                time,
                event,
                censor_features_obs,
                estimator=self._censoring_estimator,
            )
        return _fit_censoring_model(time, event, estimator=self._censoring_estimator)

    def _predict_censoring_at_values(self, censor_features_obs, values):
        if self._censoring_mode == "conditional":
            return _predict_conditional_censoring_survival_at_values(
                self._censor_model,
                censor_features_obs,
                values,
            )
        return _predict_censoring_survival_at_values(self._censor_model, values)

    def _predict_censoring_on_grid(self, censor_features_obs, n_rows):
        if self._censoring_mode == "conditional":
            return _predict_conditional_censoring_survival_on_grid(
                self._censor_model,
                censor_features_obs,
                self._t_grid,
            )
        return _predict_censoring_survival_on_grid(self._censor_model, self._t_grid, n_rows)

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
        censor_features = self._build_censor_features(x, w, z)
        censor_features_obs = self._augment_censor_features_with_treatment(censor_features, a)

        all_times = np.sort(np.unique(target_inputs["grid_time"]))
        if len(all_times) > self.max_grid:
            idx = np.linspace(0, len(all_times) - 1, self.max_grid, dtype=int)
            all_times = all_times[idx]
        self._t_grid = all_times

        # Step 1: fit the censoring survival used for IPCW.
        self._censor_model = self._fit_censoring_nuisance(
            target_inputs["nuisance_time"],
            target_inputs["nuisance_delta"],
            censor_features_obs,
        )

        # Step 2: build the IPCW pseudo-outcome y_tilde used to fit h1 / h0.
        y_tilde_eval_time = (
            target_inputs["eval_time"]
            if self._target == "survival.probability"
            else target_inputs["nuisance_time"]
        )
        sc_for_y_tilde = self._predict_censoring_at_values(censor_features_obs, y_tilde_eval_time)
        y_tilde = _compute_target_pseudo_outcome_from_sc(
            y_time=y_time,
            target=self._target,
            horizon=self._horizon,
            nuisance_time=target_inputs["nuisance_time"],
            nuisance_delta=target_inputs["nuisance_delta"],
            sc_at_eval=sc_for_y_tilde,
            f_y=target_inputs["f_y"],
        )
        # Step 3: fit q(X, Z), h1(X, W), h0(X, W).
        self._q_model = clone(self._q_model_template)
        self._q_model.fit(xq, a, **filter_none_kwargs(sample_weight=sample_weight))

        treated_mask = a == 1
        control_mask = a == 0

        # Winsorize the IPCW pseudo-outcome WITHIN ARM, not pooled. y_tilde is split
        # below into the two arm-specific bridge regressions, and the arms have
        # different follow-up/censoring/IPCW-tail distributions -- so a single POOLED
        # cutoff truncates a different FRACTION of each arm, shrinking h1 and h0 by
        # different amounts and biasing their difference (the effect). Clipping per
        # arm makes each arm lose exactly the same tail fraction.
        y_tilde = _clip_quantile_by_arm(y_tilde, self._y_tilde_clip_quantile, treated_mask, control_mask)

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

        # Step 4: fit arm-specific event survival curves. Use event_time/event_delta
        # (horizon-reachers censored at h), NOT the censoring-relabeled nuisance_delta,
        # so the fitted S(h) nuisance is not collapsed by a spurious event mass at t=h.
        self._fit_event_survival_models(
            target_inputs["event_time"],
            target_inputs["event_delta"],
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
        censor_features = self._build_censor_features(x, w, z)
        censor_features_obs = self._augment_censor_features_with_treatment(censor_features, a)

        q_pred = self._q_model.predict_proba(xq)[:, 1]
        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        h1_pred = self._h1_model.predict(xh)
        h0_pred = self._h0_model.predict(xh)
        h1_pred, h0_pred = self._bound_h_predictions(h1_pred, h0_pred)
        s_hat_1, s_hat_0 = self._predict_event_survival_curves(surv_features)
        m_pred = self._compute_m_pred(q_pred, h1_pred, h0_pred)

        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, self._t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, self._t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, self._t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, self._t_grid)

        q_hat = np.where((a == 1)[:, None], q_hat_1, q_hat_0)
        sc_grid = self._predict_censoring_on_grid(censor_features_obs, len(x))

        if self._target == "RMST" and self._horizon is None:
            sc_at_eval = self._predict_censoring_at_values(censor_features_obs, y_time)
            y_res = _compute_ipcw_3term_y_res_from_survival(
                y_time,
                delta,
                m_pred,
                q_hat,
                self._t_grid,
                sc_at_eval,
                sc_grid,
                clip_percentiles=self._y_res_clip_percentiles,
                a=a,
            )
        else:
            sc_at_eval = self._predict_censoring_at_values(censor_features_obs, target_inputs["eval_time"])
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
                a=a,
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
        h1_pred, h0_pred = self._bound_h_predictions(h1_pred, h0_pred)
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

    def __init__(self, base_model_final, *, raw_proxy_supplier=None, final_feature_mode="full"):
        self._base_model_final = base_model_final
        self._raw_proxy_supplier = raw_proxy_supplier
        self._final_feature_mode = final_feature_mode
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

        x_final = _build_final_features(self._final_feature_mode, X, w_for_final, z_for_final, bridge)
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
        censoring_mode="marginal",
        censoring_estimator="nelson-aalen",
        event_survival_estimator="cox",
        m_pred_mode="bridge",
        nuisance_feature_mode="broad_dup",
        final_feature_mode="full",
        n_jobs=1,
    ):
        self._q_model_template = make_q_model(
            q_kind,
            random_state=random_state,
            n_estimators=q_trees,
            min_samples_leaf=q_min_samples_leaf,
            poly_degree=q_poly_degree,
            n_jobs=n_jobs,
        )
        self._h_model_template = make_h_model(
            h_kind,
            random_state=random_state,
            n_estimators=h_n_estimators,
            min_samples_leaf=h_min_samples_leaf,
            n_jobs=n_jobs,
        )
        self._target = target
        self._horizon = horizon
        self._censoring_mode = censoring_mode
        self._censoring_estimator = censoring_estimator
        self._event_survival_estimator = event_survival_estimator
        self._m_pred_mode = m_pred_mode
        self._nuisance_feature_mode = nuisance_feature_mode
        self._final_feature_mode = final_feature_mode
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
            n_jobs=n_jobs,
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
            censoring_mode=self._censoring_mode,
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
            final_feature_mode=self._final_feature_mode,
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
        censoring_mode="marginal",
        censoring_estimator="nelson-aalen",
        event_survival_estimator="cox",
        m_pred_mode="bridge",
        nuisance_feature_mode="broad_dup",
        final_feature_mode="full",
        prediction_nuisance_mode="full_refit",
        surv_scalar_mode="full",
        observed_only=False,
        n_jobs=1,
    ):
        if surv_scalar_mode != "full":
            raise ValueError("This single-file implementation reproduces only surv_scalar_mode='full'.")
        if prediction_nuisance_mode != "full_refit":
            raise ValueError("This single-file implementation supports prediction_nuisance_mode='full_refit' only.")
        if event_survival_estimator != "cox":
            raise ValueError("This single-file implementation supports event_survival_estimator='cox' only.")
        if censoring_mode not in {"marginal", "conditional"}:
            raise ValueError("censoring_mode must be either 'marginal' or 'conditional'.")
        if censoring_mode == "conditional" and censoring_estimator == "nelson-aalen":
            censoring_estimator = "cox"
        if m_pred_mode != "bridge":
            raise ValueError("This single-file implementation supports m_pred_mode='bridge' only.")
        if final_feature_mode not in SUPPORTED_FINAL_FEATURE_MODES:
            raise ValueError(
                f"Unsupported final_feature_mode={final_feature_mode!r}. "
                f"Expected one of {SUPPORTED_FINAL_FEATURE_MODES}."
            )

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
        # survival.probability's IPCW pseudo-outcome is f_y / S_c on a 0/1 survivor
        # indicator, so its 1/S_c tails are far heavier than the day-scale RMST
        # target. The default (2, 98) arm-wise winsor truncates the treatment signal
        # that lives in those tails and OVER-SHRINKS the survival-probability CATE
        # toward 0 -- which collapses the E-value (RR -> 1). Empirically (MIMIC
        # warfarin+amiodarone), relaxing the survival-probability winsor to
        # (0.5, 99.5) recovers ~70% of the effect (tau -0.046 -> -0.079, vs naive/
        # AIPW -0.11) at negligible variance cost; RMST keeps the tighter default.
        if self._target == "survival.probability" and tuple(y_res_clip_percentiles) == (2.0, 98.0):
            self._y_res_clip_percentiles = (0.5, 99.5)
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._censoring_mode = censoring_mode
        self._censoring_estimator = censoring_estimator
        self._event_survival_estimator = event_survival_estimator
        self._m_pred_mode = m_pred_mode
        self._nuisance_feature_mode = nuisance_feature_mode
        self._final_feature_mode = final_feature_mode
        self._prediction_nuisance_mode = prediction_nuisance_mode
        self._observed_only = bool(observed_only)
        self._n_jobs = int(n_jobs)

        self._q_model_template = make_q_model(
            q_kind,
            random_state=random_state,
            n_estimators=q_trees,
            min_samples_leaf=q_min_samples_leaf,
            poly_degree=q_poly_degree,
            n_jobs=self._n_jobs,
        )
        self._h_model_template = make_h_model(
            h_kind,
            random_state=random_state,
            n_estimators=h_n_estimators,
            min_samples_leaf=h_min_samples_leaf,
            n_jobs=self._n_jobs,
        )

        self._dml_model = None
        self._feature_nuisance = None
        self._feature_nuisance_folds = None
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
            censoring_mode=self._censoring_mode,
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
        # representation selected by `final_feature_mode`.
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
            censoring_mode=self._censoring_mode,
            censoring_estimator=self._censoring_estimator,
            event_survival_estimator=self._event_survival_estimator,
            m_pred_mode=self._m_pred_mode,
            nuisance_feature_mode=self._nuisance_feature_mode,
            final_feature_mode=self._final_feature_mode,
            n_jobs=self._n_jobs,
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

        # Prediction-time nuisances must be CROSS-FITTED, matching the ones the final
        # forest was actually trained on. Refitting a single model on the FULL sample
        # (the previous behaviour) produced IN-SAMPLE bridge columns at predict time
        # while the forest's splits were calibrated on OUT-OF-FOLD ones -- a
        # train/serve skew that measurably shifted per-patient tau. We therefore fit a
        # K-fold ensemble and AVERAGE its bridge outputs at prediction time, which is
        # how cross-fitted nuisances are normally served.
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        a_arr = np.asarray(A, dtype=float).ravel()
        n_rows = len(a_arr)
        n_splits = max(2, min(int(self._cv), n_rows))
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=self._random_state)
        self._feature_nuisance_folds = []
        for train_idx, _ in folds.split(np.arange(n_rows)):
            nu = self._make_feature_nuisance()
            nu.train(
                False,
                None,
                y_packed[train_idx],
                a_arr[train_idx],
                X=x[train_idx],
                W=w_nuis[train_idx],
                Z=z_nuis[train_idx],
            )
            self._feature_nuisance_folds.append(nu)

        # Keep a full-sample fit for callers that need a single nuisance object
        # (e.g. the light-bootstrap final-stage score). NOT used for effects.
        self._feature_nuisance = self._make_feature_nuisance()
        self._feature_nuisance.train(
            False,
            None,
            y_packed,
            a_arr,
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def _predict_bridge_outputs_crossfit(self, x, w_nuis, z_nuis):
        """Average the K fold-fitted nuisances' bridge outputs.

        Serves the SAME kind of (cross-fitted) bridge columns the final forest was
        trained on, instead of the in-sample columns a single full-data refit gives.
        """
        models = getattr(self, "_feature_nuisance_folds", None) or [self._feature_nuisance]
        preds = [m.predict_bridge_outputs(X=x, W=w_nuis, Z=z_nuis) for m in models]
        if len(preds) == 1:
            return preds[0]
        return {
            key: np.mean([np.asarray(p[key], dtype=float) for p in preds], axis=0)
            for key in preds[0]
        }

    def effect_from_components(self, X, W, Z):
        """Predict treatment effects by rebuilding x_final from CROSS-FITTED nuisances."""

        if self._dml_model is None or self._feature_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")

        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)

        # Recompute the nuisance summaries at the new point, then rebuild the
        # same `x_final` representation used during training.
        bridge = self._predict_bridge_outputs_crossfit(x, w_nuis, z_nuis)
        x_final = _build_final_features(self._final_feature_mode, x, raw_w, raw_z, bridge)
        tau = self._dml_model.effect_on_final_features(x_final)
        if self._target == "survival.probability":
            # Final guarantee: a survival-probability CATE is a DIFFERENCE OF
            # PROBABILITIES and cannot leave [-1, +1]. _bound_h_predictions already
            # constrains the bridges; this catches any residual forest extrapolation
            # on rare x_final rows so the reported estimand is always in range.
            tau = np.clip(tau, -1.0, 1.0)
        return tau


class FinalPCISurvOnlyCensoredModel(FinalCensoredModel):
    """PCI-residualized Final Model with survival-only final splitting features."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "surv_only")
        super().__init__(*args, **kwargs)


class FinalPCIXOnlyCensoredModel(FinalCensoredModel):
    """PCI-residualized Final Model with X-only final splitting features."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "x_only")
        super().__init__(*args, **kwargs)


class ConditionalCensoringFinalCensoredModel(FinalCensoredModel):
    """Final Model with conditional Cox censoring nuisance S_C(t | X, W, Z, A)."""

    def __init__(self, *args, censoring_estimator="cox", **kwargs):
        kwargs.setdefault("censoring_mode", "conditional")
        if kwargs["censoring_mode"] != "conditional":
            raise ValueError("ConditionalCensoringFinalCensoredModel requires censoring_mode='conditional'.")
        if censoring_estimator != "cox":
            raise ValueError(
                "ConditionalCensoringFinalCensoredModel currently supports only censoring_estimator='cox'."
            )
        super().__init__(*args, censoring_estimator=censoring_estimator, **kwargs)


class ConditionalCensoringFinalPCISurvOnlyCensoredModel(ConditionalCensoringFinalCensoredModel):
    """Conditional-censoring Final Model with PCI residuals and survival-only final features."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "surv_only")
        super().__init__(*args, **kwargs)


class ConditionalCensoringFinalPCIXOnlyCensoredModel(ConditionalCensoringFinalCensoredModel):
    """Conditional-censoring Final Model with PCI residuals and X-only final features."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "x_only")
        super().__init__(*args, **kwargs)


# Backward-friendly alias. The file name distinguishes the single-file version,
# so reusing the familiar class name makes comparisons less awkward.
FinalModelCensoredSurvivalForest = FinalCensoredModel
FinalModelPCISurvOnlyCensoredSurvivalForest = FinalPCISurvOnlyCensoredModel
FinalModelPCIXOnlyCensoredSurvivalForest = FinalPCIXOnlyCensoredModel
FinalModelConditionalCensoredSurvivalForest = ConditionalCensoringFinalCensoredModel
FinalModelConditionalPCISurvOnlyCensoredSurvivalForest = ConditionalCensoringFinalPCISurvOnlyCensoredModel
FinalModelConditionalPCIXOnlyCensoredSurvivalForest = ConditionalCensoringFinalPCIXOnlyCensoredModel


__all__ = [
    "SUPPORTED_FINAL_FEATURE_MODES",
    "FinalCensoredModel",
    "FinalModelCensoredSurvivalForest",
    "FinalPCISurvOnlyCensoredModel",
    "FinalPCIXOnlyCensoredModel",
    "ConditionalCensoringFinalCensoredModel",
    "ConditionalCensoringFinalPCISurvOnlyCensoredModel",
    "ConditionalCensoringFinalPCIXOnlyCensoredModel",
    "FinalModelPCISurvOnlyCensoredSurvivalForest",
    "FinalModelPCIXOnlyCensoredSurvivalForest",
    "FinalModelConditionalCensoredSurvivalForest",
    "FinalModelConditionalPCISurvOnlyCensoredSurvivalForest",
    "FinalModelConditionalPCIXOnlyCensoredSurvivalForest",
]
