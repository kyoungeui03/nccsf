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

from ..backends import NativeSurvivalForest
from ..core import default_mtry, expected_survival, find_interval


def _ensure_2d(array):
    array = np.asarray(array)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _pairwise_products(left, right):
    left = _ensure_2d(left)
    right = _ensure_2d(right)
    if left.shape[1] == 0 or right.shape[1] == 0:
        return np.empty((left.shape[0], 0))
    return (left[:, :, None] * right[:, None, :]).reshape(left.shape[0], -1)


def _recover_raw_x(final_x, w, z, final_feature_mode):
    final_x = _ensure_2d(final_x)
    w = _ensure_2d(w)
    z = _ensure_2d(z)
    if final_feature_mode == "x_only":
        return final_x
    if final_feature_mode == "xwz":
        p_raw = final_x.shape[1] - w.shape[1] - z.shape[1]
        if p_raw <= 0:
            raise ValueError("Could not recover raw X from final X under final_feature_mode='xwz'.")
        return final_x[:, :p_raw]
    if final_feature_mode == "summary_minimal":
        p_raw = final_x.shape[1] - 4
        if p_raw <= 0:
            raise ValueError("Could not recover raw X from final X under final_feature_mode='summary_minimal'.")
        return final_x[:, :p_raw]
    if final_feature_mode == "summary_surv":
        p_raw = final_x.shape[1] - 7
        if p_raw <= 0:
            raise ValueError("Could not recover raw X from final X under final_feature_mode='summary_surv'.")
        return final_x[:, :p_raw]
    if final_feature_mode == "summary_surv_pair":
        p_raw = final_x.shape[1] - 6
        if p_raw <= 0:
            raise ValueError("Could not recover raw X from final X under final_feature_mode='summary_surv_pair'.")
        return final_x[:, :p_raw]
    if final_feature_mode == "augmented_minimal":
        p_raw = final_x.shape[1] - w.shape[1] - z.shape[1] - 4
        if p_raw <= 0:
            raise ValueError("Could not recover raw X from final X under final_feature_mode='augmented_minimal'.")
        return final_x[:, :p_raw]
    if final_feature_mode == "augmented_surv":
        p_raw = final_x.shape[1] - w.shape[1] - z.shape[1] - 7
        if p_raw <= 0:
            raise ValueError("Could not recover raw X from final X under final_feature_mode='augmented_surv'.")
        return final_x[:, :p_raw]
    if final_feature_mode == "augmented_surv_pair":
        p_raw = final_x.shape[1] - w.shape[1] - z.shape[1] - 6
        if p_raw <= 0:
            raise ValueError("Could not recover raw X from final X under final_feature_mode='augmented_surv_pair'.")
        return final_x[:, :p_raw]
    if final_feature_mode.startswith("summary_custom_"):
        summary_count = int(final_feature_mode.split("_")[-1])
        p_raw = final_x.shape[1] - summary_count
        if p_raw <= 0:
            raise ValueError(f"Could not recover raw X from final X under final_feature_mode='{final_feature_mode}'.")
        return final_x[:, :p_raw]
    if final_feature_mode.startswith("augmented_custom_"):
        summary_count = int(final_feature_mode.split("_")[-1])
        p_raw = final_x.shape[1] - w.shape[1] - z.shape[1] - summary_count
        if p_raw <= 0:
            raise ValueError(f"Could not recover raw X from final X under final_feature_mode='{final_feature_mode}'.")
        return final_x[:, :p_raw]
    raise ValueError(f"Unsupported final_feature_mode: {final_feature_mode}")


def _build_nuisance_features(raw_x, w, z, feature_mode):
    raw_x = _ensure_2d(raw_x)
    w = _ensure_2d(w)
    z = _ensure_2d(z)
    base = np.column_stack([raw_x, w, z])

    if feature_mode == "broad_dup":
        # Mirror the non-censored broad-dup logic:
        # build a shared proxy-aware base [X, W, Z], then duplicate the
        # branch-specific proxy block one more time for q / h / survival.
        q_features = np.column_stack([base, z])
        h_features = np.column_stack([base, w])
        surv_features = np.column_stack([base, w, z])
    else:
        q_features = np.column_stack([raw_x, z])
        h_features = np.column_stack([raw_x, w])
        surv_features = np.column_stack([raw_x, w, z])

    if feature_mode == "interact":
        xw = _pairwise_products(raw_x, w)
        xz = _pairwise_products(raw_x, z)
        wz = _pairwise_products(w, z)
        extra = np.column_stack([xw, xz, wz])
        q_features = np.column_stack([q_features, extra])
        h_features = np.column_stack([h_features, extra])
        surv_features = np.column_stack([surv_features, extra])
    elif feature_mode not in {"dup", "broad_dup"}:
        raise ValueError(f"Unsupported nuisance feature mode: {feature_mode}")

    return q_features, h_features, surv_features, base


def make_q_model(
    kind="logit",
    *,
    random_state=42,
    n_estimators=300,
    min_samples_leaf=20,
    poly_degree=2,
):
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
    kind="rf",
    *,
    random_state=42,
    n_estimators=300,
    min_samples_leaf=20,
    n_jobs=1,
):
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


def _fit_kaplan_meier_censoring(time, event):
    censor_indicator = 1 - np.asarray(event).astype(int)
    kmf = KaplanMeierFitter()
    kmf.fit(np.asarray(time, dtype=float), event_observed=censor_indicator)
    return np.asarray(kmf.survival_function_.index, dtype=float), np.asarray(
        kmf.survival_function_["KM_estimate"], dtype=float
    )


def _fit_nelson_aalen_censoring(time, event):
    censor_indicator = 1 - np.asarray(event).astype(int)
    naf = NelsonAalenFitter()
    naf.fit(np.asarray(time, dtype=float), event_observed=censor_indicator)
    times = np.asarray(naf.cumulative_hazard_.index, dtype=float)
    cum_hazard = np.asarray(naf.cumulative_hazard_["NA_estimate"], dtype=float)
    surv = np.exp(-cum_hazard)
    return times, surv


def _fit_censoring_survival(time, event, estimator="kaplan-meier"):
    if estimator in {"aalen", "nelson-aalen"}:
        return _fit_nelson_aalen_censoring(time, event)
    return _fit_kaplan_meier_censoring(time, event)


def _fit_censoring_model(
    time,
    event,
    features,
    estimator="kaplan-meier",
    *,
    failure_times=None,
    num_trees=50,
    min_node_size=15,
    n_jobs=1,
    random_state=42,
    fast_logrank=False,
    sample_weight=None,
):
    if estimator == "cox":
        cox, col_names, keep_mask = _fit_event_cox(time, 1 - np.asarray(event).astype(float), features)
        return {
            "kind": "cox",
            "model": cox,
            "col_names": col_names,
            "keep_mask": keep_mask,
        }
    if estimator == "survival_forest":
        return {
            "kind": "survival_forest",
            "model": _fit_native_survival_forest(
                features,
                time,
                1.0 - np.asarray(event).astype(float),
                failure_times=failure_times,
                num_trees=num_trees,
                min_node_size=min_node_size,
                n_jobs=n_jobs,
                random_state=random_state,
                fast_logrank=fast_logrank,
                sample_weight=sample_weight,
            ),
        }
    times, surv = _fit_censoring_survival(time, event, estimator=estimator)
    return {
        "kind": "marginal",
        "times": times,
        "surv": surv,
    }


def _predict_censoring_survival_at_values(censor_model, features, values, *, clip_min=0.01):
    values = np.asarray(values, dtype=float)
    if censor_model["kind"] == "marginal":
        return _evaluate_sc(values, censor_model["times"], censor_model["surv"], clip_min=clip_min)

    if censor_model["kind"] == "survival_forest":
        eval_times = np.sort(np.unique(values))
        surv = censor_model["model"].predict(np.asarray(features, dtype=float), failure_times=eval_times)
        idx = np.searchsorted(eval_times, values, side="right") - 1
        idx = np.clip(idx, 0, len(eval_times) - 1)
        return np.clip(surv[np.arange(len(values)), idx], clip_min, 1.0)

    eval_times = np.sort(np.unique(values))
    surv = _predict_s_on_grid(
        censor_model["model"],
        censor_model["col_names"],
        features,
        eval_times,
        censor_model["keep_mask"],
    )
    idx = np.searchsorted(eval_times, values, side="right") - 1
    idx = np.clip(idx, 0, len(eval_times) - 1)
    return np.clip(surv[np.arange(len(values)), idx], clip_min, 1.0)


def _predict_censoring_survival_on_grid(censor_model, features, t_grid, *, clip_min=0.01):
    t_grid = np.asarray(t_grid, dtype=float)
    if censor_model["kind"] == "marginal":
        surv = _evaluate_sc(t_grid, censor_model["times"], censor_model["surv"], clip_min=clip_min)
        return np.broadcast_to(surv[None, :], (len(features), len(t_grid)))

    if censor_model["kind"] == "survival_forest":
        surv = censor_model["model"].predict(np.asarray(features, dtype=float), failure_times=t_grid)
        return np.clip(surv, clip_min, 1.0)

    surv = _predict_s_on_grid(
        censor_model["model"],
        censor_model["col_names"],
        features,
        t_grid,
        censor_model["keep_mask"],
    )
    return np.clip(surv, clip_min, 1.0)


def _evaluate_sc(values, km_times, km_surv, clip_min=0.01):
    idx = np.searchsorted(km_times, np.asarray(values, dtype=float), side="right") - 1
    idx = np.clip(idx, 0, len(km_surv) - 1)
    return np.clip(km_surv[idx], clip_min, 1.0)


def _compute_ipcw_pseudo_outcome(y_time, delta, km_times, km_surv):
    sc = _evaluate_sc(y_time, km_times, km_surv)
    return np.asarray(y_time, dtype=float) * np.asarray(delta, dtype=float) / np.maximum(sc, 1e-10)


def _compute_target_pseudo_outcome(
    *,
    y_time,
    delta,
    target,
    horizon,
    nuisance_time,
    nuisance_delta,
    km_times,
    km_surv,
):
    if target == "survival.probability":
        eval_time = np.minimum(np.asarray(y_time, dtype=float), float(horizon))
        sc = _evaluate_sc(eval_time, km_times, km_surv)
        return (np.asarray(y_time, dtype=float) > float(horizon)).astype(float) / np.maximum(sc, 1e-10)
    return _compute_ipcw_pseudo_outcome(nuisance_time, nuisance_delta, km_times, km_surv)


def _compute_target_pseudo_outcome_from_sc(
    *,
    y_time,
    horizon,
    target,
    nuisance_time,
    nuisance_delta,
    sc_at_eval,
):
    if target == "survival.probability":
        return (np.asarray(y_time, dtype=float) > float(horizon)).astype(float) / np.maximum(sc_at_eval, 1e-10)
    return np.asarray(nuisance_time, dtype=float) * np.asarray(nuisance_delta, dtype=float) / np.maximum(sc_at_eval, 1e-10)


def _compute_q_from_s(s_hat, t_grid):
    n, g = s_hat.shape
    if g < 2:
        return np.full((n, g), t_grid[0] if g == 1 else 0.0)

    dt = np.diff(np.concatenate([[0.0], t_grid]))
    dot_products = s_hat[:, :-1] * dt[1:]

    cum = np.zeros((n, g))
    cum[:, 0] = dot_products.sum(axis=1)
    for k in range(1, g - 1):
        cum[:, k] = cum[:, k - 1] - dot_products[:, k - 1]

    q_hat = t_grid[None, :] + cum / np.maximum(s_hat, 1e-10)
    q_hat[:, -1] = t_grid[-1]
    return q_hat


def _compute_survival_probability_q_from_s(s_hat, t_grid, horizon):
    t_grid = np.asarray(t_grid, dtype=float)
    horizon_index = np.searchsorted(t_grid, float(horizon), side="right")
    if horizon_index == 0:
        raise ValueError("horizon cannot be before the first event.")
    q_hat = s_hat[:, [horizon_index - 1]] / np.maximum(s_hat, 1e-10)
    q_hat[:, horizon_index - 1 :] = 1.0
    return q_hat


def _fit_event_cox(y_time, delta, features, penalizer=0.01):
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
        except Exception as exc:  # lifelines raises several convergence exception types
            last_error = exc
    raise last_error


def _predict_s_on_grid(cox, col_names, features, t_grid, keep_mask):
    pred_df = pd.DataFrame(np.asarray(features, dtype=float)[:, keep_mask], columns=col_names)
    surv = cox.predict_survival_function(pred_df, times=np.asarray(t_grid, dtype=float))
    return np.clip(surv.values.T, 1e-10, 1.0)


def _fit_native_survival_forest(
    features,
    y_time,
    delta,
    *,
    failure_times,
    num_trees,
    min_node_size,
    n_jobs,
    random_state,
    fast_logrank=False,
    sample_weight=None,
):
    features = np.asarray(features, dtype=float)
    mtry = default_mtry(features.shape[1])
    return NativeSurvivalForest.fit(
        features,
        np.asarray(y_time, dtype=float),
        np.asarray(delta, dtype=float),
        failure_times=np.asarray(failure_times, dtype=float),
        num_trees=int(num_trees),
        sample_fraction=0.5,
        mtry=mtry,
        min_node_size=int(min_node_size),
        honesty=True,
        honesty_fraction=0.5,
        honesty_prune_leaves=True,
        alpha=0.05,
        prediction_type=1,
        fast_logrank=bool(fast_logrank),
        compute_oob_predictions=True,
        num_threads=int(n_jobs),
        seed=int(random_state),
        sample_weights=None if sample_weight is None else np.asarray(sample_weight, dtype=float),
    )


def _predict_native_survival_counterfactuals(sf_model, surv_features, t_grid):
    surv_features = np.asarray(surv_features, dtype=float)
    x_w1 = np.column_stack([surv_features, np.ones(surv_features.shape[0], dtype=float)])
    x_w0 = np.column_stack([surv_features, np.zeros(surv_features.shape[0], dtype=float)])
    s_hat_1 = np.clip(sf_model.predict(x_w1, failure_times=t_grid), 1e-10, 1.0)
    s_hat_0 = np.clip(sf_model.predict(x_w0, failure_times=t_grid), 1e-10, 1.0)
    return s_hat_1, s_hat_0


def _compute_survival_based_yhat(s_hat_1, s_hat_0, w_hat, t_grid, *, target, horizon):
    w_hat = np.asarray(w_hat, dtype=float).ravel()
    if target == "RMST":
        y_hat_1 = expected_survival(s_hat_1, t_grid)
        y_hat_0 = expected_survival(s_hat_0, t_grid)
    else:
        horizon_index = find_interval(np.array([float(horizon)]), np.asarray(t_grid, dtype=float))[0]
        if horizon_index == 0:
            y_hat_1 = np.ones(s_hat_1.shape[0], dtype=float)
            y_hat_0 = np.ones(s_hat_0.shape[0], dtype=float)
        else:
            y_hat_1 = s_hat_1[:, horizon_index - 1]
            y_hat_0 = s_hat_0[:, horizon_index - 1]
    return w_hat * y_hat_1 + (1.0 - w_hat) * y_hat_0


def _clip_quantile(values, q):
    if q is None:
        return values
    lo = float(np.quantile(values, 1.0 - q))
    hi = float(np.quantile(values, q))
    return np.clip(values, lo, hi)


def _compute_ipcw_3term_y_res(
    y_time,
    delta,
    m_pred,
    q_hat,
    t_grid,
    km_times,
    km_surv,
    *,
    clip_percentiles,
):
    n = len(y_time)
    g = len(t_grid)
    sc_at_y = _evaluate_sc(y_time, km_times, km_surv)
    sc_grid = _evaluate_sc(t_grid, km_times, km_surv)

    y_idx = np.searchsorted(t_grid, y_time, side="right") - 1
    y_idx = np.clip(y_idx, 0, g - 1)
    q_at_y = q_hat[np.arange(n), y_idx]

    numerator = delta * y_time + (1.0 - delta) * q_at_y
    term1 = numerator / np.maximum(sc_at_y, 1e-10)

    log_sc = -np.log(np.maximum(sc_grid, 1e-10))
    d_lambda_c = np.diff(np.concatenate([[0.0], log_sc]))
    grid_weight = d_lambda_c / np.maximum(sc_grid, 1e-10)
    integrand = grid_weight[None, :] * q_hat
    mask = np.arange(g)[None, :] <= y_idx[:, None]
    term2 = (integrand * mask).sum(axis=1)

    y_res = (term1 - term2) - m_pred
    lo, hi = np.percentile(y_res, clip_percentiles)
    return np.clip(y_res, lo, hi)


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
    n = len(y_time)
    g = len(t_grid)
    sc_grid = np.asarray(sc_grid, dtype=float)
    if sc_grid.ndim == 1:
        sc_grid = np.broadcast_to(sc_grid[None, :], (n, g))
    if sc_grid.shape != (n, g):
        raise ValueError("sc_grid must have shape (n_samples, len(t_grid)).")

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


def _compute_target_ipcw_3term_y_res(
    f_y,
    y_time,
    delta,
    m_pred,
    q_hat,
    t_grid,
    km_times,
    km_surv,
    *,
    clip_percentiles,
):
    n = len(y_time)
    g = len(t_grid)
    sc_at_y = _evaluate_sc(y_time, km_times, km_surv)
    sc_grid = _evaluate_sc(t_grid, km_times, km_surv)

    y_idx = np.searchsorted(t_grid, y_time, side="right") - 1
    y_idx = np.clip(y_idx, 0, g - 1)
    q_at_y = q_hat[np.arange(n), y_idx]

    term1 = (delta * (f_y - m_pred) + (1.0 - delta) * (q_at_y - m_pred)) / np.maximum(sc_at_y, 1e-10)

    log_sc = -np.log(np.maximum(sc_grid, 1e-10))
    d_lambda_c = np.diff(np.concatenate([[0.0], log_sc]))
    grid_weight = d_lambda_c / np.maximum(sc_grid, 1e-10)
    integrand = grid_weight[None, :] * (q_hat - m_pred[:, None])
    mask = np.arange(g)[None, :] <= y_idx[:, None]
    term2 = (integrand * mask).sum(axis=1)

    y_res = term1 - term2
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
    n = len(y_time)
    g = len(t_grid)
    sc_grid = np.asarray(sc_grid, dtype=float)
    if sc_grid.ndim == 1:
        sc_grid = np.broadcast_to(sc_grid[None, :], (n, g))
    if sc_grid.shape != (n, g):
        raise ValueError("sc_grid must have shape (n_samples, len(t_grid)).")

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


def _prepare_target_inputs(y_time, delta, *, target, horizon):
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


class _MildShrinkNCSurvivalNuisance:
    max_grid = 500

    def __init__(
        self,
        q_model,
        h_model,
        *,
        target,
        horizon,
        final_feature_mode,
        nuisance_feature_mode,
        censoring_estimator,
        q_clip,
        y_tilde_clip_quantile,
        y_res_clip_percentiles,
        event_survival_estimator="cox",
        m_pred_mode="bridge",
        survival_forest_num_trees=50,
        survival_forest_min_node_size=15,
        survival_fast_logrank=False,
        n_jobs=1,
        random_state=42,
        enforce_finite_horizon=False,
    ):
        self._q_model_template = q_model
        self._h_model_template = h_model
        self._target = target
        self._horizon = horizon
        self._final_feature_mode = final_feature_mode
        self._nuisance_feature_mode = nuisance_feature_mode
        self._censoring_estimator = censoring_estimator
        self._q_clip = q_clip
        self._y_tilde_clip_quantile = y_tilde_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._event_survival_estimator = event_survival_estimator
        self._m_pred_mode = m_pred_mode
        self._survival_forest_num_trees = int(survival_forest_num_trees)
        self._survival_forest_min_node_size = int(survival_forest_min_node_size)
        self._survival_fast_logrank = bool(survival_fast_logrank)
        self._n_jobs = int(n_jobs)
        self._random_state = int(random_state)
        self._enforce_finite_horizon = bool(enforce_finite_horizon)

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
        self._event_survival_model = None
        self._t_grid = None

    def _unpack_y(self, y):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] >= 2:
            return y[:, 0], y[:, 1]
        raise ValueError("Y must be a 2-column array [time, event].")

    def __del__(self):
        self._close_native_handles()

    def __getstate__(self):
        state = dict(self.__dict__)
        # EconML clones the nuisance object fold-by-fold. Native handles and
        # fitted state are not cloneable, so we drop them during deepcopy.
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
        state["_event_survival_model"] = None
        state["_t_grid"] = None
        return state

    def _close_native_handles(self):
        event_model = getattr(self, "_event_survival_model", None)
        if event_model is not None:
            try:
                event_model.close()
            except Exception:
                pass
            self._event_survival_model = None
        censor_model = getattr(self, "_censor_model", None)
        if censor_model is not None and censor_model.get("kind") == "survival_forest":
            try:
                censor_model["model"].close()
            except Exception:
                pass
        self._censor_model = None

    def _prepare_target_inputs(self, y_time, delta):
        if self._enforce_finite_horizon and self._horizon is None:
            raise ValueError(
                "This survival-nuisance variant requires a finite horizon. "
                "Pass horizon=<float> when enforce_finite_horizon=True."
            )
        return _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )

    @staticmethod
    def _augment_with_treatment(features, a):
        return np.column_stack([np.asarray(features, dtype=float), np.asarray(a, dtype=float).reshape(-1, 1)])

    def _predict_event_survival_curves(self, surv_features):
        if self._event_survival_estimator == "survival_forest":
            if self._event_survival_model is None:
                raise RuntimeError("Event survival forest has not been fit.")
            return _predict_native_survival_counterfactuals(
                self._event_survival_model,
                surv_features,
                self._t_grid,
            )
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

    def _compute_m_pred(self, q_pred, h1_pred, h0_pred, s_hat_1, s_hat_0):
        bridge_pred = q_pred * h1_pred + (1.0 - q_pred) * h0_pred
        if self._m_pred_mode == "bridge":
            return bridge_pred
        survival_pred = _compute_survival_based_yhat(
            s_hat_1,
            s_hat_0,
            q_pred,
            self._t_grid,
            target=self._target,
            horizon=self._horizon,
        )
        if self._m_pred_mode == "survival":
            return survival_pred
        if isinstance(self._m_pred_mode, str) and self._m_pred_mode.startswith("blend"):
            suffix = self._m_pred_mode[len("blend") :]
            suffix = suffix.lstrip("_")
            try:
                alpha = float(suffix)
            except ValueError as exc:
                raise ValueError(f"Unsupported m_pred_mode: {self._m_pred_mode}") from exc
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"Blend weight must be in [0, 1], got {alpha}.")
            return alpha * bridge_pred + (1.0 - alpha) * survival_pred
        raise ValueError(f"Unsupported m_pred_mode: {self._m_pred_mode}")

    def _predict_censoring_from_counterfactuals(self, censor_features, q_pred, *, values=None, on_grid=False):
        if self._censor_model is None or self._censor_model.get("kind") != "survival_forest":
            raise RuntimeError("Counterfactual censoring prediction requires a survival-forest censor model.")
        c1, c0 = _predict_native_survival_counterfactuals(
            self._censor_model["model"],
            censor_features,
            self._t_grid,
        )
        mixed = q_pred[:, None] * c1 + (1.0 - q_pred)[:, None] * c0
        mixed = np.clip(mixed, 0.01, 1.0)
        if on_grid:
            return mixed
        if values is None:
            raise ValueError("values must be provided when on_grid=False.")
        values = np.asarray(values, dtype=float)
        idx = np.searchsorted(self._t_grid, values, side="right") - 1
        idx = np.clip(idx, 0, len(self._t_grid) - 1)
        return mixed[np.arange(len(values)), idx]

    def train(self, is_selecting, folds, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        self._close_native_handles()
        y_time, delta = self._unpack_y(Y)
        a = np.asarray(T).ravel()
        target_inputs = self._prepare_target_inputs(y_time, delta)

        w = _ensure_2d(W)
        z = _ensure_2d(Z)
        raw_x = _recover_raw_x(X, w, z, self._final_feature_mode)
        xz, xw, surv_features, censor_features = _build_nuisance_features(
            raw_x,
            w,
            z,
            self._nuisance_feature_mode,
        )

        all_times = np.sort(np.unique(target_inputs["grid_time"]))
        if len(all_times) > self.max_grid:
            idx = np.linspace(0, len(all_times) - 1, self.max_grid, dtype=int)
            all_times = all_times[idx]
        self._t_grid = all_times

        surv_features_obs = self._augment_with_treatment(surv_features, a)
        censor_features_obs = self._augment_with_treatment(censor_features, a)

        self._censor_model = _fit_censoring_model(
            target_inputs["nuisance_time"],
            target_inputs["nuisance_delta"],
            censor_features_obs if self._censoring_estimator == "survival_forest" else censor_features,
            estimator=self._censoring_estimator,
            failure_times=self._t_grid if self._censoring_estimator == "survival_forest" else None,
            num_trees=self._survival_forest_num_trees,
            min_node_size=self._survival_forest_min_node_size,
            n_jobs=self._n_jobs,
            random_state=self._random_state,
            fast_logrank=self._survival_fast_logrank,
            sample_weight=sample_weight,
        )
        y_tilde_eval_time = target_inputs["eval_time"] if self._target == "survival.probability" else target_inputs["nuisance_time"]
        sc_for_y_tilde = _predict_censoring_survival_at_values(
            self._censor_model,
            censor_features_obs if self._censoring_estimator == "survival_forest" else censor_features,
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

        self._q_model = clone(self._q_model_template)
        self._q_model.fit(xz, a, **filter_none_kwargs(sample_weight=sample_weight))

        treated_mask = a == 1
        control_mask = a == 0

        self._h1_model = clone(self._h_model_template)
        self._h0_model = clone(self._h_model_template)

        if treated_mask.sum() > 10:
            self._h1_model.fit(
                xw[treated_mask],
                y_tilde[treated_mask],
                **filter_none_kwargs(sample_weight=None if sample_weight is None else sample_weight[treated_mask]),
            )
        if control_mask.sum() > 10:
            self._h0_model.fit(
                xw[control_mask],
                y_tilde[control_mask],
                **filter_none_kwargs(sample_weight=None if sample_weight is None else sample_weight[control_mask]),
            )

        if self._event_survival_estimator == "survival_forest":
            self._event_cox_1 = None
            self._event_cox_0 = None
            self._cox_col_names_1 = None
            self._cox_col_names_0 = None
            self._cox_keep_mask_1 = None
            self._cox_keep_mask_0 = None
            self._event_survival_model = _fit_native_survival_forest(
                surv_features_obs,
                target_inputs["nuisance_time"],
                target_inputs["nuisance_delta"],
                failure_times=self._t_grid,
                num_trees=self._survival_forest_num_trees,
                min_node_size=self._survival_forest_min_node_size,
                n_jobs=self._n_jobs,
                random_state=self._random_state,
                fast_logrank=self._survival_fast_logrank,
                sample_weight=sample_weight,
            )
        elif self._event_survival_estimator == "cox":
            self._event_survival_model = None
            self._event_cox_1, self._cox_col_names_1, self._cox_keep_mask_1 = _fit_event_cox(
                target_inputs["nuisance_time"][treated_mask],
                target_inputs["nuisance_delta"][treated_mask],
                surv_features[treated_mask],
            )
            self._event_cox_0, self._cox_col_names_0, self._cox_keep_mask_0 = _fit_event_cox(
                target_inputs["nuisance_time"][control_mask],
                target_inputs["nuisance_delta"][control_mask],
                surv_features[control_mask],
            )
        else:
            raise ValueError(f"Unsupported event_survival_estimator: {self._event_survival_estimator}")
        return self

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_time, delta = self._unpack_y(Y)
        a = np.asarray(T).ravel()
        target_inputs = self._prepare_target_inputs(y_time, delta)

        w = _ensure_2d(W)
        z = _ensure_2d(Z)
        raw_x = _recover_raw_x(X, w, z, self._final_feature_mode)
        xz, xw, surv_features, censor_features = _build_nuisance_features(
            raw_x,
            w,
            z,
            self._nuisance_feature_mode,
        )

        q_pred = self._q_model.predict_proba(xz)[:, 1]
        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        h1_pred = self._h1_model.predict(xw)
        h0_pred = self._h0_model.predict(xw)
        s_hat_1, s_hat_0 = self._predict_event_survival_curves(surv_features)
        m_pred = self._compute_m_pred(q_pred, h1_pred, h0_pred, s_hat_1, s_hat_0)
        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, self._t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, self._t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, self._t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, self._t_grid)
        q_hat = np.where((a == 1)[:, None], q_hat_1, q_hat_0)

        censor_features_obs = self._augment_with_treatment(censor_features, a)
        sc_grid = _predict_censoring_survival_on_grid(
            self._censor_model,
            censor_features_obs if self._censoring_estimator == "survival_forest" else censor_features,
            self._t_grid,
        )

        if self._target == "RMST" and self._horizon is None:
            sc_at_eval = _predict_censoring_survival_at_values(
                self._censor_model,
                censor_features_obs if self._censoring_estimator == "survival_forest" else censor_features,
                y_time,
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
            sc_at_eval = _predict_censoring_survival_at_values(
                self._censor_model,
                censor_features_obs if self._censoring_estimator == "survival_forest" else censor_features,
                target_inputs["eval_time"],
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
        a_res = (a - q_pred).reshape(-1, 1)
        return y_res, a_res

    def predict_bridge_outputs(self, X=None, W=None, Z=None):
        w = _ensure_2d(W)
        z = _ensure_2d(Z)
        raw_x = _recover_raw_x(X, w, z, self._final_feature_mode)
        xz, xw, surv_features, censor_features = _build_nuisance_features(
            raw_x,
            w,
            z,
            self._nuisance_feature_mode,
        )
        q_pred = self._q_model.predict_proba(xz)[:, 1]
        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        h1_pred = self._h1_model.predict(xw)
        h0_pred = self._h0_model.predict(xw)
        s_hat_1, s_hat_0 = self._predict_event_survival_curves(surv_features)
        m_bridge_pred = q_pred * h1_pred + (1.0 - q_pred) * h0_pred
        m_survival_pred = _compute_survival_based_yhat(
            s_hat_1,
            s_hat_0,
            q_pred,
            self._t_grid,
            target=self._target,
            horizon=self._horizon,
        )
        m_pred = self._compute_m_pred(q_pred, h1_pred, h0_pred, s_hat_1, s_hat_0)
        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, self._t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, self._t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, self._t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, self._t_grid)
        if self._censoring_estimator == "survival_forest":
            c_curve = self._predict_censoring_from_counterfactuals(
                censor_features,
                q_pred,
                on_grid=True,
            )
        else:
            c_curve = _predict_censoring_survival_on_grid(self._censor_model, censor_features, self._t_grid)
        surv1_pred = q_hat_1[:, 0]
        surv0_pred = q_hat_0[:, 0]
        return {
            "q_pred": q_pred,
            "h1_pred": h1_pred,
            "h0_pred": h0_pred,
            "m_pred": m_pred,
            "m_bridge_pred": m_bridge_pred,
            "m_survival_pred": m_survival_pred,
            "m_gap_pred": m_bridge_pred - m_survival_pred,
            "h_diff_pred": h1_pred - h0_pred,
            "surv1_pred": surv1_pred,
            "surv0_pred": surv0_pred,
            "surv_diff_pred": surv1_pred - surv0_pred,
            "qhat1_curve": q_hat_1,
            "qhat0_curve": q_hat_0,
            "s1_curve": s_hat_1,
            "s0_curve": s_hat_0,
            "c_curve": c_curve,
            "t_grid": np.asarray(self._t_grid, dtype=float),
        }

    def predict_target_pseudo_outcome(self, Y, X=None, W=None, Z=None):
        y_time, delta = self._unpack_y(Y)
        target_inputs = self._prepare_target_inputs(y_time, delta)
        w = _ensure_2d(W)
        z = _ensure_2d(Z)
        raw_x = _recover_raw_x(X, w, z, self._final_feature_mode)
        xz, _, _, censor_features = _build_nuisance_features(
            raw_x,
            w,
            z,
            self._nuisance_feature_mode,
        )
        y_tilde_eval_time = (
            target_inputs["eval_time"]
            if self._target == "survival.probability"
            else target_inputs["nuisance_time"]
        )
        if self._censoring_estimator == "survival_forest":
            q_pred = self._q_model.predict_proba(xz)[:, 1]
            q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
            sc_for_y_tilde = self._predict_censoring_from_counterfactuals(
                censor_features,
                q_pred,
                values=y_tilde_eval_time,
                on_grid=False,
            )
        else:
            sc_for_y_tilde = _predict_censoring_survival_at_values(
                self._censor_model,
                censor_features,
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
        return _clip_quantile(y_tilde, self._y_tilde_clip_quantile)


class EconmlMildShrinkNCSurvivalForest(CausalForestDML):
    """
    Finalized best-performing C3:
      - final heterogeneity features use X+W+Z
      - nuisance duplicates proxies inside q/h/survival models
      - q uses a tuned logistic bridge with clipping
      - h uses a tuned ExtraTrees bridge with mild shrink
      - IPCW pseudo-outcome and residuals are winsorized
    """

    def __init__(
        self,
        *,
        target="RMST",
        horizon=None,
        n_estimators=200,
        min_samples_leaf=20,
        cv=3,
        random_state=42,
        final_feature_mode="xwz",
        nuisance_feature_mode="dup",
        q_kind="logit",
        q_trees=300,
        q_min_samples_leaf=20,
        q_poly_degree=2,
        q_clip=0.02,
        y_tilde_clip_quantile=0.99,
        y_res_clip_percentiles=(1.0, 99.0),
        h_kind="extra",
        h_n_estimators=800,
        h_min_samples_leaf=5,
        censoring_estimator="kaplan-meier",
        event_survival_estimator="cox",
        m_pred_mode="bridge",
        survival_forest_num_trees=50,
        survival_forest_min_node_size=15,
        survival_fast_logrank=False,
        enforce_finite_horizon=False,
        n_jobs=1,
        **kwargs,
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
            n_jobs=n_jobs,
        )
        self._target = target
        self._horizon = horizon
        self._final_feature_mode = final_feature_mode
        self._nuisance_feature_mode = nuisance_feature_mode
        self._censoring_estimator = censoring_estimator
        self._event_survival_estimator = event_survival_estimator
        self._m_pred_mode = m_pred_mode
        self._survival_forest_num_trees = int(survival_forest_num_trees)
        self._survival_forest_min_node_size = int(survival_forest_min_node_size)
        self._survival_fast_logrank = bool(survival_fast_logrank)
        self._n_jobs = int(n_jobs)
        self._random_state = int(random_state)
        self._enforce_finite_horizon = bool(enforce_finite_horizon)
        self._custom_q_clip = q_clip
        self._custom_y_tilde_clip_quantile = y_tilde_clip_quantile
        self._custom_y_res_clip_percentiles = y_res_clip_percentiles

        kwargs["discrete_treatment"] = True
        kwargs["criterion"] = "het"
        kwargs.pop("model_y", None)
        kwargs.pop("model_t", None)

        super().__init__(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            cv=cv,
            random_state=random_state,
            **kwargs,
        )

    def _gen_ortho_learner_model_nuisance(self):
        return _MildShrinkNCSurvivalNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            target=self._target,
            horizon=self._horizon,
            final_feature_mode=self._final_feature_mode,
            nuisance_feature_mode=self._nuisance_feature_mode,
            censoring_estimator=self._censoring_estimator,
            q_clip=self._custom_q_clip,
            y_tilde_clip_quantile=self._custom_y_tilde_clip_quantile,
            y_res_clip_percentiles=self._custom_y_res_clip_percentiles,
            event_survival_estimator=self._event_survival_estimator,
            m_pred_mode=self._m_pred_mode,
            survival_forest_num_trees=self._survival_forest_num_trees,
            survival_forest_min_node_size=self._survival_forest_min_node_size,
            survival_fast_logrank=self._survival_fast_logrank,
            n_jobs=self._n_jobs,
            random_state=self._random_state,
            enforce_finite_horizon=self._enforce_finite_horizon,
        )

    @staticmethod
    def stack_final_features(X, W, Z, mode="xwz"):
        if mode == "x_only":
            return _ensure_2d(X)
        if mode == "xwz":
            return np.hstack([_ensure_2d(X), _ensure_2d(W), _ensure_2d(Z)])
        raise ValueError(f"Unsupported final_feature_mode: {mode}")

    def fit_survival(self, X, A, time, event, Z, W, **kwargs):
        x = np.asarray(X, dtype=float)
        y = np.asarray(time, dtype=float).ravel()
        delta = np.asarray(event, dtype=float).ravel()
        z = _ensure_2d(Z)
        w = _ensure_2d(W)
        y_packed = np.column_stack([y, delta])
        return _OrthoLearner.fit(self, y_packed, A, X=x, W=w, Z=z, **kwargs)

    def fit_components(self, X, A, time, event, Z, W, **kwargs):
        x_full = self.stack_final_features(X, W, Z, mode=self._final_feature_mode)
        return self.fit_survival(x_full, A, time, event, Z, W, **kwargs)

    def effect_from_components(self, X, W, Z):
        x_full = self.stack_final_features(X, W, Z, mode=self._final_feature_mode)
        return self.effect(x_full)
