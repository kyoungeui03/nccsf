from __future__ import annotations

import numpy as np
import pandas as pd
from econml._ortho_learner import _OrthoLearner
from econml.dml import CausalForestDML
from econml.utilities import filter_none_kwargs
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression


def _ensure_2d(array):
    array = np.asarray(array)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _fit_kaplan_meier_censoring(time, event):
    censor_indicator = 1 - np.asarray(event).astype(int)
    kmf = KaplanMeierFitter()
    kmf.fit(np.asarray(time, dtype=float), event_observed=censor_indicator)
    return np.asarray(kmf.survival_function_.index, dtype=float), np.asarray(
        kmf.survival_function_["KM_estimate"], dtype=float
    )


def _evaluate_sc(values, km_times, km_surv, clip_min=0.01):
    idx = np.searchsorted(km_times, np.asarray(values, dtype=float), side="right") - 1
    idx = np.clip(idx, 0, len(km_surv) - 1)
    return np.clip(km_surv[idx], clip_min, 1.0)


def _compute_ipcw_pseudo_outcome(y_time, delta, km_times, km_surv):
    sc = _evaluate_sc(y_time, km_times, km_surv)
    return np.asarray(y_time, dtype=float) * np.asarray(delta, dtype=float) / np.maximum(sc, 1e-10)


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


def _fit_event_cox(y_time, delta, features, penalizer=0.01):
    col_names = [f"cxf{j}" for j in range(features.shape[1])]
    train_df = pd.DataFrame(np.asarray(features, dtype=float), columns=col_names)
    train_df["_duration"] = np.asarray(y_time, dtype=float)
    train_df["_event"] = np.asarray(delta, dtype=float)
    cox = CoxPHFitter(penalizer=penalizer)
    cox.fit(train_df, duration_col="_duration", event_col="_event")
    return cox, col_names


def _predict_s_on_grid(cox, col_names, features, t_grid):
    pred_df = pd.DataFrame(np.asarray(features, dtype=float), columns=col_names)
    surv = cox.predict_survival_function(pred_df, times=np.asarray(t_grid, dtype=float))
    return np.clip(surv.values.T, 1e-10, 1.0)


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


class _MildShrinkNCSurvivalNuisance:
    max_grid = 500

    def __init__(
        self,
        q_model,
        h_model,
        *,
        q_clip,
        y_tilde_clip_quantile,
        y_res_clip_percentiles,
    ):
        self._q_model_template = q_model
        self._h_model_template = h_model
        self._q_clip = q_clip
        self._y_tilde_clip_quantile = y_tilde_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles

        self._q_model = None
        self._h1_model = None
        self._h0_model = None
        self._km_times = None
        self._km_surv = None
        self._event_cox_1 = None
        self._event_cox_0 = None
        self._cox_col_names = None
        self._t_grid = None

    def _unpack_y(self, y):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] >= 2:
            return y[:, 0], y[:, 1]
        raise ValueError("Y must be a 2-column array [time, event].")

    def train(self, is_selecting, folds, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_time, delta = self._unpack_y(Y)
        a = np.asarray(T).ravel()

        w = _ensure_2d(W)
        z = _ensure_2d(Z)
        xz = np.column_stack([X, z])
        xw = np.column_stack([X, w])

        self._km_times, self._km_surv = _fit_kaplan_meier_censoring(y_time, delta)
        y_tilde = _compute_ipcw_pseudo_outcome(y_time, delta, self._km_times, self._km_surv)
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

        surv_features = np.column_stack([X, w, z])
        self._event_cox_1, self._cox_col_names = _fit_event_cox(
            y_time[treated_mask],
            delta[treated_mask],
            surv_features[treated_mask],
        )
        self._event_cox_0, _ = _fit_event_cox(
            y_time[control_mask],
            delta[control_mask],
            surv_features[control_mask],
        )

        all_times = np.sort(np.unique(y_time))
        if len(all_times) > self.max_grid:
            idx = np.linspace(0, len(all_times) - 1, self.max_grid, dtype=int)
            all_times = all_times[idx]
        self._t_grid = all_times
        return self

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_time, delta = self._unpack_y(Y)
        a = np.asarray(T).ravel()

        w = _ensure_2d(W)
        z = _ensure_2d(Z)
        xz = np.column_stack([X, z])
        xw = np.column_stack([X, w])

        q_pred = self._q_model.predict_proba(xz)[:, 1]
        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        h1_pred = self._h1_model.predict(xw)
        h0_pred = self._h0_model.predict(xw)
        m_pred = q_pred * h1_pred + (1.0 - q_pred) * h0_pred

        surv_features = np.column_stack([X, w, z])
        s_hat_1 = _predict_s_on_grid(self._event_cox_1, self._cox_col_names, surv_features, self._t_grid)
        s_hat_0 = _predict_s_on_grid(self._event_cox_0, self._cox_col_names, surv_features, self._t_grid)
        q_hat_1 = _compute_q_from_s(s_hat_1, self._t_grid)
        q_hat_0 = _compute_q_from_s(s_hat_0, self._t_grid)
        q_hat = np.where((a == 1)[:, None], q_hat_1, q_hat_0)

        y_res = _compute_ipcw_3term_y_res(
            y_time,
            delta,
            m_pred,
            q_hat,
            self._t_grid,
            self._km_times,
            self._km_surv,
            clip_percentiles=self._y_res_clip_percentiles,
        )
        a_res = (a - q_pred).reshape(-1, 1)
        return y_res, a_res


class EconmlMildShrinkNCSurvivalForest(CausalForestDML):
    """
    Finalized best-performing C3:
      - final heterogeneity features use X+W+Z
      - nuisance duplicates proxies inside q/h/survival models
      - q uses logistic regression with clipping
      - h uses RF with mild shrink
      - IPCW pseudo-outcome and residuals are winsorized
    """

    def __init__(
        self,
        *,
        n_estimators=200,
        min_samples_leaf=20,
        cv=3,
        random_state=42,
        q_clip=0.02,
        y_tilde_clip_quantile=0.99,
        y_res_clip_percentiles=(1.0, 99.0),
        h_n_estimators=300,
        h_min_samples_leaf=20,
        n_jobs=1,
        **kwargs,
    ):
        self._q_model_template = LogisticRegression(max_iter=2000)
        self._h_model_template = RandomForestRegressor(
            n_estimators=h_n_estimators,
            min_samples_leaf=h_min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )
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
            q_clip=self._custom_q_clip,
            y_tilde_clip_quantile=self._custom_y_tilde_clip_quantile,
            y_res_clip_percentiles=self._custom_y_res_clip_percentiles,
        )

    @staticmethod
    def stack_final_features(X, W, Z):
        return np.hstack([_ensure_2d(X), _ensure_2d(W), _ensure_2d(Z)])

    def fit_survival(self, X, A, time, event, Z, W, **kwargs):
        x = np.asarray(X, dtype=float)
        y = np.asarray(time, dtype=float).ravel()
        delta = np.asarray(event, dtype=float).ravel()
        z = _ensure_2d(Z)
        w = _ensure_2d(W)
        y_packed = np.column_stack([y, delta])
        return _OrthoLearner.fit(self, y_packed, A, X=x, W=w, Z=z, **kwargs)

    def fit_components(self, X, A, time, event, Z, W, **kwargs):
        x_full = self.stack_final_features(X, W, Z)
        return self.fit_survival(x_full, A, time, event, Z, W, **kwargs)

    def effect_from_components(self, X, W, Z):
        x_full = self.stack_final_features(X, W, Z)
        return self.effect(x_full)
