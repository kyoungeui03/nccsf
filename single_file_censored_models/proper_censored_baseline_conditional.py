"""Single-file conditional-censoring variant of ProperNoPCICensoredSurvivalForest.

This module keeps the same no-PCI residualization and survival-only final-stage
features as the marginal Proper baseline, but replaces the marginal censoring
nuisance with a conditional Cox model for

    S_C(t | X, W, Z, A).

The treatment/outcome/survival nuisance branches still follow the proper
baseline convention `observed_only=True`, i.e. the proxy blocks are zeroed for
those nuisances. The raw observed proxies are used only by the conditional
censoring nuisance and by the final feature block `[X, W, Z, surv1, surv0,
surv_diff]`.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone

try:  # pragma: no cover
    from .final_censored_model_conditional import (  # noqa: E402
        _fit_conditional_censoring_model,
        _predict_conditional_censoring_survival_at_values,
        _predict_conditional_censoring_survival_on_grid,
    )
    from .proper_censored_baseline import (  # noqa: E402
        ProperNoPCICensoredSurvivalForest,
        _ProperNoPCICensoredDML,
        _ProperNoPCINuisance,
        _build_final_features_raw_surv,
        _build_nuisance_features,
        _compute_ipcw_3term_y_res_from_survival,
        _compute_q_from_s,
        _compute_survival_probability_q_from_s,
        _compute_target_ipcw_3term_y_res_from_survival,
        _compute_target_pseudo_outcome_from_sc,
        _ensure_2d,
        _prepare_target_inputs,
    )
except ImportError:  # pragma: no cover
    from single_file_censored_models.final_censored_model_conditional import (  # type: ignore  # noqa: E402
        _fit_conditional_censoring_model,
        _predict_conditional_censoring_survival_at_values,
        _predict_conditional_censoring_survival_on_grid,
    )
    from single_file_censored_models.proper_censored_baseline import (  # type: ignore  # noqa: E402
        ProperNoPCICensoredSurvivalForest,
        _ProperNoPCICensoredDML,
        _ProperNoPCINuisance,
        _build_final_features_raw_surv,
        _build_nuisance_features,
        _compute_ipcw_3term_y_res_from_survival,
        _compute_q_from_s,
        _compute_survival_probability_q_from_s,
        _compute_target_ipcw_3term_y_res_from_survival,
        _compute_target_pseudo_outcome_from_sc,
        _ensure_2d,
        _prepare_target_inputs,
    )


class _ConditionalProperNoPCINuisance(_ProperNoPCINuisance):
    """Proper-no-PCI nuisance layer with conditional censoring survival."""

    def __init__(self, *args, observed_only=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._observed_only = bool(observed_only)

    @staticmethod
    def _build_censor_features(x, raw_w, raw_z):
        return np.column_stack([_ensure_2d(x), _ensure_2d(raw_w), _ensure_2d(raw_z)])

    @staticmethod
    def _augment_with_treatment(features, a):
        return np.column_stack([np.asarray(features, dtype=float), np.asarray(a, dtype=float).reshape(-1, 1)])

    def _prepare_no_pci_inputs(self, raw_w, raw_z):
        w = _ensure_2d(raw_w).astype(float)
        z = _ensure_2d(raw_z).astype(float)
        if self._observed_only:
            w = np.zeros_like(w)
            z = np.zeros_like(z)
        return w, z

    def train(self, is_selecting, folds, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        del is_selecting, folds, groups

        y_time, delta = self._unpack_y(Y)
        a = np.asarray(T, dtype=float).ravel()
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w, z = self._prepare_no_pci_inputs(raw_w, raw_z)

        target_inputs = _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )
        xq, xh, surv_features = _build_nuisance_features(x, w, z, self._nuisance_feature_mode)
        censor_features = self._build_censor_features(x, raw_w, raw_z)
        censor_features_obs = self._augment_with_treatment(censor_features, a)

        all_times = np.sort(np.unique(target_inputs["grid_time"]))
        if len(all_times) > self.max_grid:
            idx = np.linspace(0, len(all_times) - 1, self.max_grid, dtype=int)
            all_times = all_times[idx]
        self._t_grid = all_times

        self._censor_model = _fit_conditional_censoring_model(
            target_inputs["nuisance_time"],
            target_inputs["nuisance_delta"],
            censor_features_obs,
            estimator=self._censoring_estimator,
        )

        y_tilde_eval_time = (
            target_inputs["eval_time"]
            if self._target == "survival.probability"
            else target_inputs["nuisance_time"]
        )
        sc_for_y_tilde = _predict_conditional_censoring_survival_at_values(
            self._censor_model,
            censor_features_obs,
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
        if self._y_tilde_clip_quantile is not None:
            y_tilde = np.clip(
                y_tilde,
                np.quantile(y_tilde, 1.0 - self._y_tilde_clip_quantile),
                np.quantile(y_tilde, self._y_tilde_clip_quantile),
            )

        self._q_model = clone(self._q_model_template)
        self._q_model.fit(xq, a)

        treated_mask = a == 1
        control_mask = a == 0
        self._h1_model = clone(self._h_model_template)
        self._h0_model = clone(self._h_model_template)

        if treated_mask.sum() <= 10 or control_mask.sum() <= 10:
            raise ValueError("Each treatment arm must have more than 10 samples per fold for this model.")

        self._h1_model.fit(xh[treated_mask], y_tilde[treated_mask])
        self._h0_model.fit(xh[control_mask], y_tilde[control_mask])

        self._fit_event_survival_models(
            target_inputs["nuisance_time"],
            target_inputs["nuisance_delta"],
            surv_features,
            a,
        )
        return self

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        del sample_weight, groups

        y_time, delta = self._unpack_y(Y)
        a = np.asarray(T, dtype=float).ravel()
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w, z = self._prepare_no_pci_inputs(raw_w, raw_z)

        target_inputs = _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )
        xq, xh, surv_features = _build_nuisance_features(x, w, z, self._nuisance_feature_mode)
        censor_features = self._build_censor_features(x, raw_w, raw_z)
        censor_features_obs = self._augment_with_treatment(censor_features, a)

        q_pred = self._q_model.predict_proba(xq)[:, 1]
        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        h1_pred = self._h1_model.predict(xh)
        h0_pred = self._h0_model.predict(xh)
        s_hat_1, s_hat_0 = self._predict_event_survival_curves(surv_features)
        m_pred = self._compute_m_pred(q_pred, h1_pred, h0_pred, s_hat_1, s_hat_0)

        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, self._t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, self._t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, self._t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, self._t_grid)

        q_hat = np.where((a == 1)[:, None], q_hat_1, q_hat_0)
        sc_grid = _predict_conditional_censoring_survival_on_grid(
            self._censor_model,
            censor_features_obs,
            self._t_grid,
        )

        if self._target == "RMST" and self._horizon is None:
            sc_at_eval = _predict_conditional_censoring_survival_at_values(
                self._censor_model,
                censor_features_obs,
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
            sc_at_eval = _predict_conditional_censoring_survival_at_values(
                self._censor_model,
                censor_features_obs,
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
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w, z = self._prepare_no_pci_inputs(raw_w, raw_z)
        xq, xh, surv_features = _build_nuisance_features(x, w, z, self._nuisance_feature_mode)

        q_pred = self._q_model.predict_proba(xq)[:, 1]
        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        h1_pred = self._h1_model.predict(xh)
        h0_pred = self._h0_model.predict(xh)
        s_hat_1, s_hat_0 = self._predict_event_survival_curves(surv_features)
        m_pred = self._compute_m_pred(q_pred, h1_pred, h0_pred, s_hat_1, s_hat_0)

        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, self._t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, self._t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, self._t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, self._t_grid)

        surv1_pred = q_hat_1[:, 0]
        surv0_pred = q_hat_0[:, 0]
        return {
            "surv1_pred": surv1_pred,
            "surv0_pred": surv0_pred,
            "surv_diff_pred": surv1_pred - surv0_pred,
            "m_pred": m_pred,
        }


class _ConditionalProperBridgeOutputNuisance(_ConditionalProperNoPCINuisance):
    """Bridge-output wrapper for the conditional Proper-no-PCI path."""

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_res, a_res = super().predict(Y, T, X=X, W=W, Z=Z, sample_weight=sample_weight, groups=groups)
        bridge = super().predict_bridge_outputs(X=X, W=W, Z=Z)
        return (
            y_res,
            a_res,
            bridge["surv1_pred"],
            bridge["surv0_pred"],
            bridge["surv_diff_pred"],
        )


class _ConditionalProperNoPCICensoredDML(_ProperNoPCICensoredDML):
    """Proper-no-PCI CausalForestDML with conditional censoring nuisance."""

    def __init__(self, *args, observed_only=True, **kwargs):
        self._observed_only = bool(observed_only)
        super().__init__(*args, **kwargs)

    def _gen_ortho_learner_model_nuisance(self):
        return _ConditionalProperBridgeOutputNuisance(
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
            observed_only=self._observed_only,
        )


class ProperNoPCIConditionalCensoredSurvivalForest(ProperNoPCICensoredSurvivalForest):
    """Proper-no-PCI censored baseline with conditional censoring S_C(t|X,W,Z,A)."""

    def __init__(self, *args, censoring_estimator="cox", **kwargs):
        if censoring_estimator != "cox":
            raise ValueError(
                "ProperNoPCIConditionalCensoredSurvivalForest currently supports only censoring_estimator='cox'."
            )
        super().__init__(*args, censoring_estimator=censoring_estimator, **kwargs)

    def _make_feature_nuisance(self):
        return _ConditionalProperNoPCINuisance(
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
            observed_only=self._observed_only,
        )

    def fit_components(self, X, A, time, event, Z, W):
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)

        self._dml_model = _ConditionalProperNoPCICensoredDML(
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
            observed_only=self._observed_only,
        )
        self._dml_model._raw_w_for_final = raw_w
        self._dml_model._raw_z_for_final = raw_z
        self._dml_model.fit_survival(x, A, time, event, raw_z, raw_w)

        self._train_x = np.asarray(x, dtype=float).copy()
        self._train_w = np.asarray(raw_w, dtype=float).copy()
        self._train_z = np.asarray(raw_z, dtype=float).copy()
        self._train_x_final = self._dml_model.training_x_final()

        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        self._feature_nuisance = self._make_feature_nuisance()
        self._feature_nuisance.train(
            False,
            None,
            y_packed,
            np.asarray(A, dtype=float).ravel(),
            X=x,
            W=raw_w,
            Z=raw_z,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._dml_model is None or self._feature_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")

        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)

        bridge = self._feature_nuisance.predict_bridge_outputs(
            X=x,
            W=raw_w,
            Z=raw_z,
        )
        x_final = _build_final_features_raw_surv(x, raw_w, raw_z, bridge)
        return self._dml_model.effect_on_final_features(x_final)


ProperConditionalCensoredBaseline = ProperNoPCIConditionalCensoredSurvivalForest


__all__ = [
    "ProperNoPCIConditionalCensoredSurvivalForest",
    "ProperConditionalCensoredBaseline",
]
