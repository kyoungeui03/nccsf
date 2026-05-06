"""Oracle counterpart for the single-file censored Final Model.

This module is used only in synthetic simulations where the latent confounder
``U`` and the data-generating mechanism are available. It keeps the same final
forest architecture as the Final Model while replacing proxy-estimated nuisance
quantities by oracle nuisances from the DGP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from econml._ortho_learner import _OrthoLearner

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"

if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks.econml_8variant import (
    _compute_true_target_ipcw_3term_y_res,
    _true_survival_components,
    true_event_surv_on_grid,
    true_outcome_oracle,
    true_propensity_oracle,
)

try:  # pragma: no cover
    from .final_censored_model import (
        _SinglePassBridgeFeatureCensoredSurvivalForest,
        _build_final_features_full,
        _compute_q_from_s,
        _compute_survival_probability_q_from_s,
        _ensure_2d,
        _prepare_target_inputs,
    )
except ImportError:  # pragma: no cover
    from censored.final_censored_model import (  # type: ignore
        _SinglePassBridgeFeatureCensoredSurvivalForest,
        _build_final_features_full,
        _compute_q_from_s,
        _compute_survival_probability_q_from_s,
        _ensure_2d,
        _prepare_target_inputs,
    )


class _FinalCensoredOracleNuisance:
    """Oracle nuisance layer for the single-file censored Final Model."""

    max_grid = 500

    def __init__(self, *, cfg, dgp, p_x, target, horizon, q_clip, y_res_clip_percentiles):
        self._cfg = cfg
        self._dgp = dgp
        self._p_x = int(p_x)
        self._target = target
        self._horizon = horizon
        self._q_clip = float(q_clip)
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._t_grid = None

    @staticmethod
    def _unpack_y(y):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] >= 2:
            return y[:, 0], y[:, 1]
        raise ValueError("Y must be a 2-column array [time, event].")

    def _set_time_grid(self, y_time, delta):
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
        return target_inputs

    def _oracle_components(self, x_full, u_vec):
        if self._t_grid is None:
            raise RuntimeError("Oracle nuisance must be trained before prediction.")

        x_base = np.asarray(x_full, dtype=float)[:, : self._p_x]
        q_pred = true_propensity_oracle(x_base, u_vec, self._dgp, self._cfg)
        h0_pred, h1_pred = true_outcome_oracle(
            x_base,
            u_vec,
            self._cfg,
            self._dgp,
            target=self._target,
            horizon=self._horizon,
        )
        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        m_pred = q_pred * h1_pred + (1.0 - q_pred) * h0_pred

        s_hat_1 = true_event_surv_on_grid(
            x_base,
            u_vec,
            np.ones(x_base.shape[0], dtype=float),
            self._t_grid,
            self._cfg,
            self._dgp,
        )
        s_hat_0 = true_event_surv_on_grid(
            x_base,
            u_vec,
            np.zeros(x_base.shape[0], dtype=float),
            self._t_grid,
            self._cfg,
            self._dgp,
        )

        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, self._t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, self._t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, self._t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, self._t_grid)

        return {
            "q_pred": q_pred,
            "h1_pred": h1_pred,
            "h0_pred": h0_pred,
            "m_pred": m_pred,
            "q_hat_1": q_hat_1,
            "q_hat_0": q_hat_0,
        }

    def train(self, is_selecting, folds, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        del is_selecting, folds, T, X, W, Z, sample_weight, groups
        y_time, delta = self._unpack_y(Y)
        self._set_time_grid(y_time, delta)
        return self

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        del Z, sample_weight, groups

        y_time, delta = self._unpack_y(Y)
        a = np.asarray(T).ravel()
        x_full = np.asarray(X, dtype=float)
        u_vec = _ensure_2d(W)[:, 0]

        target_inputs = _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )
        oracle = self._oracle_components(x_full, u_vec)
        q_pred = oracle["q_pred"]
        h1_pred = oracle["h1_pred"]
        h0_pred = oracle["h0_pred"]
        m_pred = oracle["m_pred"]
        q_hat_1 = oracle["q_hat_1"]
        q_hat_0 = oracle["q_hat_0"]

        t_grid, surv_c, hazard_c, sc_at_y = _true_survival_components(
            x_full[:, : self._p_x],
            u_vec,
            target_inputs["eval_time"],
            self._t_grid,
            self._cfg,
            self._dgp,
        )

        q_hat = np.where((a == 1)[:, None], q_hat_1, q_hat_0)
        y_res = _compute_true_target_ipcw_3term_y_res(
            target_inputs["f_y"],
            target_inputs["eval_time"],
            target_inputs["eval_delta"],
            m_pred,
            q_hat,
            t_grid,
            surv_c,
            hazard_c,
            sc_at_y,
            clip_percentiles=self._y_res_clip_percentiles,
        )

        a_res = (a - q_pred).reshape(-1, 1)
        surv1_pred = q_hat_1[:, 0]
        surv0_pred = q_hat_0[:, 0]
        return (
            y_res,
            a_res,
            q_pred,
            h1_pred,
            h0_pred,
            m_pred,
            surv1_pred,
            surv0_pred,
            surv1_pred - surv0_pred,
        )

    def predict_bridge_outputs(self, *, X, U):
        x_full = np.asarray(X, dtype=float)
        u_vec = _ensure_2d(U)[:, 0]
        oracle = self._oracle_components(x_full, u_vec)
        q_hat_1 = oracle["q_hat_1"]
        q_hat_0 = oracle["q_hat_0"]
        return {
            "q_pred": oracle["q_pred"],
            "h1_pred": oracle["h1_pred"],
            "h0_pred": oracle["h0_pred"],
            "m_pred": oracle["m_pred"],
            "surv1_pred": q_hat_1[:, 0],
            "surv0_pred": q_hat_0[:, 0],
            "surv_diff_pred": q_hat_1[:, 0] - q_hat_0[:, 0],
        }


class FinalCensoredOracle(_SinglePassBridgeFeatureCensoredSurvivalForest):
    """Oracle counterpart of the single-file censored Final Model."""

    def __init__(self, *, cfg, dgp, p_x, **kwargs):
        self._benchmark_cfg = cfg
        self._benchmark_dgp = dgp
        self._benchmark_p_x = int(p_x)
        self._oracle_feature_nuisance = None
        super().__init__(**kwargs)

    def _gen_ortho_learner_model_nuisance(self):
        return _FinalCensoredOracleNuisance(
            cfg=self._benchmark_cfg,
            dgp=self._benchmark_dgp,
            p_x=self._benchmark_p_x,
            target=self._target,
            horizon=self._horizon,
            q_clip=self._custom_q_clip,
            y_res_clip_percentiles=self._custom_y_res_clip_percentiles,
        )

    def fit_oracle(self, X, A, time, event, U, **kwargs):
        x = np.asarray(X, dtype=float)
        y = np.asarray(time, dtype=float).ravel()
        delta = np.asarray(event, dtype=float).ravel()
        a = np.asarray(A, dtype=float).ravel()
        u = _ensure_2d(U)
        y_packed = np.column_stack([y, delta])
        z_dummy = np.zeros((len(y), 1), dtype=float)

        self._raw_w_for_final = np.zeros((len(x), 0), dtype=float)
        self._raw_z_for_final = np.zeros((len(x), 0), dtype=float)
        _OrthoLearner.fit(self, y_packed, a, X=x, W=u, Z=z_dummy, **kwargs)

        self._oracle_feature_nuisance = _FinalCensoredOracleNuisance(
            cfg=self._benchmark_cfg,
            dgp=self._benchmark_dgp,
            p_x=self._benchmark_p_x,
            target=self._target,
            horizon=self._horizon,
            q_clip=self._custom_q_clip,
            y_res_clip_percentiles=self._custom_y_res_clip_percentiles,
        )
        self._oracle_feature_nuisance.train(
            False,
            None,
            y_packed,
            a,
            X=x,
            W=u,
            Z=z_dummy,
        )
        return self

    def effect_oracle(self, X, U):
        if self._oracle_feature_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_oracle.")

        x_base = _ensure_2d(X).astype(float)
        u = _ensure_2d(U).astype(float)
        x_oracle = np.column_stack([x_base, u])
        bridge = self._oracle_feature_nuisance.predict_bridge_outputs(X=x_oracle, U=u)
        x_final = _build_final_features_full(
            x_oracle,
            np.zeros((len(x_oracle), 0), dtype=float),
            np.zeros((len(x_oracle), 0), dtype=float),
            bridge,
        )
        return np.asarray(self.effect_on_final_features(x_final), dtype=float)


SingleFileFinalOracleCensoredSurvivalForest = FinalCensoredOracle


__all__ = [
    "FinalCensoredOracle",
    "SingleFileFinalOracleCensoredSurvivalForest",
]
