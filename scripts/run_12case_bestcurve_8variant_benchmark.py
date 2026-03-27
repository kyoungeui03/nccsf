#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.utilities import filter_none_kwargs
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks.econml_8variant import (  # noqa: E402
    CASE_SPECS,
    SURV_FLOOR,
    _BenchmarkNCSurvivalNuisance,
    _BenchmarkOracleSurvivalNuisance,
    _build_true_y_tilde,
    _cap_time_grid,
    _compute_true_ipcw_3term_y_res,
    _compute_true_target_ipcw_3term_y_res,
    _evaluate_predictions,
    _make_forest_kwargs,
    _true_survival_components,
    evaluate_r_csf_variant,
    prepare_case,
    render_case_table_png,
    true_censoring_on_grid,
    true_event_surv_on_grid,
    true_outcome_nc,
    true_outcome_oracle,
    true_propensity_nc,
    true_propensity_oracle,
)
from grf.methods import (  # noqa: E402
    BestCurveLocalCensoredPCISurvivalForest,
    BestCurveLocalObservedCensoredSurvivalForest,
)
from grf.methods.econml_bridge_summary_survival import _BaseTwoStageSummarySurvivalForest  # noqa: E402
from grf.methods.econml_mild_shrink import (  # noqa: E402
    _build_nuisance_features,
    _clip_quantile,
    _compute_ipcw_3term_y_res_from_survival,
    _compute_q_from_s,
    _compute_survival_probability_q_from_s,
    _compute_target_ipcw_3term_y_res_from_survival,
    _compute_target_pseudo_outcome_from_sc,
    _ensure_2d,
    _fit_censoring_model,
    _fit_event_cox,
    _predict_censoring_survival_at_values,
    _predict_censoring_survival_on_grid,
    _predict_s_on_grid,
    _prepare_target_inputs,
)

matplotlib.use("Agg")


VARIANT_SPECS = [
    ("A1  Oracle BestCurve (all true)", "oracle", dict(true_surv=True, true_qh=True)),
    ("A2  Oracle BestCurve (true surv, est q/h)", "oracle", dict(true_surv=True, true_qh=False)),
    ("A3  Oracle BestCurve (all estimated)", "oracle", dict(true_surv=False, true_qh=False)),
    ("R-CSF Baseline", "b2", {}),
    ("C1  BestCurve NC-CSF (all true)", "nc", dict(true_surv=True, true_qh=True)),
    ("C2  BestCurve NC-CSF (true surv, est q/h)", "nc", dict(true_surv=True, true_qh=False)),
    ("C3  BestCurve NC-CSF (all estimated)", "nc", dict(true_surv=False, true_qh=False)),
    ("D2  BestCurve no-PCI baseline", "d2", {}),
]

MODEL_ORDER = [name for name, _, _ in VARIANT_SPECS]
SHORT_LABELS = ["A1", "A2", "A3", "B2", "C1", "C2", "C3", "D2"]
MODEL_COLORS = {
    "A1  Oracle BestCurve (all true)": "#264653",
    "A2  Oracle BestCurve (true surv, est q/h)": "#287271",
    "A3  Oracle BestCurve (all estimated)": "#2a9d8f",
    "R-CSF Baseline": "#e76f51",
    "C1  BestCurve NC-CSF (all true)": "#7b61ff",
    "C2  BestCurve NC-CSF (true surv, est q/h)": "#6d597a",
    "C3  BestCurve NC-CSF (all estimated)": "#355070",
    "D2  BestCurve no-PCI baseline": "#9c6644",
}

CASE_LABELS = {
    1: "lin / lin / info / strong / bene / large",
    2: "lin / lin / info / weak / harm / small",
    3: "lin / lin / weakproxy / strong / harm / mod",
    4: "lin / lin / weakproxy / weak / near0 / harm",
    5: "lin / nonlin / info / strong / bene / large",
    6: "lin / nonlin / info / weak / harm / small",
    7: "lin / nonlin / weakproxy / strong / harm / mod",
    8: "lin / nonlin / weakproxy / weak / near0 / harm",
    9: "nonlin / nonlin / info / strong / bene / large",
    10: "nonlin / nonlin / info / weak / harm / small",
    11: "nonlin / nonlin / weakproxy / strong / harm / mod",
    12: "nonlin / nonlin / weakproxy / weak / near0 / harm",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 12-case censored BestCurve 8-variant benchmark.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benchmark_bestcurve_8variant_12case",
    )
    parser.add_argument("--num-trees-b2", type=int, default=200)
    parser.add_argument("--target", choices=["RMST", "survival.probability"], default="RMST")
    parser.add_argument("--horizon-quantile", type=float, default=0.60)
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p-w", type=int, default=1)
    parser.add_argument("--p-z", type=int, default=1)
    return parser.parse_args()


def _format_case_title(case_spec, cfg):
    base = str(case_spec["title"]).split(", n=", 1)[0]
    censor_pct = int(round(100 * float(cfg.target_censor_rate)))
    return (
        f"{base}, n={cfg.n}, p_x={cfg.p_x}, p_w={cfg.p_w}, "
        f"p_z={cfg.p_z}, seed={cfg.seed}, censoring rate={censor_pct}%"
    )


def _case_with_overrides(case_spec, *, n, p_w, p_z):
    case_copy = dict(case_spec)
    cfg_updates = dict(case_spec["cfg"])
    cfg_updates.update({"n": int(n), "p_w": int(p_w), "p_z": int(p_z)})
    case_copy["cfg"] = cfg_updates
    return case_copy


class _BenchmarkBestCurveNCNuisance:
    def __init__(
        self,
        cfg,
        dgp,
        p_x,
        z_proxy_dim,
        q_model,
        h_model,
        *,
        true_surv,
        true_qh,
        target,
        horizon,
        q_clip,
        y_tilde_clip_quantile,
        y_res_clip_percentiles,
        nuisance_feature_mode,
        censoring_estimator,
    ):
        self._cfg = cfg
        self._dgp = dgp
        self._p_x = p_x
        self._z_proxy_dim = z_proxy_dim
        self._true_surv = true_surv
        self._true_qh = true_qh
        self._target = target
        self._horizon = horizon
        self._q_model_template = q_model
        self._h_model_template = h_model
        self._q_clip = q_clip
        self._y_tilde_clip_quantile = y_tilde_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._nuisance_feature_mode = nuisance_feature_mode
        self._censoring_estimator = censoring_estimator

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

    @staticmethod
    def _unpack_y(y):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] >= 2:
            return y[:, 0], y[:, 1]
        raise ValueError("Y must be a 2-column array [time, event].")

    def _split_inputs(self, x, w, z):
        raw_x = np.asarray(x, dtype=float)
        w_proxy = _ensure_2d(w).astype(float)
        z_bundle = _ensure_2d(z).astype(float)
        z_proxy = z_bundle[:, : self._z_proxy_dim]
        u_cols = z_bundle[:, self._z_proxy_dim :]
        u_vec = None if u_cols.shape[1] == 0 else u_cols[:, 0]
        q_features, h_features, surv_features, censor_features = _build_nuisance_features(
            raw_x,
            w_proxy,
            z_proxy,
            self._nuisance_feature_mode,
        )
        return raw_x, w_proxy, z_proxy, u_vec, q_features, h_features, surv_features, censor_features

    def train(self, is_selecting, folds, y, t, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_time, delta = self._unpack_y(y)
        a = np.asarray(t).ravel()
        raw_x, w_proxy, z_proxy, u_vec, q_features, h_features, surv_features, censor_features = self._split_inputs(X, W, Z)
        target_inputs = _prepare_target_inputs(y_time, delta, target=self._target, horizon=self._horizon)

        if self._true_surv:
            if u_vec is None:
                raise ValueError("True survival nuisance requires bundled U in Z.")
            y_tilde = _build_true_y_tilde(
                raw_x,
                u_vec,
                y_time,
                delta,
                self._cfg,
                self._dgp,
                target=self._target,
                horizon=self._horizon,
            )
            self._t_grid = _cap_time_grid(target_inputs["grid_time"])
        else:
            self._censor_model = _fit_censoring_model(
                target_inputs["nuisance_time"],
                target_inputs["nuisance_delta"],
                censor_features,
                estimator=self._censoring_estimator,
            )
            sc_at_eval = _predict_censoring_survival_at_values(
                self._censor_model,
                censor_features,
                target_inputs["eval_time"],
                clip_min=SURV_FLOOR,
            )
            y_tilde = _compute_target_pseudo_outcome_from_sc(
                y_time=y_time,
                horizon=self._horizon,
                target=self._target,
                nuisance_time=target_inputs["nuisance_time"],
                nuisance_delta=target_inputs["nuisance_delta"],
                sc_at_eval=sc_at_eval,
            )
            self._t_grid = _cap_time_grid(target_inputs["grid_time"])

        y_tilde = _clip_quantile(y_tilde, self._y_tilde_clip_quantile)

        if not self._true_qh:
            self._q_model = clone(self._q_model_template)
            self._q_model.fit(q_features, a, **filter_none_kwargs(sample_weight=sample_weight))

            treated_mask = a == 1
            control_mask = a == 0
            self._h1_model = clone(self._h_model_template)
            self._h0_model = clone(self._h_model_template)

            if treated_mask.sum() > 10:
                self._h1_model.fit(
                    h_features[treated_mask],
                    y_tilde[treated_mask],
                    **filter_none_kwargs(sample_weight=None if sample_weight is None else sample_weight[treated_mask]),
                )
            if control_mask.sum() > 10:
                self._h0_model.fit(
                    h_features[control_mask],
                    y_tilde[control_mask],
                    **filter_none_kwargs(sample_weight=None if sample_weight is None else sample_weight[control_mask]),
                )

        if not self._true_surv:
            treated_mask = a == 1
            control_mask = a == 0
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

        return self

    def _predict_q_h(self, raw_x, w_proxy, z_proxy, u_vec, q_features, h_features):
        if self._true_qh:
            if u_vec is None:
                raise ValueError("True q/h requires bundled U in Z.")
            q_pred = true_propensity_nc(z_proxy, raw_x, self._dgp, self._cfg)
            h0_pred, h1_pred = true_outcome_nc(
                w_proxy,
                raw_x,
                self._cfg,
                self._dgp,
                target=self._target,
                horizon=self._horizon,
            )
        else:
            q_pred = self._q_model.predict_proba(q_features)[:, 1]
            h1_pred = self._h1_model.predict(h_features)
            h0_pred = self._h0_model.predict(h_features)
        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        m_pred = q_pred * h1_pred + (1.0 - q_pred) * h0_pred
        return q_pred, h1_pred, h0_pred, m_pred

    def _predict_survival_blocks(self, raw_x, w_proxy, z_proxy, u_vec, surv_features, censor_features):
        if self._true_surv:
            if u_vec is None:
                raise ValueError("True survival nuisance requires bundled U in Z.")
            t_grid = self._t_grid
            s_hat_1 = true_event_surv_on_grid(raw_x, u_vec, np.ones(raw_x.shape[0]), t_grid, self._cfg, self._dgp)
            s_hat_0 = true_event_surv_on_grid(raw_x, u_vec, np.zeros(raw_x.shape[0]), t_grid, self._cfg, self._dgp)
            c_curve, _, _ = true_censoring_on_grid(
                raw_x,
                u_vec,
                np.repeat(float(t_grid[-1]), raw_x.shape[0]),
                t_grid,
                self._cfg,
                self._dgp["beta_c"],
                self._dgp["lam_c"],
            )
            c_curve = np.maximum(c_curve, SURV_FLOOR)
        else:
            t_grid = self._t_grid
            s_hat_1 = _predict_s_on_grid(
                self._event_cox_1,
                self._cox_col_names_1,
                surv_features,
                t_grid,
                self._cox_keep_mask_1,
            )
            s_hat_0 = _predict_s_on_grid(
                self._event_cox_0,
                self._cox_col_names_0,
                surv_features,
                t_grid,
                self._cox_keep_mask_0,
            )
            c_curve = _predict_censoring_survival_on_grid(
                self._censor_model,
                censor_features,
                t_grid,
                clip_min=SURV_FLOOR,
            )

        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, t_grid)
        return t_grid, q_hat_1, q_hat_0, s_hat_1, s_hat_0, c_curve

    def predict(self, y, t, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_time, delta = self._unpack_y(y)
        a = np.asarray(t).ravel()
        raw_x, w_proxy, z_proxy, u_vec, q_features, h_features, surv_features, censor_features = self._split_inputs(X, W, Z)
        target_inputs = _prepare_target_inputs(y_time, delta, target=self._target, horizon=self._horizon)

        q_pred, h1_pred, h0_pred, m_pred = self._predict_q_h(raw_x, w_proxy, z_proxy, u_vec, q_features, h_features)
        t_grid, q_hat_1, q_hat_0, _, _, _ = self._predict_survival_blocks(raw_x, w_proxy, z_proxy, u_vec, surv_features, censor_features)
        q_hat = np.where((a == 1)[:, None], q_hat_1, q_hat_0)

        if self._true_surv:
            _, surv_c, hazard_c, sc_at_eval = _true_survival_components(
                raw_x,
                u_vec,
                target_inputs["eval_time"],
                t_grid,
                self._cfg,
                self._dgp,
            )
            if self._target == "RMST" and self._horizon is None:
                y_res = _compute_true_ipcw_3term_y_res(
                    y_time,
                    delta,
                    m_pred,
                    q_hat,
                    t_grid,
                    surv_c,
                    hazard_c,
                    sc_at_eval,
                    clip_percentiles=self._y_res_clip_percentiles,
                )
            else:
                y_res = _compute_true_target_ipcw_3term_y_res(
                    target_inputs["f_y"],
                    target_inputs["eval_time"],
                    target_inputs["eval_delta"],
                    m_pred,
                    q_hat,
                    t_grid,
                    surv_c,
                    hazard_c,
                    sc_at_eval,
                    clip_percentiles=self._y_res_clip_percentiles,
                )
        else:
            sc_at_eval = _predict_censoring_survival_at_values(
                self._censor_model,
                censor_features,
                target_inputs["eval_time"],
                clip_min=SURV_FLOOR,
            )
            sc_grid = _predict_censoring_survival_on_grid(
                self._censor_model,
                censor_features,
                t_grid,
                clip_min=SURV_FLOOR,
            )
            if self._target == "RMST" and self._horizon is None:
                y_res = _compute_ipcw_3term_y_res_from_survival(
                    y_time,
                    delta,
                    m_pred,
                    q_hat,
                    t_grid,
                    sc_at_eval,
                    sc_grid,
                    clip_percentiles=self._y_res_clip_percentiles,
                )
            else:
                y_res = _compute_target_ipcw_3term_y_res_from_survival(
                    target_inputs["f_y"],
                    target_inputs["eval_time"],
                    target_inputs["eval_delta"],
                    m_pred,
                    q_hat,
                    t_grid,
                    sc_at_eval,
                    sc_grid,
                    clip_percentiles=self._y_res_clip_percentiles,
                )

        a_res = (a - q_pred).reshape(-1, 1)
        return y_res, a_res

    def predict_bridge_outputs(self, X=None, W=None, Z=None):
        raw_x, w_proxy, z_proxy, u_vec, q_features, h_features, surv_features, censor_features = self._split_inputs(X, W, Z)
        q_pred, h1_pred, h0_pred, m_pred = self._predict_q_h(raw_x, w_proxy, z_proxy, u_vec, q_features, h_features)
        _, q_hat_1, q_hat_0, s_hat_1, s_hat_0, c_curve = self._predict_survival_blocks(raw_x, w_proxy, z_proxy, u_vec, surv_features, censor_features)
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
            "qhat1_curve": q_hat_1,
            "qhat0_curve": q_hat_0,
            "s1_curve": s_hat_1,
            "s0_curve": s_hat_0,
            "c_curve": c_curve,
        }


class _BenchmarkBestCurveOracleNuisance:
    def __init__(
        self,
        cfg,
        dgp,
        p_x,
        q_model,
        h_model,
        *,
        true_surv,
        true_qh,
        target,
        horizon,
        q_clip,
        y_tilde_clip_quantile,
        y_res_clip_percentiles,
        censoring_estimator,
    ):
        self._cfg = cfg
        self._dgp = dgp
        self._p_x = p_x
        self._true_surv = true_surv
        self._true_qh = true_qh
        self._target = target
        self._horizon = horizon
        self._q_model_template = q_model
        self._h_model_template = h_model
        self._q_clip = q_clip
        self._y_tilde_clip_quantile = y_tilde_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._censoring_estimator = censoring_estimator

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

    @staticmethod
    def _unpack_y(y):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] >= 2:
            return y[:, 0], y[:, 1]
        raise ValueError("Y must be a 2-column array [time, event].")

    def _split_inputs(self, x, w):
        raw_x = np.asarray(x, dtype=float)
        u_vec = _ensure_2d(w).astype(float)[:, 0]
        x_oracle = np.column_stack([raw_x, u_vec])
        return raw_x, u_vec, x_oracle

    def train(self, is_selecting, folds, y, t, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_time, delta = self._unpack_y(y)
        a = np.asarray(t).ravel()
        raw_x, u_vec, x_oracle = self._split_inputs(X, W)
        target_inputs = _prepare_target_inputs(y_time, delta, target=self._target, horizon=self._horizon)

        if self._true_surv:
            y_tilde = _build_true_y_tilde(
                raw_x,
                u_vec,
                y_time,
                delta,
                self._cfg,
                self._dgp,
                target=self._target,
                horizon=self._horizon,
            )
            self._t_grid = _cap_time_grid(target_inputs["grid_time"])
        else:
            self._censor_model = _fit_censoring_model(
                target_inputs["nuisance_time"],
                target_inputs["nuisance_delta"],
                x_oracle,
                estimator=self._censoring_estimator,
            )
            sc_at_eval = _predict_censoring_survival_at_values(
                self._censor_model,
                x_oracle,
                target_inputs["eval_time"],
                clip_min=SURV_FLOOR,
            )
            y_tilde = _compute_target_pseudo_outcome_from_sc(
                y_time=y_time,
                horizon=self._horizon,
                target=self._target,
                nuisance_time=target_inputs["nuisance_time"],
                nuisance_delta=target_inputs["nuisance_delta"],
                sc_at_eval=sc_at_eval,
            )
            self._t_grid = _cap_time_grid(target_inputs["grid_time"])

        y_tilde = _clip_quantile(y_tilde, self._y_tilde_clip_quantile)

        if not self._true_qh:
            self._q_model = clone(self._q_model_template)
            self._q_model.fit(x_oracle, a, **filter_none_kwargs(sample_weight=sample_weight))
            treated_mask = a == 1
            control_mask = a == 0
            self._h1_model = clone(self._h_model_template)
            self._h0_model = clone(self._h_model_template)
            if treated_mask.sum() > 10:
                self._h1_model.fit(
                    x_oracle[treated_mask],
                    y_tilde[treated_mask],
                    **filter_none_kwargs(sample_weight=None if sample_weight is None else sample_weight[treated_mask]),
                )
            if control_mask.sum() > 10:
                self._h0_model.fit(
                    x_oracle[control_mask],
                    y_tilde[control_mask],
                    **filter_none_kwargs(sample_weight=None if sample_weight is None else sample_weight[control_mask]),
                )

        if not self._true_surv:
            treated_mask = a == 1
            control_mask = a == 0
            self._event_cox_1, self._cox_col_names_1, self._cox_keep_mask_1 = _fit_event_cox(
                target_inputs["nuisance_time"][treated_mask],
                target_inputs["nuisance_delta"][treated_mask],
                x_oracle[treated_mask],
            )
            self._event_cox_0, self._cox_col_names_0, self._cox_keep_mask_0 = _fit_event_cox(
                target_inputs["nuisance_time"][control_mask],
                target_inputs["nuisance_delta"][control_mask],
                x_oracle[control_mask],
            )
        return self

    def _predict_q_h(self, raw_x, u_vec, x_oracle):
        if self._true_qh:
            q_pred = true_propensity_oracle(raw_x, u_vec, self._dgp, self._cfg)
            h0_pred, h1_pred = true_outcome_oracle(
                raw_x,
                u_vec,
                self._cfg,
                self._dgp,
                target=self._target,
                horizon=self._horizon,
            )
        else:
            q_pred = self._q_model.predict_proba(x_oracle)[:, 1]
            h1_pred = self._h1_model.predict(x_oracle)
            h0_pred = self._h0_model.predict(x_oracle)
        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        m_pred = q_pred * h1_pred + (1.0 - q_pred) * h0_pred
        return q_pred, h1_pred, h0_pred, m_pred

    def _predict_survival_blocks(self, raw_x, u_vec, x_oracle):
        if self._true_surv:
            t_grid = self._t_grid
            s_hat_1 = true_event_surv_on_grid(raw_x, u_vec, np.ones(raw_x.shape[0]), t_grid, self._cfg, self._dgp)
            s_hat_0 = true_event_surv_on_grid(raw_x, u_vec, np.zeros(raw_x.shape[0]), t_grid, self._cfg, self._dgp)
            c_curve, _, _ = true_censoring_on_grid(
                raw_x,
                u_vec,
                np.repeat(float(t_grid[-1]), raw_x.shape[0]),
                t_grid,
                self._cfg,
                self._dgp["beta_c"],
                self._dgp["lam_c"],
            )
            c_curve = np.maximum(c_curve, SURV_FLOOR)
        else:
            t_grid = self._t_grid
            s_hat_1 = _predict_s_on_grid(
                self._event_cox_1,
                self._cox_col_names_1,
                x_oracle,
                t_grid,
                self._cox_keep_mask_1,
            )
            s_hat_0 = _predict_s_on_grid(
                self._event_cox_0,
                self._cox_col_names_0,
                x_oracle,
                t_grid,
                self._cox_keep_mask_0,
            )
            c_curve = _predict_censoring_survival_on_grid(
                self._censor_model,
                x_oracle,
                t_grid,
                clip_min=SURV_FLOOR,
            )
        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, t_grid)
        return t_grid, q_hat_1, q_hat_0, s_hat_1, s_hat_0, c_curve

    def predict(self, y, t, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_time, delta = self._unpack_y(y)
        a = np.asarray(t).ravel()
        raw_x, u_vec, x_oracle = self._split_inputs(X, W)
        target_inputs = _prepare_target_inputs(y_time, delta, target=self._target, horizon=self._horizon)

        q_pred, h1_pred, h0_pred, m_pred = self._predict_q_h(raw_x, u_vec, x_oracle)
        t_grid, q_hat_1, q_hat_0, _, _, _ = self._predict_survival_blocks(raw_x, u_vec, x_oracle)
        q_hat = np.where((a == 1)[:, None], q_hat_1, q_hat_0)

        if self._true_surv:
            _, surv_c, hazard_c, sc_at_eval = _true_survival_components(
                raw_x,
                u_vec,
                target_inputs["eval_time"],
                t_grid,
                self._cfg,
                self._dgp,
            )
            if self._target == "RMST" and self._horizon is None:
                y_res = _compute_true_ipcw_3term_y_res(
                    y_time,
                    delta,
                    m_pred,
                    q_hat,
                    t_grid,
                    surv_c,
                    hazard_c,
                    sc_at_eval,
                    clip_percentiles=self._y_res_clip_percentiles,
                )
            else:
                y_res = _compute_true_target_ipcw_3term_y_res(
                    target_inputs["f_y"],
                    target_inputs["eval_time"],
                    target_inputs["eval_delta"],
                    m_pred,
                    q_hat,
                    t_grid,
                    surv_c,
                    hazard_c,
                    sc_at_eval,
                    clip_percentiles=self._y_res_clip_percentiles,
                )
        else:
            sc_at_eval = _predict_censoring_survival_at_values(
                self._censor_model,
                x_oracle,
                target_inputs["eval_time"],
                clip_min=SURV_FLOOR,
            )
            sc_grid = _predict_censoring_survival_on_grid(
                self._censor_model,
                x_oracle,
                t_grid,
                clip_min=SURV_FLOOR,
            )
            if self._target == "RMST" and self._horizon is None:
                y_res = _compute_ipcw_3term_y_res_from_survival(
                    y_time,
                    delta,
                    m_pred,
                    q_hat,
                    t_grid,
                    sc_at_eval,
                    sc_grid,
                    clip_percentiles=self._y_res_clip_percentiles,
                )
            else:
                y_res = _compute_target_ipcw_3term_y_res_from_survival(
                    target_inputs["f_y"],
                    target_inputs["eval_time"],
                    target_inputs["eval_delta"],
                    m_pred,
                    q_hat,
                    t_grid,
                    sc_at_eval,
                    sc_grid,
                    clip_percentiles=self._y_res_clip_percentiles,
                )

        a_res = (a - q_pred).reshape(-1, 1)
        return y_res, a_res

    def predict_bridge_outputs(self, X=None, W=None, Z=None):
        raw_x, u_vec, x_oracle = self._split_inputs(X, W)
        q_pred, h1_pred, h0_pred, m_pred = self._predict_q_h(raw_x, u_vec, x_oracle)
        _, q_hat_1, q_hat_0, s_hat_1, s_hat_0, c_curve = self._predict_survival_blocks(raw_x, u_vec, x_oracle)
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
            "qhat1_curve": q_hat_1,
            "qhat0_curve": q_hat_0,
            "s1_curve": s_hat_1,
            "s0_curve": s_hat_0,
            "c_curve": c_curve,
        }


class BenchmarkBestCurveNCSurvivalForest(BestCurveLocalCensoredPCISurvivalForest):
    def __init__(self, cfg, dgp, p_x, *, true_surv, true_qh, z_proxy_dim=1, **kwargs):
        self._benchmark_cfg = cfg
        self._benchmark_dgp = dgp
        self._benchmark_p_x = p_x
        self._benchmark_true_surv = true_surv
        self._benchmark_true_qh = true_qh
        self._benchmark_z_proxy_dim = z_proxy_dim
        super().__init__(**kwargs)

    def _make_nuisance(self):
        return _BenchmarkBestCurveNCNuisance(
            cfg=self._benchmark_cfg,
            dgp=self._benchmark_dgp,
            p_x=self._benchmark_p_x,
            z_proxy_dim=self._benchmark_z_proxy_dim,
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            true_surv=self._benchmark_true_surv,
            true_qh=self._benchmark_true_qh,
            target=self._target,
            horizon=self._horizon,
            q_clip=self._q_clip,
            y_tilde_clip_quantile=self._y_tilde_clip_quantile,
            y_res_clip_percentiles=self._y_res_clip_percentiles,
            nuisance_feature_mode=self._nuisance_feature_mode,
            censoring_estimator=self._censoring_estimator,
        )


class BenchmarkBestCurveOracleSurvivalForest(BestCurveLocalCensoredPCISurvivalForest):
    def __init__(self, cfg, dgp, p_x, *, true_surv, true_qh, **kwargs):
        self._benchmark_cfg = cfg
        self._benchmark_dgp = dgp
        self._benchmark_p_x = p_x
        self._benchmark_true_surv = true_surv
        self._benchmark_true_qh = true_qh
        super().__init__(**kwargs)

    def _make_nuisance(self):
        return _BenchmarkBestCurveOracleNuisance(
            cfg=self._benchmark_cfg,
            dgp=self._benchmark_dgp,
            p_x=self._benchmark_p_x,
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            true_surv=self._benchmark_true_surv,
            true_qh=self._benchmark_true_qh,
            target=self._target,
            horizon=self._horizon,
            q_clip=self._q_clip,
            y_tilde_clip_quantile=self._y_tilde_clip_quantile,
            y_res_clip_percentiles=self._y_res_clip_percentiles,
            censoring_estimator=self._censoring_estimator,
        )

    def fit_oracle(self, X, A, time, event, U):
        z_dummy = np.zeros((len(np.asarray(time).ravel()), 1), dtype=float)
        return self.fit_components(X, A, time, event, z_dummy, U)

    def effect_oracle(self, X, U):
        z_dummy = np.zeros((len(np.asarray(X)), 1), dtype=float)
        return self.effect_from_components(X, U, z_dummy)


def summarize_results_rmse_first(combined_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        combined_df.groupby("name", as_index=False)
        .agg(
            avg_pred_cate=("mean_pred", "mean"),
            avg_true_cate=("mean_true_cate", "mean"),
            avg_acc=("sign_acc", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_pehe=("pehe", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
            avg_time=("total_time", "mean"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def build_case_winner_table(combined_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id in sorted(combined_df["case_id"].unique()):
        case_rows = combined_df[combined_df["case_id"] == case_id].copy()
        rmse_best_row = case_rows.sort_values("rmse").iloc[0]
        mae_best_row = case_rows.sort_values("mae").iloc[0]
        rows.append(
            {
                "case_id": int(case_id),
                "case_label": CASE_LABELS[int(case_id)],
                "rmse_winner": str(rmse_best_row["name"]),
                "rmse_best": float(rmse_best_row["rmse"]),
                "mae_winner": str(mae_best_row["name"]),
                "mae_best": float(mae_best_row["mae"]),
            }
        )
    return pd.DataFrame(rows)


def build_winner_count_table(case_winner_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        rows.append(
            {
                "name": name,
                "rmse_wins": int((case_winner_df["rmse_winner"] == name).sum()),
                "mae_wins": int((case_winner_df["mae_winner"] == name).sum()),
                "avg_rmse": float(summary_df.loc[summary_df["name"] == name, "avg_rmse"].iloc[0]),
                "avg_mae": float(summary_df.loc[summary_df["name"] == name, "avg_mae"].iloc[0]),
                "avg_pearson": float(summary_df.loc[summary_df["name"] == name, "avg_pearson"].iloc[0]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["rmse_wins", "mae_wins", "avg_rmse"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def render_text_table_png(df: pd.DataFrame, output_path: Path, *, title: str, subtitle: str = ""):
    display = df.copy()
    for col in display.columns:
        if col.endswith("_rmse") or col.endswith("_mae") or col.endswith("_pearson"):
            display[col] = display[col].map(lambda v: f"{float(v):.4f}")
    for col in ["avg_rmse", "avg_mae", "avg_pearson", "rmse_best", "mae_best"]:
        if col in display.columns:
            display[col] = display[col].map(lambda v: f"{float(v):.4f}")

    fig_h = max(4.5, 0.42 * (len(display) + 2))
    fig, ax = plt.subplots(figsize=(22, fig_h))
    ax.axis("off")
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.98)
    if subtitle:
        ax.set_title(subtitle, fontsize=12, color="#4b5563", pad=12)
    table = ax.table(
        cellText=display.values,
        colLabels=list(display.columns),
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.02, 0.98, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("#d1d5db")
            cell.set_linewidth(1.2)
        else:
            cell.set_facecolor("#ffffff" if row % 2 == 1 else "#f3f4f6")
            cell.set_edgecolor("#d1d5db")
            cell.set_linewidth(0.8)
            if col in {0, 1}:
                cell.set_text_props(ha="left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_metric_compare_png(combined_df: pd.DataFrame, output_path: Path, *, metric: str, title: str):
    order = sorted(combined_df["case_id"].unique())
    y = np.arange(len(order))
    height = 0.08
    offsets = np.linspace(-3.5 * height, 3.5 * height, len(MODEL_ORDER))

    fig = plt.figure(figsize=(26, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.2])
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    winner_counts = {name: 0 for name in MODEL_ORDER}
    for idx, name in enumerate(MODEL_ORDER):
        frame = combined_df[combined_df["name"] == name].sort_values("case_id")
        ax.barh(y + offsets[idx], frame[metric], height=height, color=MODEL_COLORS[name], label=SHORT_LABELS[idx])

    for row_idx, case_id in enumerate(order):
        case_rows = combined_df[combined_df["case_id"] == case_id].set_index("name")
        best_name = min(MODEL_ORDER, key=lambda name: float(case_rows.loc[name, metric]))
        best_value = float(case_rows.loc[best_name, metric])
        winner_counts[best_name] += 1
        ax.text(best_value + 0.006, row_idx, SHORT_LABELS[MODEL_ORDER.index(best_name)], va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([CASE_LABELS[i] for i in order], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Case-wise {metric.upper()} comparison", fontsize=22, fontweight="bold", pad=16)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, ncols=2)

    win_values = [winner_counts[name] for name in MODEL_ORDER]
    bars = ax2.bar(SHORT_LABELS, win_values, color=[MODEL_COLORS[n] for n in MODEL_ORDER])
    ax2.set_ylim(0, len(order) + 2)
    ax2.set_ylabel("Number of cases won")
    ax2.set_title("Who wins more cases?", fontsize=22, fontweight="bold", pad=16)
    ax2.tick_params(axis="x", labelrotation=20)
    for bar in bars:
        value = int(bar.get_height())
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{value}", ha="center", va="bottom", fontsize=14, fontweight="bold")

    annotation_lines = [f"{label}: {combined_df[combined_df['name'] == name][metric].mean():.4f}" for label, name in zip(SHORT_LABELS, MODEL_ORDER)]
    ax2.text(
        0.05,
        0.04,
        f"Primary criterion: average {metric.upper()} across 12 cases\n\n" + "\n".join(annotation_lines),
        transform=ax2.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#bbbbbb"),
    )

    fig.suptitle(title, fontsize=28, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_pdf(bundle_paths: list[Path], output_pdf: Path) -> None:
    with PdfPages(output_pdf) as pdf:
        for path in bundle_paths:
            img = plt.imread(path)
            height, width = img.shape[:2]
            fig = plt.figure(figsize=(width / 150, height / 150))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _evaluate_oracle_variant(name, case, *, true_surv, true_qh, target):
    model = BenchmarkBestCurveOracleSurvivalForest(
        cfg=case.cfg,
        dgp=case.dgp,
        p_x=case.X.shape[1],
        true_surv=true_surv,
        true_qh=true_qh,
        target=target,
        horizon=case.horizon if target != "RMST" else None,
    )
    t0 = time.time()
    model.fit_oracle(case.X, case.A, case.Y, case.delta, case.U)
    preds = model.effect_oracle(case.X, case.U).ravel()
    elapsed = time.time() - t0
    return _evaluate_predictions(name, preds, case.true_cate, elapsed, backend=model.__class__.__name__)


def _evaluate_nc_variant(name, case, *, true_surv, true_qh, target):
    model = BenchmarkBestCurveNCSurvivalForest(
        cfg=case.cfg,
        dgp=case.dgp,
        p_x=case.X.shape[1],
        z_proxy_dim=case.Z.shape[1],
        true_surv=true_surv,
        true_qh=true_qh,
        target=target,
        horizon=case.horizon if target != "RMST" else None,
    )
    z_bundle = np.hstack([case.Z, _ensure_2d(case.U)])
    t0 = time.time()
    model.fit_components(case.X, case.A, case.Y, case.delta, z_bundle, case.W)
    preds = model.effect_from_components(case.X, case.W, z_bundle).ravel()
    elapsed = time.time() - t0
    return _evaluate_predictions(name, preds, case.true_cate, elapsed, backend=model.__class__.__name__)


def _evaluate_d2_variant(name, case, *, target):
    model = BestCurveLocalObservedCensoredSurvivalForest(target=target, horizon=case.horizon if target != "RMST" else None)
    t0 = time.time()
    model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
    preds = model.effect_from_components(case.X, case.W, case.Z).ravel()
    elapsed = time.time() - t0
    return _evaluate_predictions(name, preds, case.true_cate, elapsed, backend=model.__class__.__name__)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = set(args.case_ids) if args.case_ids else None
    selected_cases = [
        _case_with_overrides(case, n=args.n, p_w=args.p_w, p_z=args.p_z)
        for case in CASE_SPECS
        if selected_ids is None or case["case_id"] in selected_ids
    ]

    case_frames = []
    bundle_paths: list[Path] = []
    for case_spec in selected_cases:
        case = prepare_case(case_spec, target=args.target, horizon_quantile=args.horizon_quantile)
        case_title = _format_case_title(case_spec, case.cfg)
        w_cols = ["W"] if "W" in case.obs_df.columns else sorted(col for col in case.obs_df.columns if col.startswith("W"))
        z_cols = ["Z"] if "Z" in case.obs_df.columns else sorted(col for col in case.obs_df.columns if col.startswith("Z"))
        print("=" * 100)
        print(f"Running case {case_spec['case_id']:02d}")
        print(case_title)
        print(f"target={args.target}, horizon_quantile={args.horizon_quantile:.2f}")
        print("=" * 100)

        rows = []
        for name, kind, kwargs in VARIANT_SPECS:
            if kind == "oracle":
                rows.append(_evaluate_oracle_variant(name, case, target=args.target, **kwargs))
            elif kind == "nc":
                rows.append(_evaluate_nc_variant(name, case, target=args.target, **kwargs))
            elif kind == "b2":
                rows.append(
                    evaluate_r_csf_variant(
                        name,
                        case.obs_df,
                        case.x_cols + w_cols + z_cols,
                        case.true_cate,
                        case.horizon,
                        args.num_trees_b2,
                        target=args.target,
                    )
                )
            elif kind == "d2":
                rows.append(_evaluate_d2_variant(name, case, target=args.target))
            else:
                raise ValueError(f"Unknown variant kind: {kind}")

        case_df = pd.DataFrame(rows)
        case_df = case_df.loc[
            :,
            [
                "name",
                "mean_pred",
                "std_pred",
                "mean_true_cate",
                "std_true_cate",
                "bias",
                "rmse",
                "pehe",
                "mae",
                "pearson",
                "sign_acc",
                "total_time",
                "backend",
            ],
        ]
        case_df.insert(0, "case_id", case_spec["case_id"])
        case_df.insert(1, "case_slug", case_spec["slug"])
        case_df.insert(2, "case_title", case_title)
        case_df["target"] = args.target
        case_df["estimand_horizon"] = float(case.horizon)
        case_df["horizon_quantile"] = float(args.horizon_quantile) if args.target == "survival.probability" else None
        case_df["n"] = case.cfg.n
        case_df["p_x"] = case.cfg.p_x
        case_df["p_w"] = case.cfg.p_w
        case_df["p_z"] = case.cfg.p_z
        case_df["seed"] = case.cfg.seed
        case_df["target_censor_rate"] = case.cfg.target_censor_rate
        case_df["actual_censor_rate"] = float(1.0 - case.delta.mean())
        case_df["linear_treatment"] = case.cfg.linear_treatment
        case_df["linear_outcome"] = case.cfg.linear_outcome
        case_df["tau_log_hr"] = case.cfg.tau_log_hr

        case_frames.append(case_df)
        pd.concat(case_frames, ignore_index=True).to_csv(output_dir / "results_partial.csv", index=False)

        base_name = f"case_{case_spec['case_id']:02d}_{case_spec['slug']}"
        case_csv = output_dir / f"{base_name}.csv"
        case_png = output_dir / f"{base_name}.png"
        case_df.to_csv(case_csv, index=False)
        render_case_table_png(case_df, case_png)
        bundle_paths.append(case_png)

    combined_df = pd.concat(case_frames, ignore_index=True)
    results_csv = output_dir / "all_12case_bestcurve_8variant_results.csv"
    combined_df.to_csv(results_csv, index=False)

    summary_df = summarize_results_rmse_first(combined_df)
    top5_df = summary_df.head(5).copy()
    best_df = summary_df.head(1).copy()
    summary_csv = output_dir / "all_12case_bestcurve_8variant_summary.csv"
    top5_csv = output_dir / "all_12case_bestcurve_8variant_top5.csv"
    best_csv = output_dir / "best_model.csv"
    summary_df.to_csv(summary_csv, index=False)
    top5_df.to_csv(top5_csv, index=False)
    best_df.to_csv(best_csv, index=False)

    summary_png = output_dir / "all_12case_bestcurve_8variant_summary.png"
    top5_png = output_dir / "all_12case_bestcurve_8variant_top5.png"
    summary_subtitle = (
        f"Sorted by avg RMSE, then avg MAE, then avg Pearson | "
        f"n={args.n}, p_x=5, p_w={args.p_w}, p_z={args.p_z}"
    )
    render_text_table_png(summary_df, summary_png, title="Censored BestCurve 8-variant average summary", subtitle=summary_subtitle)
    render_text_table_png(top5_df, top5_png, title="Censored BestCurve 8-variant top 5", subtitle=f"Top 5 by avg RMSE | n={args.n}, p_w={args.p_w}, p_z={args.p_z}")

    case_winner_df = build_case_winner_table(combined_df)
    case_winner_csv = output_dir / "case_metric_winners.csv"
    case_winner_png = output_dir / "case_metric_winners.png"
    case_winner_df.to_csv(case_winner_csv, index=False)
    render_text_table_png(case_winner_df, case_winner_png, title="12-case RMSE / MAE winners", subtitle="Per-case winning model for each metric")

    winner_count_df = build_winner_count_table(case_winner_df, summary_df)
    winner_count_csv = output_dir / "metric_winner_counts.csv"
    winner_count_png = output_dir / "metric_winner_counts.png"
    winner_count_df.to_csv(winner_count_csv, index=False)
    render_text_table_png(winner_count_df, winner_count_png, title="Winner counts across 12 cases", subtitle="How many cases each model wins on RMSE and MAE")

    rmse_png = output_dir / "bestcurve_8variant_rmse_comparison.png"
    mae_png = output_dir / "bestcurve_8variant_mae_comparison.png"
    render_metric_compare_png(combined_df, rmse_png, metric="rmse", title="Censored BestCurve 8-variant benchmark\nRMSE-based comparison")
    render_metric_compare_png(combined_df, mae_png, metric="mae", title="Censored BestCurve 8-variant benchmark\nMAE-based comparison")

    pdf_path = output_dir / "bestcurve_8variant_bundle.pdf"
    render_pdf(
        [summary_png, top5_png, case_winner_png, winner_count_png, rmse_png, mae_png, *bundle_paths],
        pdf_path,
    )
    print(f"Saved {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
