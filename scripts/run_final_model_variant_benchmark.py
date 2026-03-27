#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from econml._ortho_learner import _OrthoLearner
from econml.utilities import filter_none_kwargs
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks.econml_8variant import (  # noqa: E402
    CASE_SPECS,
    _BenchmarkNCSurvivalNuisance,
    _BenchmarkOracleSurvivalNuisance,
    _compute_q_from_s,
    _compute_survival_probability_q_from_s,
    _ensure_2d,
    _evaluate_predictions,
    _predict_s_on_grid,
    _true_survival_components,
    evaluate_r_csf_variant,
    prepare_case,
    render_avg_summary_png,
    render_case_table_png,
    render_top5_png,
    summarize_results,
    true_event_surv_on_grid,
    true_outcome_nc,
    true_outcome_oracle,
    true_propensity_nc,
    true_propensity_oracle,
)
from grf.methods.econml_oldc3_ablation_survival import (  # noqa: E402
    SinglePassBridgeFeatureCensoredSurvivalForest,
)

matplotlib.use("Agg")


VARIANT_SPECS = [
    ("A1  Final Model Oracle (all true)", "oracle", dict(true_surv=True, true_qh=True)),
    ("A2  Final Model Oracle (true surv, est q/h)", "oracle", dict(true_surv=True, true_qh=False)),
    ("A3  Final Model Oracle (all estimated)", "oracle", dict(true_surv=False, true_qh=False)),
    ("R-CSF Baseline", "baseline", {}),
    ("C1  Final Model (all true)", "proxy", dict(true_surv=True, true_qh=True)),
    ("C2  Final Model (true surv, est q/h)", "proxy", dict(true_surv=True, true_qh=False)),
    ("C3  Final Model (all estimated)", "proxy", dict(true_surv=False, true_qh=False)),
]

SETTINGS = [
    {"setting_id": "S01", "n": 1000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S02", "n": 2000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S03", "n": 4000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S04", "n": 8000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S05", "n": 2000, "p_x": 5, "p_w": 3, "p_z": 3},
    {"setting_id": "S06", "n": 2000, "p_x": 5, "p_w": 5, "p_z": 5},
    {"setting_id": "S07", "n": 2000, "p_x": 5, "p_w": 10, "p_z": 10},
    {"setting_id": "S08", "n": 2000, "p_x": 10, "p_w": 1, "p_z": 1},
    {"setting_id": "S09", "n": 2000, "p_x": 20, "p_w": 1, "p_z": 1},
    {"setting_id": "S10", "n": 2000, "p_x": 10, "p_w": 5, "p_z": 5},
    {"setting_id": "S11", "n": 2000, "p_x": 20, "p_w": 5, "p_z": 5},
    {"setting_id": "S12", "n": 1000, "p_x": 10, "p_w": 5, "p_z": 5},
    {"setting_id": "S13", "n": 2000, "p_x": 10, "p_w": 10, "p_z": 10},
    {"setting_id": "S14", "n": 4000, "p_x": 20, "p_w": 10, "p_z": 10},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the censored Final Model variant benchmark for basic12 or structured14."
    )
    parser.add_argument("--suite", choices=["basic12", "structured14"], default="basic12")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--setting-ids", nargs="*")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p-x", type=int, default=None)
    parser.add_argument("--p-w", type=int, default=1)
    parser.add_argument("--p-z", type=int, default=1)
    parser.add_argument("--num-trees-baseline", type=int, default=200)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


def _resolve_output_dir(args) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    if args.suite == "structured14":
        return (PROJECT_ROOT / "outputs" / "benchmark_structured_14settings_final_model_variants").resolve()
    return (PROJECT_ROOT / "outputs" / "benchmark_final_model_variants_12case").resolve()


def _case_with_overrides(case_spec, *, n=None, p_x=None, p_w=None, p_z=None):
    case_copy = dict(case_spec)
    cfg_updates = dict(case_spec["cfg"])
    if n is not None:
        cfg_updates["n"] = int(n)
    if p_x is not None:
        cfg_updates["p_x"] = int(p_x)
    if p_w is not None:
        cfg_updates["p_w"] = int(p_w)
    if p_z is not None:
        cfg_updates["p_z"] = int(p_z)
    case_copy["cfg"] = cfg_updates
    return case_copy


def _setting_slug(setting: dict[str, int | str]) -> str:
    return (
        f"{setting['setting_id']}_n{setting['n']}_px{setting['p_x']}"
        f"_pw{setting['p_w']}_pz{setting['p_z']}"
    )


def _format_case_title(case_spec, cfg):
    base = str(case_spec["title"]).split(", n=", 1)[0]
    censor_pct = int(round(100 * float(cfg.target_censor_rate)))
    return (
        f"{base}, n={cfg.n}, p_x={cfg.p_x}, p_w={cfg.p_w}, "
        f"p_z={cfg.p_z}, seed={cfg.seed}, censoring rate={censor_pct}%"
    )


class _BenchmarkBridgeNCSurvivalNuisance(_BenchmarkNCSurvivalNuisance):
    def predict_bridge_outputs(self, X=None, W=None, Z=None):
        x_full, x_base, w_proxy, z_proxy, u_vec = self._split_inputs(X, W, Z)

        if self._true_qh:
            q_pred = true_propensity_nc(z_proxy, x_base, self._dgp, self._cfg)
            h0_pred, h1_pred = true_outcome_nc(
                w_proxy,
                x_base,
                self._cfg,
                self._dgp,
                target=self._target,
                horizon=self._horizon,
            )
        else:
            xz = np.column_stack([x_full, z_proxy])
            xw = np.column_stack([x_full, w_proxy])
            q_pred = self._q_model.predict_proba(xz)[:, 1]
            h1_pred = self._h1_model.predict(xw)
            h0_pred = self._h0_model.predict(xw)

        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        m_pred = q_pred * h1_pred + (1.0 - q_pred) * h0_pred

        if self._true_surv:
            if u_vec is None:
                raise ValueError("True survival nuisances require bundled U.")
            t_grid, _, _, _ = _true_survival_components(
                x_base,
                u_vec,
                np.zeros(len(x_base), dtype=float),
                np.zeros(1, dtype=float),
                self._cfg,
                self._dgp,
            )
            s_hat_1 = true_event_surv_on_grid(x_base, u_vec, np.ones(len(x_base), dtype=int), t_grid, self._cfg, self._dgp)
            s_hat_0 = true_event_surv_on_grid(x_base, u_vec, np.zeros(len(x_base), dtype=int), t_grid, self._cfg, self._dgp)
        else:
            surv_features = np.column_stack([x_full, w_proxy, z_proxy])
            s_hat_1 = _predict_s_on_grid(
                self._event_cox_1,
                self._cox_col_names_1,
                surv_features,
                self._t_grid,
                self._cox_keep_mask_1,
            )
            s_hat_0 = _predict_s_on_grid(
                self._event_cox_0,
                self._cox_col_names_0,
                surv_features,
                self._t_grid,
                self._cox_keep_mask_0,
            )
            t_grid = self._t_grid

        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, t_grid)

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

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_res, a_res = super().predict(Y, T, X=X, W=W, Z=Z, sample_weight=sample_weight, groups=groups)
        bridge = self.predict_bridge_outputs(X=X, W=W, Z=Z)
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


class _BenchmarkBridgeOracleSurvivalNuisance(_BenchmarkOracleSurvivalNuisance):
    def predict_bridge_outputs(self, X=None, W=None, Z=None):
        x_full, x_base, u_vec = self._split_inputs(X, W)

        if self._true_qh:
            q_pred = true_propensity_oracle(x_base, u_vec, self._dgp, self._cfg)
            h0_pred, h1_pred = true_outcome_oracle(
                x_base,
                u_vec,
                self._cfg,
                self._dgp,
                target=self._target,
                horizon=self._horizon,
            )
        else:
            q_pred = self._q_model.predict_proba(x_full)[:, 1]
            h1_pred = self._h1_model.predict(x_full)
            h0_pred = self._h0_model.predict(x_full)

        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        m_pred = q_pred * h1_pred + (1.0 - q_pred) * h0_pred

        if self._true_surv:
            t_grid, _, _, _ = _true_survival_components(
                x_base,
                u_vec,
                np.zeros(len(x_base), dtype=float),
                np.zeros(1, dtype=float),
                self._cfg,
                self._dgp,
            )
            s_hat_1 = true_event_surv_on_grid(x_base, u_vec, np.ones(len(x_base), dtype=int), t_grid, self._cfg, self._dgp)
            s_hat_0 = true_event_surv_on_grid(x_base, u_vec, np.zeros(len(x_base), dtype=int), t_grid, self._cfg, self._dgp)
        else:
            s_hat_1 = _predict_s_on_grid(
                self._event_cox_1,
                self._cox_col_names_1,
                x_full,
                self._t_grid,
                self._cox_keep_mask_1,
            )
            s_hat_0 = _predict_s_on_grid(
                self._event_cox_0,
                self._cox_col_names_0,
                x_full,
                self._t_grid,
                self._cox_keep_mask_0,
            )
            t_grid = self._t_grid

        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, t_grid)

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

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_res, a_res = super().predict(Y, T, X=X, W=W, Z=Z, sample_weight=sample_weight, groups=groups)
        bridge = self.predict_bridge_outputs(X=X, W=W, Z=Z)
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


class _BridgeFeatureSurvivalModelFinal:
    def __init__(self, base_final, *, builder):
        self._base_final = base_final
        self._builder = builder
        self._training_x_final = None

    def fit(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, freq_weight=None, sample_var=None, groups=None):
        y_res, a_res, q_pred, h1_pred, h0_pred, m_pred, surv1_pred, surv0_pred, surv_diff_pred = nuisances
        x_final = self._builder(
            x=np.asarray(X, dtype=float),
            w_raw=None if W is None else np.asarray(W, dtype=float),
            z_raw=None if Z is None else np.asarray(Z, dtype=float),
            q_pred=np.asarray(q_pred, dtype=float),
            h1_pred=np.asarray(h1_pred, dtype=float),
            h0_pred=np.asarray(h0_pred, dtype=float),
            m_pred=np.asarray(m_pred, dtype=float),
            surv1_pred=np.asarray(surv1_pred, dtype=float),
            surv0_pred=np.asarray(surv0_pred, dtype=float),
            surv_diff_pred=np.asarray(surv_diff_pred, dtype=float),
        )
        self._training_x_final = x_final
        return self._base_final.fit(
            Y,
            T,
            X=x_final,
            W=None,
            Z=None,
            nuisances=(y_res, a_res),
            sample_weight=sample_weight,
            freq_weight=freq_weight,
            sample_var=sample_var,
            groups=groups,
        )

    def predict(self, X=None):
        return self._base_final.predict(X=X)

    @property
    def training_x_final(self):
        return self._training_x_final


class BenchmarkFinalProxyCensoredSurvivalForest(SinglePassBridgeFeatureCensoredSurvivalForest):
    def __init__(self, cfg, dgp, p_x, z_proxy_dim, *, true_surv, true_qh, **kwargs):
        self._benchmark_cfg = cfg
        self._benchmark_dgp = dgp
        self._benchmark_p_x = p_x
        self._benchmark_z_proxy_dim = z_proxy_dim
        self._benchmark_true_surv = true_surv
        self._benchmark_true_qh = true_qh
        super().__init__(**kwargs)

    def _gen_ortho_learner_model_nuisance(self):
        return _BenchmarkBridgeNCSurvivalNuisance(
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
            q_clip=self._custom_q_clip,
            y_tilde_clip_quantile=self._custom_y_tilde_clip_quantile,
            y_res_clip_percentiles=self._custom_y_res_clip_percentiles,
        )


class BenchmarkFinalOracleCensoredSurvivalForest(SinglePassBridgeFeatureCensoredSurvivalForest):
    def __init__(self, cfg, dgp, p_x, *, true_surv, true_qh, **kwargs):
        self._benchmark_cfg = cfg
        self._benchmark_dgp = dgp
        self._benchmark_p_x = p_x
        self._benchmark_true_surv = true_surv
        self._benchmark_true_qh = true_qh
        super().__init__(**kwargs)

    def _gen_ortho_learner_model_nuisance(self):
        return _BenchmarkBridgeOracleSurvivalNuisance(
            cfg=self._benchmark_cfg,
            dgp=self._benchmark_dgp,
            p_x=self._benchmark_p_x,
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            true_surv=self._benchmark_true_surv,
            true_qh=self._benchmark_true_qh,
            target=self._target,
            horizon=self._horizon,
            q_clip=self._custom_q_clip,
            y_tilde_clip_quantile=self._custom_y_tilde_clip_quantile,
            y_res_clip_percentiles=self._custom_y_res_clip_percentiles,
        )

    def fit_oracle(self, X, A, time, event, U, **kwargs):
        x = np.asarray(X, dtype=float)
        y = np.asarray(time, dtype=float).ravel()
        delta = np.asarray(event, dtype=float).ravel()
        u = _ensure_2d(U)
        y_packed = np.column_stack([y, delta])
        z_dummy = np.zeros((len(y), 1), dtype=float)
        return _OrthoLearner.fit(self, y_packed, A, X=x, W=u, Z=z_dummy, **kwargs)


def _make_final_model_kwargs():
    return dict(
        include_raw_proxy=True,
        include_surv_scalar=True,
        surv_scalar_mode="pair",
        target="RMST",
        horizon=None,
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=42,
        q_kind="logit",
        h_kind="extra",
        h_n_estimators=600,
        h_min_samples_leaf=5,
        q_clip=0.03,
        y_tilde_clip_quantile=0.98,
        y_res_clip_percentiles=(2.0, 98.0),
        censoring_estimator="nelson-aalen",
        nuisance_feature_mode="broad_dup",
        n_jobs=1,
    )


def _evaluate_oracle_variant(name, case, *, true_surv: bool, true_qh: bool):
    u = _ensure_2d(case.U)
    x_oracle = np.column_stack([case.X, u])
    model = BenchmarkFinalOracleCensoredSurvivalForest(
        cfg=case.cfg,
        dgp=case.dgp,
        p_x=case.X.shape[1],
        true_surv=true_surv,
        true_qh=true_qh,
        **_make_final_model_kwargs(),
    )
    model._raw_w_for_final = np.zeros((len(case.X), 0), dtype=float)
    model._raw_z_for_final = np.zeros((len(case.X), 0), dtype=float)
    t0 = time.time()
    model.fit_oracle(x_oracle, case.A, case.Y, case.delta, u)
    preds = model.effect_on_final_features(model.training_x_final()).ravel()
    elapsed = time.time() - t0
    return _evaluate_predictions(name, preds, case.true_cate, elapsed, backend=model.__class__.__name__)


def _evaluate_proxy_variant(name, case, *, true_surv: bool, true_qh: bool):
    z_bundle = np.hstack([case.Z, _ensure_2d(case.U)])
    model = BenchmarkFinalProxyCensoredSurvivalForest(
        cfg=case.cfg,
        dgp=case.dgp,
        p_x=case.X.shape[1],
        z_proxy_dim=case.Z.shape[1],
        true_surv=true_surv,
        true_qh=true_qh,
        **_make_final_model_kwargs(),
    )
    model._raw_w_for_final = np.asarray(case.W, dtype=float).copy()
    model._raw_z_for_final = np.asarray(case.Z, dtype=float).copy()
    t0 = time.time()
    model.fit_survival(case.X, case.A, case.Y, case.delta, z_bundle, case.W)
    preds = model.effect_on_final_features(model.training_x_final()).ravel()
    elapsed = time.time() - t0
    return _evaluate_predictions(name, preds, case.true_cate, elapsed, backend=model.__class__.__name__)


def _evaluate_case_variant(case, name: str, kind: str, kwargs: dict[str, object], num_trees_baseline: int):
    if kind == "oracle":
        return _evaluate_oracle_variant(name, case, **kwargs)
    if kind == "proxy":
        return _evaluate_proxy_variant(name, case, **kwargs)
    if kind == "baseline":
        return evaluate_r_csf_variant(
            name,
            case.obs_df,
            case.x_cols + case.w_cols + case.z_cols,
            case.true_cate,
            case.horizon,
            num_trees_baseline,
            target="RMST",
        )
    raise ValueError(f"Unknown variant kind: {kind}")


def _consolidate_structured_outputs(output_dir: Path) -> None:
    frames: list[pd.DataFrame] = []
    for setting in SETTINGS:
        path = output_dir / _setting_slug(setting) / "results.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    summary_df, top5_df = summarize_results(combined)
    combined.to_csv(output_dir / "all_settings_results.csv", index=False)
    summary_df.to_csv(output_dir / "all_settings_summary.csv", index=False)
    top5_df.to_csv(output_dir / "all_settings_top5.csv", index=False)


def _run_basic12(case_specs: list[dict[str, object]], output_dir: Path, *, num_trees_baseline: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_frames = []
    for case_spec in case_specs:
        case = prepare_case(case_spec, target="RMST", horizon_quantile=0.60)
        case_title = _format_case_title(case_spec, case.cfg)
        rows = []
        for name, kind, kwargs in VARIANT_SPECS:
            row = _evaluate_case_variant(case, name, kind, kwargs, num_trees_baseline)
            row["case_id"] = case_spec["case_id"]
            row["case_slug"] = case_spec["slug"]
            row["case_title"] = case_title
            rows.append(row)

        case_df = pd.DataFrame(rows)
        case_frames.append(case_df)
        case_csv = output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.csv"
        case_png = output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.png"
        case_df.to_csv(case_csv, index=False)
        render_case_table_png(case_df, case_png)

    combined_df = pd.concat(case_frames, ignore_index=True)
    summary_df, top5_df = summarize_results(combined_df)
    combined_df.to_csv(output_dir / "all_12case_final_model_variant_results.csv", index=False)
    summary_df.to_csv(output_dir / "all_12case_final_model_variant_summary.csv", index=False)
    top5_df.to_csv(output_dir / "all_12case_final_model_variant_top5.csv", index=False)
    render_avg_summary_png(summary_df, output_dir / "all_12case_final_model_variant_summary.png")
    render_top5_png(top5_df, output_dir / "all_12case_final_model_variant_top5.png")


def _run_structured14(
    settings: list[dict[str, int | str]],
    *,
    case_ids: set[int] | None,
    output_dir: Path,
    num_trees_baseline: int,
    skip_existing: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for setting in settings:
        setting_dir = output_dir / _setting_slug(setting)
        results_path = setting_dir / "results.csv"
        summary_path = setting_dir / "summary.csv"
        if skip_existing and results_path.exists() and summary_path.exists():
            continue

        setting_dir.mkdir(parents=True, exist_ok=True)
        case_frames = []
        for raw_case_spec in CASE_SPECS:
            if case_ids is not None and int(raw_case_spec["case_id"]) not in case_ids:
                continue
            case_spec = _case_with_overrides(
                raw_case_spec,
                n=setting["n"],
                p_x=setting["p_x"],
                p_w=setting["p_w"],
                p_z=setting["p_z"],
            )
            case = prepare_case(case_spec, target="RMST", horizon_quantile=0.60)
            case_title = _format_case_title(case_spec, case.cfg)
            rows = []
            for name, kind, kwargs in VARIANT_SPECS:
                row = _evaluate_case_variant(case, name, kind, kwargs, num_trees_baseline)
                row["setting_id"] = str(setting["setting_id"])
                row["setting_slug_full"] = _setting_slug(setting)
                row["case_id"] = int(case_spec["case_id"])
                row["case_slug"] = str(case_spec["slug"])
                row["case_title"] = case_title
                row["n"] = int(case.cfg.n)
                row["p_x"] = int(case.cfg.p_x)
                row["p_w"] = int(case.cfg.p_w)
                row["p_z"] = int(case.cfg.p_z)
                rows.append(row)
            case_frames.append(pd.DataFrame(rows))

        results = pd.concat(case_frames, ignore_index=True)
        summary_df, top5_df = summarize_results(results)
        results.to_csv(results_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        top5_df.to_csv(setting_dir / "top5.csv", index=False)
        _consolidate_structured_outputs(output_dir)

    _consolidate_structured_outputs(output_dir)


def main() -> int:
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    selected_case_ids = set(args.case_ids) if args.case_ids else None

    if args.suite == "basic12":
        case_specs = [
            _case_with_overrides(case, n=args.n, p_x=args.p_x, p_w=args.p_w, p_z=args.p_z)
            for case in CASE_SPECS
            if selected_case_ids is None or int(case["case_id"]) in selected_case_ids
        ]
        _run_basic12(case_specs, output_dir, num_trees_baseline=args.num_trees_baseline)
        return 0

    selected_setting_ids = set(args.setting_ids) if args.setting_ids else None
    settings = [s for s in SETTINGS if selected_setting_ids is None or s["setting_id"] in selected_setting_ids]
    _run_structured14(
        settings,
        case_ids=selected_case_ids,
        output_dir=output_dir,
        num_trees_baseline=args.num_trees_baseline,
        skip_existing=args.skip_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
