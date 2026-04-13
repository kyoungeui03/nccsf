from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from econml._ortho_learner import _OrthoLearner
from econml.dml import CausalForestDML
from econml.grf import CausalForest
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from grf.r_runtime import resolve_rscript

from .econml_mild_shrink import (
    EconmlMildShrinkNCSurvivalForest,
    _MildShrinkNCSurvivalNuisance,
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
    make_h_model,
    make_q_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
R_CSF_FINAL_FOREST_SCRIPT = PROJECT_ROOT / "scripts" / "run_grf_csf_final_forest.R"
R_CSF_SURVIVAL_FINAL_FOREST_SCRIPT = PROJECT_ROOT / "scripts" / "run_grf_csf_survival_final_forest.R"

_SURV_SCALAR_MODES = {
    "none",
    "pair",
    "full",
    "raw",
    "raw_surv",
    "qhat_diff_pts",
    "qhat_full_pts",
    "qhat_stats",
    "s_stats",
    "censor_stats",
    "disagreement",
    "qhat_censor",
    "qhat_disagree",
    "all_lite",
}
_SURV_SCALAR_SUMMARY_COUNTS = {
    "none": 4,
    "pair": 6,
    "full": 7,
    "raw_surv": 3,
    "qhat_diff_pts": 10,
    "qhat_full_pts": 16,
    "qhat_stats": 13,
    "s_stats": 13,
    "censor_stats": 11,
    "disagreement": 11,
    "qhat_censor": 17,
    "qhat_disagree": 17,
    "all_lite": 21,
}


def _resolve_surv_scalar_mode(include_surv_scalar: bool, surv_scalar_mode: str | None) -> str:
    if surv_scalar_mode is None:
        return "full" if include_surv_scalar else "none"
    if surv_scalar_mode not in _SURV_SCALAR_MODES:
        raise ValueError(f"Unsupported surv_scalar_mode: {surv_scalar_mode}")
    return surv_scalar_mode


def _mode_uses_surv_scalar(surv_scalar_mode: str) -> bool:
    return surv_scalar_mode not in {"none", "raw"}


def _curve_anchor_values(curve, fractions=(0.25, 0.50, 0.75)):
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2:
        raise ValueError("curve must be 2D.")
    if curve.shape[1] == 0:
        return np.zeros((curve.shape[0], len(fractions)), dtype=float)
    indices = [
        int(np.clip(round((curve.shape[1] - 1) * frac), 0, max(curve.shape[1] - 1, 0)))
        for frac in fractions
    ]
    return curve[:, indices]


def _curve_window_means(curve, n_windows=3):
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2:
        raise ValueError("curve must be 2D.")
    if curve.shape[1] == 0:
        return np.zeros((curve.shape[0], n_windows), dtype=float)
    splits = np.array_split(np.arange(curve.shape[1]), n_windows)
    means = []
    for split in splits:
        if len(split) == 0:
            means.append(np.zeros(curve.shape[0], dtype=float))
        else:
            means.append(np.mean(curve[:, split], axis=1))
    return np.column_stack(means)


def _curve_auc(curve, t_grid):
    curve = np.asarray(curve, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    if curve.ndim != 2:
        raise ValueError("curve must be 2D.")
    if curve.shape[1] == 0:
        return np.zeros(curve.shape[0], dtype=float)
    if t_grid.shape[0] != curve.shape[1]:
        return np.mean(curve, axis=1)
    span = float(t_grid[-1] - t_grid[0]) if t_grid.shape[0] > 1 else 1.0
    span = max(span, 1e-8)
    return np.trapezoid(curve, x=t_grid, axis=1) / span


def _survival_feature_extras(bridge, surv_scalar_mode: str):
    if surv_scalar_mode in {"none", "pair", "full", "raw", "raw_surv"}:
        return []
    summary = _bridge_scalar_summaries(bridge)

    extras: list[np.ndarray] = []
    if surv_scalar_mode == "qhat_diff_pts":
        extras.append(summary["qhat_diff_pts"])
    elif surv_scalar_mode == "qhat_full_pts":
        extras.extend(
            [
                summary["qhat1_pts"],
                summary["qhat0_pts"],
                summary["qhat_diff_pts"],
            ]
        )
    elif surv_scalar_mode == "qhat_stats":
        extras.extend(
            [
                summary["qhat_auc_stats"],
                summary["qhat_diff_windows"],
            ]
        )
    elif surv_scalar_mode == "s_stats":
        extras.extend(
            [
                summary["s_auc_stats"],
                summary["s_diff_windows"],
            ]
        )
    elif surv_scalar_mode == "censor_stats":
        extras.append(summary["c_stats"])
    elif surv_scalar_mode == "disagreement":
        extras.extend(
            [
                np.asarray(bridge["h_diff_pred"], dtype=float).reshape(-1, 1),
                np.asarray(bridge["m_survival_pred"], dtype=float).reshape(-1, 1),
                np.asarray(bridge["m_gap_pred"], dtype=float).reshape(-1, 1),
                np.abs(np.asarray(bridge["m_gap_pred"], dtype=float)).reshape(-1, 1),
            ]
        )
    elif surv_scalar_mode == "qhat_censor":
        extras.extend(
            [
                summary["qhat_auc_stats"],
                summary["qhat_diff_windows"],
                summary["c_stats"],
            ]
        )
    elif surv_scalar_mode == "qhat_disagree":
        extras.extend(
            [
                summary["qhat_auc_stats"],
                summary["qhat_diff_windows"],
                np.asarray(bridge["h_diff_pred"], dtype=float).reshape(-1, 1),
                np.asarray(bridge["m_survival_pred"], dtype=float).reshape(-1, 1),
                np.asarray(bridge["m_gap_pred"], dtype=float).reshape(-1, 1),
                np.abs(np.asarray(bridge["m_gap_pred"], dtype=float)).reshape(-1, 1),
            ]
        )
    elif surv_scalar_mode == "all_lite":
        extras.extend(
            [
                summary["qhat_auc_stats"],
                summary["qhat_diff_windows"],
                summary["c_stats"],
                np.asarray(bridge["h_diff_pred"], dtype=float).reshape(-1, 1),
                np.asarray(bridge["m_survival_pred"], dtype=float).reshape(-1, 1),
                np.asarray(bridge["m_gap_pred"], dtype=float).reshape(-1, 1),
                np.abs(np.asarray(bridge["m_gap_pred"], dtype=float)).reshape(-1, 1),
            ]
        )
    else:
        raise ValueError(f"Unsupported surv_scalar_mode: {surv_scalar_mode}")
    return extras


def _bridge_scalar_summaries(bridge):
    if "qhat_auc_stats" in bridge:
        return {
            "qhat1_pts": np.asarray(bridge["qhat1_pts"], dtype=float),
            "qhat0_pts": np.asarray(bridge["qhat0_pts"], dtype=float),
            "qhat_diff_pts": np.asarray(bridge["qhat_diff_pts"], dtype=float),
            "qhat_auc_stats": np.asarray(bridge["qhat_auc_stats"], dtype=float),
            "qhat_diff_windows": np.asarray(bridge["qhat_diff_windows"], dtype=float),
            "s_auc_stats": np.asarray(bridge["s_auc_stats"], dtype=float),
            "s_diff_windows": np.asarray(bridge["s_diff_windows"], dtype=float),
            "c_stats": np.asarray(bridge["c_stats"], dtype=float),
        }

    qhat1_curve = np.asarray(bridge["qhat1_curve"], dtype=float)
    qhat0_curve = np.asarray(bridge["qhat0_curve"], dtype=float)
    s1_curve = np.asarray(bridge["s1_curve"], dtype=float)
    s0_curve = np.asarray(bridge["s0_curve"], dtype=float)
    c_curve = np.asarray(bridge["c_curve"], dtype=float)
    qhat_diff_curve = qhat1_curve - qhat0_curve
    s_diff_curve = s1_curve - s0_curve
    t_grid = np.asarray(bridge.get("t_grid", np.arange(qhat1_curve.shape[1], dtype=float)), dtype=float).ravel()

    return {
        "qhat1_pts": _curve_anchor_values(qhat1_curve),
        "qhat0_pts": _curve_anchor_values(qhat0_curve),
        "qhat_diff_pts": _curve_anchor_values(qhat_diff_curve),
        "qhat_auc_stats": np.column_stack(
            [
                _curve_auc(qhat1_curve, t_grid),
                _curve_auc(qhat0_curve, t_grid),
                _curve_auc(qhat_diff_curve, t_grid),
            ]
        ),
        "qhat_diff_windows": _curve_window_means(qhat_diff_curve),
        "s_auc_stats": np.column_stack(
            [
                _curve_auc(s1_curve, t_grid),
                _curve_auc(s0_curve, t_grid),
                _curve_auc(s_diff_curve, t_grid),
            ]
        ),
        "s_diff_windows": _curve_window_means(s_diff_curve),
        "c_stats": np.column_stack(
            [
                c_curve[:, -1],
                np.mean(c_curve, axis=1),
                np.min(c_curve, axis=1),
                np.mean(1.0 / np.maximum(c_curve, 1e-6), axis=1),
            ]
        ),
    }


def _build_oldc3_survival_ablation_features(
    X,
    W_raw,
    Z_raw,
    bridge,
    *,
    include_raw_proxy: bool,
    surv_scalar_mode: str,
):
    x = _ensure_2d(X).astype(float)
    parts = [x]
    if include_raw_proxy:
        parts.extend([_ensure_2d(W_raw).astype(float), _ensure_2d(Z_raw).astype(float)])
    if surv_scalar_mode == "raw":
        return np.hstack(parts)
    if surv_scalar_mode == "raw_surv":
        parts.extend(
            [
                np.asarray(bridge["surv1_pred"], dtype=float).reshape(-1, 1),
                np.asarray(bridge["surv0_pred"], dtype=float).reshape(-1, 1),
                np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1),
            ]
        )
        return np.hstack(parts)
    parts.extend(
        [
            np.asarray(bridge["q_pred"], dtype=float).reshape(-1, 1),
            np.asarray(bridge["h1_pred"], dtype=float).reshape(-1, 1),
            np.asarray(bridge["h0_pred"], dtype=float).reshape(-1, 1),
            np.asarray(bridge["m_pred"], dtype=float).reshape(-1, 1),
        ]
    )
    if _mode_uses_surv_scalar(surv_scalar_mode):
        parts.extend(
            [
                np.asarray(bridge["surv1_pred"], dtype=float).reshape(-1, 1),
                np.asarray(bridge["surv0_pred"], dtype=float).reshape(-1, 1),
            ]
        )
    if surv_scalar_mode not in {"none", "pair", "raw"}:
        parts.append(np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1))
    parts.extend(_survival_feature_extras(bridge, surv_scalar_mode))
    return np.hstack(parts)


def _rcsf_final_feature_columns(x_final) -> list[str]:
    return [f"f{j}" for j in range(np.asarray(x_final, dtype=float).shape[1])]


def _train_r_csf_final_forest(
    x_final,
    y_res,
    a_res,
    *,
    num_trees: int,
    min_node_size: int,
    seed: int,
):
    feature_cols = _rcsf_final_feature_columns(x_final)
    tempdir = tempfile.TemporaryDirectory(prefix="r_csf_final_forest_")
    tempdir_path = Path(tempdir.name)
    train_path = tempdir_path / "train.csv"
    model_path = tempdir_path / "forest.rds"

    train_df = pd.DataFrame(np.asarray(x_final, dtype=float), columns=feature_cols)
    train_df["Y"] = np.asarray(y_res, dtype=float).ravel()
    train_df["A"] = np.asarray(a_res, dtype=float).ravel()
    train_df.to_csv(train_path, index=False)

    cmd = [
        resolve_rscript(),
        str(R_CSF_FINAL_FOREST_SCRIPT),
        "train",
        str(train_path),
        ",".join(feature_cols),
        str(int(num_trees)),
        str(int(min_node_size)),
        str(model_path),
        str(int(seed)),
    ]
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        tempdir.cleanup()
        raise RuntimeError(
            "R causal_survival_forest final-stage training failed.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return tempdir, model_path, feature_cols


def _predict_r_csf_final_forest(
    model_path: Path,
    x_final,
    *,
    feature_cols: list[str],
):
    with tempfile.TemporaryDirectory(prefix="r_csf_final_predict_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        predict_path = tmp_dir_path / "predict.csv"
        output_path = tmp_dir_path / "predictions.csv"
        pd.DataFrame(np.asarray(x_final, dtype=float), columns=feature_cols).to_csv(
            predict_path,
            index=False,
        )
        cmd = [
            resolve_rscript(),
            str(R_CSF_FINAL_FOREST_SCRIPT),
            "predict",
            str(model_path),
            str(predict_path),
            ",".join(feature_cols),
            str(output_path),
        ]
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "R causal_survival_forest final-stage prediction failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
        )
        return pd.read_csv(output_path)["prediction"].to_numpy(dtype=float)


def _train_r_csf_survival_final_forest(
    x_final,
    time_obs,
    event,
    a,
    w_hat,
    *,
    target: str,
    horizon: float,
    num_trees: int,
    min_node_size: int,
    seed: int,
):
    feature_cols = _rcsf_final_feature_columns(x_final)
    tempdir = tempfile.TemporaryDirectory(prefix="r_csf_survival_final_forest_")
    tempdir_path = Path(tempdir.name)
    train_path = tempdir_path / "train.csv"
    model_path = tempdir_path / "forest.rds"

    train_df = pd.DataFrame(np.asarray(x_final, dtype=float), columns=feature_cols)
    train_df["time"] = np.asarray(time_obs, dtype=float).ravel()
    train_df["event"] = np.asarray(event, dtype=float).ravel()
    train_df["A"] = np.asarray(a, dtype=float).ravel()
    train_df["W_hat"] = np.asarray(w_hat, dtype=float).ravel()
    train_df.to_csv(train_path, index=False)

    cmd = [
        resolve_rscript(),
        str(R_CSF_SURVIVAL_FINAL_FOREST_SCRIPT),
        "train",
        str(train_path),
        ",".join(feature_cols),
        str(target),
        str(float(horizon)),
        str(int(num_trees)),
        str(int(min_node_size)),
        str(model_path),
        str(int(seed)),
    ]
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        tempdir.cleanup()
        raise RuntimeError(
            "R causal_survival_forest final-stage training failed.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return tempdir, model_path, feature_cols


def _predict_r_csf_survival_final_forest(
    model_path: Path,
    x_final,
    *,
    feature_cols: list[str],
):
    with tempfile.TemporaryDirectory(prefix="r_csf_survival_final_predict_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        predict_path = tmp_dir_path / "predict.csv"
        output_path = tmp_dir_path / "predictions.csv"
        pd.DataFrame(np.asarray(x_final, dtype=float), columns=feature_cols).to_csv(
            predict_path,
            index=False,
        )
        cmd = [
            resolve_rscript(),
            str(R_CSF_SURVIVAL_FINAL_FOREST_SCRIPT),
            "predict",
            str(model_path),
            str(predict_path),
            ",".join(feature_cols),
            str(output_path),
        ]
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "R causal_survival_forest final-stage prediction failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        return pd.read_csv(output_path)["prediction"].to_numpy(dtype=float)


def _oldc3_ablation_feature_mode(*, include_raw_proxy: bool, surv_scalar_mode: str) -> str:
    if surv_scalar_mode == "raw":
        return "xwz" if include_raw_proxy else "x_only"
    if surv_scalar_mode in {"full", "raw_surv", "qhat_diff_pts", "qhat_full_pts", "qhat_stats", "s_stats", "censor_stats", "disagreement", "qhat_censor", "qhat_disagree", "all_lite"}:
        count = _SURV_SCALAR_SUMMARY_COUNTS[surv_scalar_mode]
        return f"augmented_custom_{count}" if include_raw_proxy else f"summary_custom_{count}"
    if include_raw_proxy and surv_scalar_mode == "pair":
        return "augmented_surv_pair"
    if include_raw_proxy and surv_scalar_mode == "none":
        return "augmented_minimal"
    if (not include_raw_proxy) and surv_scalar_mode == "full":
        return "summary_surv"
    if (not include_raw_proxy) and surv_scalar_mode == "pair":
        return "summary_surv_pair"
    return "summary_minimal"


def _crossfit_oldc3_survival_ablation_arrays(owner, X, A, y_packed, W, Z):
    x = _ensure_2d(X).astype(float)
    a = np.asarray(A, dtype=float).ravel()
    y = np.asarray(y_packed, dtype=float)
    raw_w = _ensure_2d(W).astype(float)
    raw_z = _ensure_2d(Z).astype(float)
    w_nuis, z_nuis = owner._prepare_nuisance_inputs(W, Z)

    x_final = None
    y_res = np.empty(len(a), dtype=float)
    a_res = np.empty(len(a), dtype=float)

    splitter = KFold(n_splits=owner._cv, shuffle=True, random_state=owner._random_state)
    for train_idx, test_idx in splitter.split(x):
        nuisance = owner._make_nuisance()
        nuisance.train(
            False,
            None,
            y[train_idx],
            a[train_idx],
            X=x[train_idx],
            W=w_nuis[train_idx],
            Z=z_nuis[train_idx],
        )
        y_fold, a_fold = nuisance.predict(
            y[test_idx],
            a[test_idx],
            X=x[test_idx],
            W=w_nuis[test_idx],
            Z=z_nuis[test_idx],
        )
        bridge = nuisance.predict_bridge_outputs(
            X=x[test_idx],
            W=w_nuis[test_idx],
            Z=z_nuis[test_idx],
        )
        x_final_fold = _build_oldc3_survival_ablation_features(
            x[test_idx],
            raw_w[test_idx],
            raw_z[test_idx],
            bridge,
            include_raw_proxy=owner._include_raw_proxy,
            surv_scalar_mode=owner._surv_scalar_mode,
        )
        if x_final is None:
            x_final = np.empty((len(a), x_final_fold.shape[1]), dtype=float)
        x_final[test_idx] = x_final_fold
        y_res[test_idx] = np.asarray(y_fold, dtype=float).ravel()
        a_res[test_idx] = np.asarray(a_fold, dtype=float).ravel()

    return x, raw_w, raw_z, w_nuis, z_nuis, x_final, y_res, a_res


class _BaseOldC3FeatureGRFCensoredSurvivalForest:
    def __init__(
        self,
        *,
        include_raw_proxy: bool,
        include_surv_scalar: bool,
        surv_scalar_mode: str | None = None,
        observed_only: bool = False,
        target="RMST",
        horizon=None,
        n_estimators=200,
        min_samples_leaf=20,
        cv=3,
        random_state=42,
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
        n_jobs=1,
        nuisance_feature_mode="dup",
        forest_max_depth=None,
        forest_honest=True,
        forest_inference=True,
        forest_fit_intercept=True,
    ):
        self._include_raw_proxy = bool(include_raw_proxy)
        self._surv_scalar_mode = _resolve_surv_scalar_mode(include_surv_scalar, surv_scalar_mode)
        self._include_surv_scalar = _mode_uses_surv_scalar(self._surv_scalar_mode)
        self._observed_only = bool(observed_only)
        self._target = target
        self._horizon = horizon
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._cv = int(cv)
        self._random_state = int(random_state)
        self._q_kind = q_kind
        self._q_trees = int(q_trees)
        self._q_min_samples_leaf = int(q_min_samples_leaf)
        self._q_poly_degree = int(q_poly_degree)
        self._q_clip = float(q_clip)
        self._y_tilde_clip_quantile = y_tilde_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._h_kind = h_kind
        self._h_n_estimators = int(h_n_estimators)
        self._h_min_samples_leaf = int(h_min_samples_leaf)
        self._censoring_estimator = censoring_estimator
        self._n_jobs = int(n_jobs)
        self._nuisance_feature_mode = nuisance_feature_mode
        self._forest_max_depth = forest_max_depth
        self._forest_honest = bool(forest_honest)
        self._forest_inference = bool(forest_inference)
        self._forest_fit_intercept = bool(forest_fit_intercept)

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
        self._forest = None
        self._feature_nuisance = None

    def _prepare_nuisance_inputs(self, W, Z):
        w = _ensure_2d(W).astype(float)
        z = _ensure_2d(Z).astype(float)
        if self._observed_only:
            w = np.zeros_like(w)
            z = np.zeros_like(z)
        return w, z

    def _make_nuisance(self):
        return _MildShrinkNCSurvivalNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            target=self._target,
            horizon=self._horizon,
            final_feature_mode="x_only",
            nuisance_feature_mode=self._nuisance_feature_mode,
            censoring_estimator=self._censoring_estimator,
            q_clip=self._q_clip,
            y_tilde_clip_quantile=self._y_tilde_clip_quantile,
            y_res_clip_percentiles=self._y_res_clip_percentiles,
        )

    def _make_nuisance(self):
        return self._make_feature_nuisance()

    def fit_components(self, X, A, time, event, Z, W):
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        x, _, _, w_nuis, z_nuis, x_final, y_res, a_res = _crossfit_oldc3_survival_ablation_arrays(
            self,
            X,
            A,
            y_packed,
            W,
            Z,
        )
        a = np.asarray(A, dtype=float).ravel()

        self._forest = CausalForest(
            n_estimators=self._n_estimators,
            criterion="het",
            min_samples_leaf=self._min_samples_leaf,
            max_depth=self._forest_max_depth,
            honest=self._forest_honest,
            inference=self._forest_inference,
            fit_intercept=self._forest_fit_intercept,
            n_jobs=1,
            random_state=self._random_state,
        )
        self._forest.fit(x_final, a_res, y_res)

        self._feature_nuisance = self._make_nuisance()
        self._feature_nuisance.train(
            False,
            None,
            y_packed,
            a,
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._forest is None or self._feature_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._feature_nuisance.predict_bridge_outputs(
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        x_final = _build_oldc3_survival_ablation_features(
            x,
            raw_w,
            raw_z,
            bridge,
            include_raw_proxy=self._include_raw_proxy,
            surv_scalar_mode=self._surv_scalar_mode,
        )
        return self._forest.predict(x_final)


class _BaseOldC3FeatureDMLCensoredSurvivalForest:
    def __init__(
        self,
        *,
        include_raw_proxy: bool,
        include_surv_scalar: bool,
        surv_scalar_mode: str | None = None,
        observed_only: bool = False,
        target="RMST",
        horizon=None,
        n_estimators=200,
        min_samples_leaf=20,
        cv=3,
        random_state=42,
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
        n_jobs=1,
        nuisance_feature_mode="dup",
    ):
        self._include_raw_proxy = bool(include_raw_proxy)
        self._surv_scalar_mode = _resolve_surv_scalar_mode(include_surv_scalar, surv_scalar_mode)
        self._include_surv_scalar = _mode_uses_surv_scalar(self._surv_scalar_mode)
        self._observed_only = bool(observed_only)
        self._target = target
        self._horizon = horizon
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._cv = int(cv)
        self._random_state = int(random_state)
        self._q_kind = q_kind
        self._q_trees = int(q_trees)
        self._q_min_samples_leaf = int(q_min_samples_leaf)
        self._q_poly_degree = int(q_poly_degree)
        self._q_clip = float(q_clip)
        self._y_tilde_clip_quantile = y_tilde_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._h_kind = h_kind
        self._h_n_estimators = int(h_n_estimators)
        self._h_min_samples_leaf = int(h_min_samples_leaf)
        self._censoring_estimator = censoring_estimator
        self._n_jobs = int(n_jobs)
        self._nuisance_feature_mode = nuisance_feature_mode

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
        self._feature_nuisance = None
        self._dml_model = None
        self._final_feature_mode = _oldc3_ablation_feature_mode(
            include_raw_proxy=self._include_raw_proxy,
            surv_scalar_mode=self._surv_scalar_mode,
        )

    def _prepare_nuisance_inputs(self, W, Z):
        w = _ensure_2d(W).astype(float)
        z = _ensure_2d(Z).astype(float)
        if self._observed_only:
            w = np.zeros_like(w)
            z = np.zeros_like(z)
        return w, z

    def _make_feature_nuisance(self):
        return _MildShrinkNCSurvivalNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            target=self._target,
            horizon=self._horizon,
            final_feature_mode="x_only",
            nuisance_feature_mode=self._nuisance_feature_mode,
            censoring_estimator=self._censoring_estimator,
            q_clip=self._q_clip,
            y_tilde_clip_quantile=self._y_tilde_clip_quantile,
            y_res_clip_percentiles=self._y_res_clip_percentiles,
        )

    def _make_nuisance(self):
        return self._make_feature_nuisance()

    def _make_nuisance(self):
        return self._make_feature_nuisance()

    def _make_nuisance(self):
        return self._make_feature_nuisance()

    def fit_components(self, X, A, time, event, Z, W):
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        x, _, _, w_nuis, z_nuis, x_final, _, _ = _crossfit_oldc3_survival_ablation_arrays(
            self,
            X,
            A,
            y_packed,
            W,
            Z,
        )
        self._dml_model = EconmlMildShrinkNCSurvivalForest(
            target=self._target,
            horizon=self._horizon,
            n_estimators=self._n_estimators,
            min_samples_leaf=self._min_samples_leaf,
            cv=self._cv,
            random_state=self._random_state,
            final_feature_mode=self._final_feature_mode,
            nuisance_feature_mode=self._nuisance_feature_mode,
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
            n_jobs=self._n_jobs,
        )
        self._dml_model.fit_survival(x_final, A, time, event, z_nuis, w_nuis)

        self._feature_nuisance = self._make_feature_nuisance()
        self._feature_nuisance.train(
            False,
            None,
            y_packed,
            np.asarray(A, dtype=float).ravel(),
            X=_ensure_2d(X).astype(float),
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._dml_model is None or self._feature_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._feature_nuisance.predict_bridge_outputs(
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        x_final = _build_oldc3_survival_ablation_features(
            x,
            raw_w,
            raw_z,
            bridge,
            include_raw_proxy=self._include_raw_proxy,
            surv_scalar_mode=self._surv_scalar_mode,
        )
        return self._dml_model.effect(x_final)


class _BridgeOutputSurvivalNuisance(_MildShrinkNCSurvivalNuisance):
    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_res, a_res = super().predict(
            Y,
            T,
            X=X,
            W=W,
            Z=Z,
            sample_weight=sample_weight,
            groups=groups,
        )
        bridge = super().predict_bridge_outputs(X=X, W=W, Z=Z)
        summary = _bridge_scalar_summaries(bridge)
        return (
            y_res,
            a_res,
            bridge["q_pred"],
            bridge["h1_pred"],
            bridge["h0_pred"],
            bridge["m_pred"],
            bridge["m_bridge_pred"],
            bridge["m_survival_pred"],
            bridge["m_gap_pred"],
            bridge["h_diff_pred"],
            bridge["surv1_pred"],
            bridge["surv0_pred"],
            bridge["surv_diff_pred"],
            summary["qhat1_pts"],
            summary["qhat0_pts"],
            summary["qhat_diff_pts"],
            summary["qhat_auc_stats"],
            summary["qhat_diff_windows"],
            summary["s_auc_stats"],
            summary["s_diff_windows"],
            summary["c_stats"],
        )


class _BridgeFeatureSurvivalModelFinal:
    def __init__(
        self,
        base_model_final,
        *,
        include_raw_proxy: bool,
        include_surv_scalar: bool,
        surv_scalar_mode: str | None = None,
        raw_proxy_supplier=None,
    ):
        self._base_model_final = base_model_final
        self._include_raw_proxy = bool(include_raw_proxy)
        self._surv_scalar_mode = _resolve_surv_scalar_mode(include_surv_scalar, surv_scalar_mode)
        self._include_surv_scalar = _mode_uses_surv_scalar(self._surv_scalar_mode)
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
            m_bridge_pred,
            m_survival_pred,
            m_gap_pred,
            h_diff_pred,
            surv1_pred,
            surv0_pred,
            surv_diff_pred,
            qhat1_pts,
            qhat0_pts,
            qhat_diff_pts,
            qhat_auc_stats,
            qhat_diff_windows,
            s_auc_stats,
            s_diff_windows,
            c_stats,
        ) = nuisances
        bridge = {
            "q_pred": q_pred,
            "h1_pred": h1_pred,
            "h0_pred": h0_pred,
            "m_pred": m_pred,
            "m_bridge_pred": m_bridge_pred,
            "m_survival_pred": m_survival_pred,
            "m_gap_pred": m_gap_pred,
            "h_diff_pred": h_diff_pred,
            "surv1_pred": surv1_pred,
            "surv0_pred": surv0_pred,
            "surv_diff_pred": surv_diff_pred,
            "qhat1_pts": qhat1_pts,
            "qhat0_pts": qhat0_pts,
            "qhat_diff_pts": qhat_diff_pts,
            "qhat_auc_stats": qhat_auc_stats,
            "qhat_diff_windows": qhat_diff_windows,
            "s_auc_stats": s_auc_stats,
            "s_diff_windows": s_diff_windows,
            "c_stats": c_stats,
        }
        w_for_final = W
        z_for_final = Z
        if self._raw_proxy_supplier is not None:
            supplied = self._raw_proxy_supplier(X=X, W=W, Z=Z)
            if supplied is not None:
                w_for_final, z_for_final = supplied
        x_final = _build_oldc3_survival_ablation_features(
            X,
            w_for_final,
            z_for_final,
            bridge,
            include_raw_proxy=self._include_raw_proxy,
            surv_scalar_mode=self._surv_scalar_mode,
        )
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

    def score(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, groups=None, scoring="mean_squared_error"):
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


class SinglePassBridgeFeatureCensoredSurvivalForest(EconmlMildShrinkNCSurvivalForest):
    def __init__(self, *, include_raw_proxy: bool, include_surv_scalar: bool, surv_scalar_mode: str | None = None, **kwargs):
        self._include_raw_proxy = bool(include_raw_proxy)
        self._surv_scalar_mode = _resolve_surv_scalar_mode(include_surv_scalar, surv_scalar_mode)
        self._include_surv_scalar = _mode_uses_surv_scalar(self._surv_scalar_mode)
        self._raw_w_for_final = None
        self._raw_z_for_final = None
        kwargs.setdefault(
            "final_feature_mode",
            _oldc3_ablation_feature_mode(
                include_raw_proxy=self._include_raw_proxy,
                surv_scalar_mode=self._surv_scalar_mode,
            ),
        )
        super().__init__(**kwargs)

    def _raw_proxy_for_final(self, *, X=None, W=None, Z=None):
        if X is None:
            return W, Z
        if self._raw_w_for_final is None or self._raw_z_for_final is None:
            return W, Z
        if len(np.asarray(X)) != len(self._raw_w_for_final):
            return W, Z
        return self._raw_w_for_final, self._raw_z_for_final

    def _gen_ortho_learner_model_nuisance(self):
        return _BridgeOutputSurvivalNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            target=self._target,
            horizon=self._horizon,
            final_feature_mode="x_only",
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

    def _gen_ortho_learner_model_final(self):
        return _BridgeFeatureSurvivalModelFinal(
            super()._gen_ortho_learner_model_final(),
            include_raw_proxy=self._include_raw_proxy,
            surv_scalar_mode=self._surv_scalar_mode,
            include_surv_scalar=self._include_surv_scalar,
            raw_proxy_supplier=self._raw_proxy_for_final,
        )

    def effect_on_final_features(self, X_final):
        return np.asarray(self._ortho_learner_model_final.predict(X_final), dtype=float)

    def training_x_final(self):
        return self._ortho_learner_model_final.training_x_final()


class _BaseSinglePassBridgeFeatureCensoredSurvivalForest:
    def __init__(
        self,
        *,
        include_raw_proxy: bool,
        include_surv_scalar: bool,
        surv_scalar_mode: str | None = None,
        observed_only: bool = False,
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
        q_clip=0.02,
        y_tilde_clip_quantile=0.99,
        y_res_clip_percentiles=(1.0, 99.0),
        h_kind="extra",
        h_n_estimators=1200,
        h_min_samples_leaf=3,
        censoring_estimator="kaplan-meier",
        event_survival_estimator="cox",
        m_pred_mode="bridge",
        survival_forest_num_trees=50,
        survival_forest_min_node_size=15,
        survival_fast_logrank=False,
        enforce_finite_horizon=False,
        n_jobs=1,
        nuisance_feature_mode="dup",
        prediction_nuisance_mode="full_refit",
    ):
        self._include_raw_proxy = bool(include_raw_proxy)
        self._surv_scalar_mode = _resolve_surv_scalar_mode(include_surv_scalar, surv_scalar_mode)
        self._include_surv_scalar = _mode_uses_surv_scalar(self._surv_scalar_mode)
        self._observed_only = bool(observed_only)
        self._target = target
        self._horizon = horizon
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._cv = int(cv)
        self._random_state = int(random_state)
        self._q_kind = q_kind
        self._q_trees = int(q_trees)
        self._q_min_samples_leaf = int(q_min_samples_leaf)
        self._q_poly_degree = int(q_poly_degree)
        self._q_clip = float(q_clip)
        self._y_tilde_clip_quantile = y_tilde_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._h_kind = h_kind
        self._h_n_estimators = int(h_n_estimators)
        self._h_min_samples_leaf = int(h_min_samples_leaf)
        self._censoring_estimator = censoring_estimator
        self._event_survival_estimator = event_survival_estimator
        self._m_pred_mode = m_pred_mode
        self._survival_forest_num_trees = int(survival_forest_num_trees)
        self._survival_forest_min_node_size = int(survival_forest_min_node_size)
        self._survival_fast_logrank = bool(survival_fast_logrank)
        self._enforce_finite_horizon = bool(enforce_finite_horizon)
        self._n_jobs = int(n_jobs)
        self._nuisance_feature_mode = nuisance_feature_mode
        self._prediction_nuisance_mode = prediction_nuisance_mode
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
        self._feature_nuisance = None
        self._dml_model = None
        self._train_x = None
        self._train_w = None
        self._train_z = None
        self._train_x_final = None

    def _prepare_nuisance_inputs(self, W, Z):
        w = _ensure_2d(W).astype(float)
        z = _ensure_2d(Z).astype(float)
        if self._observed_only:
            w = np.zeros_like(w)
            z = np.zeros_like(z)
        return w, z

    def _make_feature_nuisance(self):
        return _MildShrinkNCSurvivalNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            target=self._target,
            horizon=self._horizon,
            final_feature_mode="x_only",
            nuisance_feature_mode=self._nuisance_feature_mode,
            censoring_estimator=self._censoring_estimator,
            q_clip=self._q_clip,
            y_tilde_clip_quantile=self._y_tilde_clip_quantile,
            y_res_clip_percentiles=self._y_res_clip_percentiles,
            event_survival_estimator=self._event_survival_estimator,
            m_pred_mode=self._m_pred_mode,
            survival_forest_num_trees=self._survival_forest_num_trees,
            survival_forest_min_node_size=self._survival_forest_min_node_size,
            survival_fast_logrank=self._survival_fast_logrank,
            n_jobs=self._n_jobs,
            random_state=self._random_state,
            enforce_finite_horizon=self._enforce_finite_horizon,
        )

    def _make_nuisance(self):
        return self._make_feature_nuisance()

    def fit_components(self, X, A, time, event, Z, W):
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)

        self._dml_model = SinglePassBridgeFeatureCensoredSurvivalForest(
            include_raw_proxy=self._include_raw_proxy,
            include_surv_scalar=self._include_surv_scalar,
            surv_scalar_mode=self._surv_scalar_mode,
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
            survival_forest_num_trees=self._survival_forest_num_trees,
            survival_forest_min_node_size=self._survival_forest_min_node_size,
            survival_fast_logrank=self._survival_fast_logrank,
            enforce_finite_horizon=self._enforce_finite_horizon,
            n_jobs=self._n_jobs,
            nuisance_feature_mode=self._nuisance_feature_mode,
        )
        self._dml_model._raw_w_for_final = raw_w
        self._dml_model._raw_z_for_final = raw_z
        self._dml_model.fit_survival(x, A, time, event, z_nuis, w_nuis)
        self._train_x = np.asarray(x, dtype=float).copy()
        self._train_w = np.asarray(raw_w, dtype=float).copy()
        self._train_z = np.asarray(raw_z, dtype=float).copy()
        self._train_x_final = self._dml_model.training_x_final()

        if self._prediction_nuisance_mode == "full_refit":
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
        elif self._prediction_nuisance_mode not in {"fold_ensemble", "cached_oof"}:
            raise ValueError(f"Unsupported prediction nuisance mode: {self._prediction_nuisance_mode}")
        return self

    def _predict_bridge_from_fold_ensemble(self, X, W, Z):
        model_groups = getattr(self._dml_model, "_models_nuisance", None)
        if not model_groups:
            raise RuntimeError("Fold nuisance models are unavailable for ensemble prediction.")
        fold_models = []
        for group in model_groups:
            if isinstance(group, list):
                fold_models.extend(group)
            else:
                fold_models.append(group)
        if not fold_models:
            raise RuntimeError("No fitted fold nuisance models were found.")
        preds = [model.predict_bridge_outputs(X=X, W=W, Z=Z) for model in fold_models]
        keys = preds[0].keys()
        return {key: np.mean(np.stack([np.asarray(pred[key], dtype=float) for pred in preds], axis=0), axis=0) for key in keys}

    def _matches_training_data(self, x, w, z):
        if self._train_x is None or self._train_w is None or self._train_z is None:
            return False
        return (
            x.shape == self._train_x.shape
            and w.shape == self._train_w.shape
            and z.shape == self._train_z.shape
            and np.array_equal(x, self._train_x)
            and np.array_equal(w, self._train_w)
            and np.array_equal(z, self._train_z)
        )

    def effect_from_components(self, X, W, Z):
        if self._dml_model is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        if self._prediction_nuisance_mode == "full_refit":
            if self._feature_nuisance is None:
                raise RuntimeError("Full nuisance retrain is unavailable.")
            bridge = self._feature_nuisance.predict_bridge_outputs(
                X=x,
                W=w_nuis,
                Z=z_nuis,
            )
            x_final = _build_oldc3_survival_ablation_features(
                x,
                raw_w,
                raw_z,
                bridge,
                include_raw_proxy=self._include_raw_proxy,
                surv_scalar_mode=self._surv_scalar_mode,
            )
        elif self._prediction_nuisance_mode == "fold_ensemble":
            bridge = self._predict_bridge_from_fold_ensemble(x, w_nuis, z_nuis)
            x_final = _build_oldc3_survival_ablation_features(
                x,
                raw_w,
                raw_z,
                bridge,
                include_raw_proxy=self._include_raw_proxy,
                surv_scalar_mode=self._surv_scalar_mode,
            )
        elif self._prediction_nuisance_mode == "cached_oof":
            if self._matches_training_data(x, raw_w, raw_z):
                if self._train_x_final is None:
                    raise RuntimeError("Cached training final features are unavailable.")
                x_final = np.asarray(self._train_x_final, dtype=float)
            else:
                bridge = self._predict_bridge_from_fold_ensemble(x, w_nuis, z_nuis)
                x_final = _build_oldc3_survival_ablation_features(
                    x,
                    raw_w,
                    raw_z,
                    bridge,
                    include_raw_proxy=self._include_raw_proxy,
                    surv_scalar_mode=self._surv_scalar_mode,
                )
        else:
            raise ValueError(f"Unsupported prediction nuisance mode: {self._prediction_nuisance_mode}")
        return self._dml_model.effect_on_final_features(x_final)


class OldC3SummaryMinimalGRFCensoredSurvivalForest(_BaseOldC3FeatureGRFCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", False)
        kwargs.setdefault("include_surv_scalar", False)
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class OldC3SummaryMinimalObservedGRFCensoredSurvivalForest(_BaseOldC3FeatureGRFCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", False)
        kwargs.setdefault("include_surv_scalar", False)
        kwargs.setdefault("observed_only", True)
        super().__init__(*args, **kwargs)


class OldC3AugmentedMinimalGRFCensoredSurvivalForest(_BaseOldC3FeatureGRFCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", False)
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class OldC3AugmentedMinimalObservedGRFCensoredSurvivalForest(_BaseOldC3FeatureGRFCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", False)
        kwargs.setdefault("observed_only", True)
        super().__init__(*args, **kwargs)


class OldC3SummarySurvGRFCensoredSurvivalForest(_BaseOldC3FeatureGRFCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", False)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class OldC3SummarySurvObservedGRFCensoredSurvivalForest(_BaseOldC3FeatureGRFCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", False)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("observed_only", True)
        super().__init__(*args, **kwargs)


class OldC3AugmentedSurvGRFCensoredSurvivalForest(_BaseOldC3FeatureGRFCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class OldC3AugmentedSurvObservedGRFCensoredSurvivalForest(_BaseOldC3FeatureGRFCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("observed_only", True)
        super().__init__(*args, **kwargs)


class OldC3SummaryMinimalDMLCensoredSurvivalForest(_BaseOldC3FeatureDMLCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", False)
        kwargs.setdefault("include_surv_scalar", False)
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class OldC3SummaryMinimalObservedDMLCensoredSurvivalForest(_BaseOldC3FeatureDMLCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", False)
        kwargs.setdefault("include_surv_scalar", False)
        kwargs.setdefault("observed_only", True)
        super().__init__(*args, **kwargs)


class OldC3AugmentedMinimalDMLCensoredSurvivalForest(_BaseOldC3FeatureDMLCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", False)
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class OldC3AugmentedMinimalObservedDMLCensoredSurvivalForest(_BaseOldC3FeatureDMLCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", False)
        kwargs.setdefault("observed_only", True)
        super().__init__(*args, **kwargs)


class OldC3SummarySurvDMLCensoredSurvivalForest(_BaseOldC3FeatureDMLCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", False)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class OldC3SummarySurvObservedDMLCensoredSurvivalForest(_BaseOldC3FeatureDMLCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", False)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("observed_only", True)
        super().__init__(*args, **kwargs)


class OldC3AugmentedSurvDMLCensoredSurvivalForest(_BaseOldC3FeatureDMLCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class OldC3AugmentedSurvObservedDMLCensoredSurvivalForest(_BaseOldC3FeatureDMLCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("observed_only", True)
        super().__init__(*args, **kwargs)


class B2SummaryCensoredSurvivalForest(OldC3AugmentedSurvGRFCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "dup")
        super().__init__(*args, **kwargs)


class B2SummaryObservedCensoredSurvivalForest(OldC3AugmentedSurvObservedGRFCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "dup")
        super().__init__(*args, **kwargs)


class B2SummaryDMLCensoredSurvivalForest(OldC3AugmentedSurvDMLCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "dup")
        super().__init__(*args, **kwargs)


class B2SummaryDMLObservedCensoredSurvivalForest(OldC3AugmentedSurvObservedDMLCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "dup")
        super().__init__(*args, **kwargs)


class B2SummaryBaselineDMLCensoredSurvivalForest:
    def __init__(
        self,
        *,
        include_raw_proxy: bool = True,
        surv_scalar_mode: str | None = "pair",
        observed_only: bool = False,
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
        q_clip=0.02,
        y_tilde_clip_quantile=0.99,
        y_res_clip_percentiles=(1.0, 99.0),
        h_kind="extra",
        h_n_estimators=1200,
        h_min_samples_leaf=3,
        censoring_estimator="nelson-aalen",
        n_jobs=1,
        nuisance_feature_mode="dup",
    ):
        self._include_raw_proxy = bool(include_raw_proxy)
        self._surv_scalar_mode = _resolve_surv_scalar_mode(True, surv_scalar_mode)
        self._observed_only = bool(observed_only)
        self._target = target
        self._horizon = horizon
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._cv = int(cv)
        self._random_state = int(random_state)
        self._q_kind = q_kind
        self._q_trees = int(q_trees)
        self._q_min_samples_leaf = int(q_min_samples_leaf)
        self._q_poly_degree = int(q_poly_degree)
        self._q_clip = float(q_clip)
        self._y_tilde_clip_quantile = y_tilde_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._h_kind = h_kind
        self._h_n_estimators = int(h_n_estimators)
        self._h_min_samples_leaf = int(h_min_samples_leaf)
        self._censoring_estimator = censoring_estimator
        self._n_jobs = int(n_jobs)
        self._nuisance_feature_mode = nuisance_feature_mode
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
        self._baseline = None
        self._feature_nuisance = None

    def _prepare_nuisance_inputs(self, W, Z):
        w = _ensure_2d(W).astype(float)
        z = _ensure_2d(Z).astype(float)
        if self._observed_only:
            w = np.zeros_like(w)
            z = np.zeros_like(z)
        return w, z

    def _make_feature_nuisance(self):
        return _MildShrinkNCSurvivalNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            target=self._target,
            horizon=self._horizon,
            final_feature_mode="x_only",
            nuisance_feature_mode=self._nuisance_feature_mode,
            censoring_estimator=self._censoring_estimator,
            q_clip=self._q_clip,
            y_tilde_clip_quantile=self._y_tilde_clip_quantile,
            y_res_clip_percentiles=self._y_res_clip_percentiles,
        )

    def _make_nuisance(self):
        return self._make_feature_nuisance()

    def fit_components(self, X, A, time, event, Z, W):
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        x, _, _, w_nuis, z_nuis, x_final, _, _ = _crossfit_oldc3_survival_ablation_arrays(
            self,
            X,
            A,
            y_packed,
            W,
            Z,
        )

        self._baseline = BaselineCensoredCausalForestDML(
            target=self._target,
            horizon=self._horizon,
            n_estimators=self._n_estimators,
            min_samples_leaf=self._min_samples_leaf,
            cv=self._cv,
            random_state=self._random_state,
            q_clip=self._q_clip,
            y_tilde_clip_quantile=self._y_tilde_clip_quantile,
            y_res_clip_percentiles=self._y_res_clip_percentiles,
            censoring_estimator=self._censoring_estimator,
        )
        self._baseline.fit_survival(x_final, A, time, event)

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
        if self._baseline is None or self._feature_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._feature_nuisance.predict_bridge_outputs(
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        x_final = _build_oldc3_survival_ablation_features(
            x,
            raw_w,
            raw_z,
            bridge,
            include_raw_proxy=self._include_raw_proxy,
            surv_scalar_mode=self._surv_scalar_mode,
        )
        return self._baseline.effect(x_final)


class _BaselineCensoredSurvivalNuisance:
    max_grid = 500

    def __init__(
        self,
        q_model,
        h_model,
        *,
        target,
        horizon,
        censoring_estimator,
        q_clip,
        y_tilde_clip_quantile,
        y_res_clip_percentiles,
    ):
        self._q_model_template = q_model
        self._h_model_template = h_model
        self._target = target
        self._horizon = horizon
        self._censoring_estimator = censoring_estimator
        self._q_clip = q_clip
        self._y_tilde_clip_quantile = y_tilde_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles

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

    def train(self, is_selecting, folds, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_time, delta = self._unpack_y(Y)
        a = np.asarray(T).ravel()
        x = _ensure_2d(X).astype(float)
        target_inputs = _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )

        self._censor_model = _fit_censoring_model(
            target_inputs["nuisance_time"],
            target_inputs["nuisance_delta"],
            x,
            estimator=self._censoring_estimator,
        )
        y_tilde_eval_time = (
            target_inputs["eval_time"] if self._target == "survival.probability" else target_inputs["nuisance_time"]
        )
        sc_for_y_tilde = _predict_censoring_survival_at_values(self._censor_model, x, y_tilde_eval_time)
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
        self._q_model.fit(x, a)

        treated_mask = a == 1
        control_mask = a == 0
        self._h1_model = clone(self._h_model_template)
        self._h0_model = clone(self._h_model_template)
        if treated_mask.sum() > 10:
            self._h1_model.fit(x[treated_mask], y_tilde[treated_mask])
        if control_mask.sum() > 10:
            self._h0_model.fit(x[control_mask], y_tilde[control_mask])

        self._event_cox_1, self._cox_col_names_1, self._cox_keep_mask_1 = _fit_event_cox(
            target_inputs["nuisance_time"][treated_mask],
            target_inputs["nuisance_delta"][treated_mask],
            x[treated_mask],
        )
        self._event_cox_0, self._cox_col_names_0, self._cox_keep_mask_0 = _fit_event_cox(
            target_inputs["nuisance_time"][control_mask],
            target_inputs["nuisance_delta"][control_mask],
            x[control_mask],
        )

        all_times = np.sort(np.unique(target_inputs["grid_time"]))
        if len(all_times) > self.max_grid:
            idx = np.linspace(0, len(all_times) - 1, self.max_grid, dtype=int)
            all_times = all_times[idx]
        self._t_grid = all_times
        return self

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_time, delta = self._unpack_y(Y)
        a = np.asarray(T).ravel()
        x = _ensure_2d(X).astype(float)
        target_inputs = _prepare_target_inputs(
            y_time,
            delta,
            target=self._target,
            horizon=self._horizon,
        )

        q_pred = self._q_model.predict_proba(x)[:, 1]
        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        h1_pred = self._h1_model.predict(x)
        h0_pred = self._h0_model.predict(x)
        m_pred = q_pred * h1_pred + (1.0 - q_pred) * h0_pred

        s_hat_1 = _predict_s_on_grid(
            self._event_cox_1,
            self._cox_col_names_1,
            x,
            self._t_grid,
            self._cox_keep_mask_1,
        )
        s_hat_0 = _predict_s_on_grid(
            self._event_cox_0,
            self._cox_col_names_0,
            x,
            self._t_grid,
            self._cox_keep_mask_0,
        )
        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, self._t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, self._t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, self._t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, self._t_grid)
        q_hat = np.where((a == 1)[:, None], q_hat_1, q_hat_0)

        if self._target == "RMST" and self._horizon is None:
            sc_at_eval = _predict_censoring_survival_at_values(self._censor_model, x, y_time)
            sc_grid = _predict_censoring_survival_on_grid(self._censor_model, x, self._t_grid)
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
                x,
                target_inputs["eval_time"],
            )
            sc_grid = _predict_censoring_survival_on_grid(self._censor_model, x, self._t_grid)
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


class BaselineCensoredCausalForestDML(CausalForestDML):
    def __init__(
        self,
        *,
        target="RMST",
        horizon=None,
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=42,
        q_clip=0.02,
        y_tilde_clip_quantile=0.99,
        y_res_clip_percentiles=(1.0, 99.0),
        censoring_estimator="nelson-aalen",
        **kwargs,
    ):
        self._q_model_template = RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=20,
            random_state=random_state,
        )
        self._h_model_template = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=20,
            random_state=random_state,
        )
        self._target = target
        self._horizon = horizon
        self._censoring_estimator = censoring_estimator
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
        return _BaselineCensoredSurvivalNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            target=self._target,
            horizon=self._horizon,
            censoring_estimator=self._censoring_estimator,
            q_clip=self._custom_q_clip,
            y_tilde_clip_quantile=self._custom_y_tilde_clip_quantile,
            y_res_clip_percentiles=self._custom_y_res_clip_percentiles,
        )

    def fit_survival(self, X, A, time, event, **kwargs):
        x = np.asarray(X, dtype=float)
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        return _OrthoLearner.fit(self, y_packed, A, X=x, **kwargs)


class UnifiedB2SumBaselineCensoredSurvivalForest(B2SummaryBaselineDMLCensoredSurvivalForest):
    """Matched censored B2Sum baseline family."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("surv_scalar_mode", "pair")
        kwargs.setdefault("censoring_estimator", "nelson-aalen")
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class RCSFStyleEconmlCensoredBaseline:
    """Strict-like censored baseline with survival/censoring summaries but no explicit bridge residualization."""

    max_grid = 500

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
        event_survival_estimator="cox",
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
        self._event_survival_estimator = event_survival_estimator
        self._extra_kwargs = dict(kwargs)
        self._model = None
        self._t_grid = None
        self._event_cox_1 = None
        self._event_cox_0 = None
        self._cox_col_names_1 = None
        self._cox_col_names_0 = None
        self._cox_keep_mask_1 = None
        self._cox_keep_mask_0 = None

    @staticmethod
    def stack_final_features(*arrays):
        parts = [_ensure_2d(np.asarray(arr, dtype=float)) for arr in arrays]
        return np.hstack(parts)

    def _predict_event_survival_curves(self, x_full):
        if self._event_survival_estimator != "cox":
            raise ValueError("RCSFStyleEconmlCensoredBaseline currently supports event_survival_estimator='cox' only.")
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

    def _build_final_features(self, x, raw_w, raw_z):
        x_full = self.stack_final_features(x, raw_w, raw_z)
        s_hat_1, s_hat_0 = self._predict_event_survival_curves(x_full)
        if self._target == "survival.probability":
            q_hat_1 = _compute_survival_probability_q_from_s(s_hat_1, self._t_grid, self._horizon)
            q_hat_0 = _compute_survival_probability_q_from_s(s_hat_0, self._t_grid, self._horizon)
        else:
            q_hat_1 = _compute_q_from_s(s_hat_1, self._t_grid)
            q_hat_0 = _compute_q_from_s(s_hat_0, self._t_grid)
        bridge = {
            "surv1_pred": q_hat_1[:, 0],
            "surv0_pred": q_hat_0[:, 0],
            "surv_diff_pred": q_hat_1[:, 0] - q_hat_0[:, 0],
        }
        return _build_oldc3_survival_ablation_features(
            x,
            raw_w,
            raw_z,
            bridge,
            include_raw_proxy=True,
            surv_scalar_mode="raw_surv",
        )

    def fit_components(self, X, A, time, event, Z, W):
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_full = self.stack_final_features(x, raw_w, raw_z)
        y_time = np.asarray(time, dtype=float).ravel()
        delta = np.asarray(event, dtype=float).ravel()
        a = np.asarray(A).ravel()

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

        censor_model = _fit_censoring_model(
            target_inputs["nuisance_time"],
            target_inputs["nuisance_delta"],
            x_full,
            estimator=self._censoring_estimator,
        )
        y_tilde_eval_time = (
            target_inputs["eval_time"] if self._target == "survival.probability" else target_inputs["nuisance_time"]
        )
        sc_at_eval = _predict_censoring_survival_at_values(
            censor_model,
            x_full,
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

        treated_mask = a == 1
        control_mask = a == 0
        self._event_cox_1, self._cox_col_names_1, self._cox_keep_mask_1 = _fit_event_cox(
            target_inputs["nuisance_time"][treated_mask],
            target_inputs["nuisance_delta"][treated_mask],
            x_full[treated_mask],
        )
        self._event_cox_0, self._cox_col_names_0, self._cox_keep_mask_0 = _fit_event_cox(
            target_inputs["nuisance_time"][control_mask],
            target_inputs["nuisance_delta"][control_mask],
            x_full[control_mask],
        )

        x_final = self._build_final_features(x, raw_w, raw_z)
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
            X=x_final,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._model is None or self._t_grid is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_final = self._build_final_features(x, raw_w, raw_z)
        return self._model.effect(x_final)


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
            x_full,
            estimator=self._censoring_estimator,
        )
        y_tilde_eval_time = (
            target_inputs["eval_time"]
            if self._target == "survival.probability"
            else target_inputs["nuisance_time"]
        )
        sc_at_eval = _predict_censoring_survival_at_values(
            censor_model,
            x_full,
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


class UnifiedB2SumMildShrinkCensoredSurvivalForest(B2SummaryDMLCensoredSurvivalForest):
    """Matched censored B2Sum mild-shrink family."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("surv_scalar_mode", "pair")
        kwargs.setdefault("censoring_estimator", "nelson-aalen")
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class UnifiedB2SumSinglePassBaselineCensoredSurvivalForest(_BaseSinglePassBridgeFeatureCensoredSurvivalForest):
    """Single-pass baseline-style B2Sum with structured broad proxy duplication."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("surv_scalar_mode", "pair")
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "rf")
        kwargs.setdefault("q_trees", 100)
        kwargs.setdefault("q_min_samples_leaf", 20)
        kwargs.setdefault("h_kind", "rf")
        kwargs.setdefault("h_n_estimators", 100)
        kwargs.setdefault("h_min_samples_leaf", 20)
        kwargs.setdefault("censoring_estimator", "nelson-aalen")
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class FinalModelCensoredSurvivalForest(_BaseSinglePassBridgeFeatureCensoredSurvivalForest):
    """Finalized censored single-pass B2Sum model."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("surv_scalar_mode", "full")
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("target", "RMST")
        kwargs.setdefault("horizon", None)
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 600)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("q_clip", 0.03)
        kwargs.setdefault("y_tilde_clip_quantile", 0.98)
        kwargs.setdefault("y_res_clip_percentiles", (2.0, 98.0))
        kwargs.setdefault("n_estimators", 200)
        kwargs.setdefault("min_samples_leaf", 20)
        kwargs.setdefault("censoring_estimator", "nelson-aalen")
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        kwargs.setdefault("n_jobs", 1)
        super().__init__(*args, **kwargs)


class FinalModelRCSFCensoredSurvivalForest(_BaseSinglePassBridgeFeatureCensoredSurvivalForest):
    """Finalized censored model with an R grf::causal_forest final stage."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("surv_scalar_mode", "pair")
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("target", "RMST")
        kwargs.setdefault("horizon", None)
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 600)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("q_clip", 0.03)
        kwargs.setdefault("y_tilde_clip_quantile", 0.98)
        kwargs.setdefault("y_res_clip_percentiles", (2.0, 98.0))
        kwargs.setdefault("n_estimators", 200)
        kwargs.setdefault("min_samples_leaf", 20)
        kwargs.setdefault("censoring_estimator", "nelson-aalen")
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        kwargs.setdefault("n_jobs", 1)
        super().__init__(*args, **kwargs)
        self._r_forest_tempdir = None
        self._r_forest_model_path = None
        self._r_forest_feature_cols = None

    def _cleanup_r_forest(self):
        if self._r_forest_tempdir is not None:
            try:
                self._r_forest_tempdir.cleanup()
            finally:
                self._r_forest_tempdir = None
                self._r_forest_model_path = None
                self._r_forest_feature_cols = None

    def __del__(self):
        self._cleanup_r_forest()

    def fit_components(self, X, A, time, event, Z, W):
        if self._prediction_nuisance_mode != "full_refit":
            raise ValueError(
                "FinalModelRCSFCensoredSurvivalForest currently supports prediction_nuisance_mode='full_refit' only."
            )

        x = _ensure_2d(X).astype(float)
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        x, raw_w, raw_z, w_nuis, z_nuis, x_final, y_res, a_res = _crossfit_oldc3_survival_ablation_arrays(
            self,
            x,
            A,
            y_packed,
            W,
            Z,
        )

        self._cleanup_r_forest()
        self._r_forest_tempdir, self._r_forest_model_path, self._r_forest_feature_cols = _train_r_csf_final_forest(
            x_final,
            y_res,
            a_res,
            num_trees=self._n_estimators,
            min_node_size=self._min_samples_leaf,
            seed=self._random_state,
        )

        self._train_x = np.asarray(x, dtype=float).copy()
        self._train_w = np.asarray(raw_w, dtype=float).copy()
        self._train_z = np.asarray(raw_z, dtype=float).copy()
        self._train_x_final = np.asarray(x_final, dtype=float).copy()

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
        if self._feature_nuisance is None or self._r_forest_model_path is None or self._r_forest_feature_cols is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")

        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._feature_nuisance.predict_bridge_outputs(
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        x_final = _build_oldc3_survival_ablation_features(
            x,
            raw_w,
            raw_z,
            bridge,
            include_raw_proxy=self._include_raw_proxy,
            surv_scalar_mode=self._surv_scalar_mode,
        )
        preds = _predict_r_csf_final_forest(
            self._r_forest_model_path,
            x_final,
            feature_cols=self._r_forest_feature_cols,
        )
        return np.asarray(preds, dtype=float).reshape(-1, 1)


class FinalModelCSFFinalCensoredSurvivalForest(_BaseSinglePassBridgeFeatureCensoredSurvivalForest):
    """Censored finalized-model family with an R grf::causal_survival_forest final learner."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("surv_scalar_mode", "pair")
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("target", "RMST")
        kwargs.setdefault("horizon", None)
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 600)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("q_clip", 0.03)
        kwargs.setdefault("y_tilde_clip_quantile", 0.98)
        kwargs.setdefault("y_res_clip_percentiles", (2.0, 98.0))
        kwargs.setdefault("n_estimators", 200)
        kwargs.setdefault("min_samples_leaf", 20)
        kwargs.setdefault("censoring_estimator", "nelson-aalen")
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        kwargs.setdefault("n_jobs", 1)
        super().__init__(*args, **kwargs)
        self._r_forest_tempdir = None
        self._r_forest_model_path = None
        self._r_forest_feature_cols = None

    def _cleanup_r_forest(self):
        if self._r_forest_tempdir is not None:
            try:
                self._r_forest_tempdir.cleanup()
            finally:
                self._r_forest_tempdir = None
                self._r_forest_model_path = None
                self._r_forest_feature_cols = None

    def __del__(self):
        self._cleanup_r_forest()

    def _validated_horizon(self) -> float:
        if self._horizon is None:
            raise ValueError(
                "FinalModelCSFFinalCensoredSurvivalForest requires a finite horizon. "
                "Pass horizon=<float> for finite-horizon RMST or survival.probability."
            )
        return float(self._horizon)

    def fit_components(self, X, A, time, event, Z, W):
        if self._prediction_nuisance_mode != "full_refit":
            raise ValueError(
                "FinalModelCSFFinalCensoredSurvivalForest currently supports prediction_nuisance_mode='full_refit' only."
            )
        if self._target not in {"RMST", "survival.probability"}:
            raise ValueError("target must be one of {'RMST', 'survival.probability'}.")
        horizon = self._validated_horizon()

        x = _ensure_2d(X).astype(float)
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        x, raw_w, raw_z, w_nuis, z_nuis, x_final, _, a_res = _crossfit_oldc3_survival_ablation_arrays(
            self,
            x,
            A,
            y_packed,
            W,
            Z,
        )
        a = np.asarray(A, dtype=float).ravel()
        time_obs = np.asarray(time, dtype=float).ravel()
        event_obs = np.asarray(event, dtype=float).ravel()

        # Keep the finalized nuisance/feature pipeline fixed and swap only the
        # final learner to an R causal_survival_forest.
        w_hat_oof = a - np.asarray(a_res, dtype=float).ravel()

        self._cleanup_r_forest()
        self._r_forest_tempdir, self._r_forest_model_path, self._r_forest_feature_cols = _train_r_csf_survival_final_forest(
            x_final,
            time_obs,
            event_obs,
            a,
            w_hat_oof,
            target=self._target,
            horizon=horizon,
            num_trees=self._n_estimators,
            min_node_size=self._min_samples_leaf,
            seed=self._random_state,
        )

        self._train_x = np.asarray(x, dtype=float).copy()
        self._train_w = np.asarray(raw_w, dtype=float).copy()
        self._train_z = np.asarray(raw_z, dtype=float).copy()
        self._train_x_final = np.asarray(x_final, dtype=float).copy()

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
        if self._feature_nuisance is None or self._r_forest_model_path is None or self._r_forest_feature_cols is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")

        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._feature_nuisance.predict_bridge_outputs(
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        x_final = _build_oldc3_survival_ablation_features(
            x,
            raw_w,
            raw_z,
            bridge,
            include_raw_proxy=self._include_raw_proxy,
            surv_scalar_mode=self._surv_scalar_mode,
        )
        preds = _predict_r_csf_survival_final_forest(
            self._r_forest_model_path,
            x_final,
            feature_cols=self._r_forest_feature_cols,
        )
        return np.asarray(preds, dtype=float).reshape(-1, 1)


class FinalModelNoPCICensoredSurvivalForest(_BaseSinglePassBridgeFeatureCensoredSurvivalForest):
    """Finalized censored model with proxy information removed from nuisance fitting."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("surv_scalar_mode", "full")
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("observed_only", True)
        kwargs.setdefault("target", "RMST")
        kwargs.setdefault("horizon", None)
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 600)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("q_clip", 0.03)
        kwargs.setdefault("y_tilde_clip_quantile", 0.98)
        kwargs.setdefault("y_res_clip_percentiles", (2.0, 98.0))
        kwargs.setdefault("n_estimators", 200)
        kwargs.setdefault("min_samples_leaf", 20)
        kwargs.setdefault("censoring_estimator", "nelson-aalen")
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        kwargs.setdefault("n_jobs", 1)
        super().__init__(*args, **kwargs)


class MatchedNoPCICensoredSurvivalForest(FinalModelNoPCICensoredSurvivalForest):
    """Matched censored baseline: keep the finalized censored backbone, remove PCI from nuisance fitting."""


class ProperNoPCICensoredSurvivalForest(_BaseSinglePassBridgeFeatureCensoredSurvivalForest):
    """Proper censored baseline: explicit residuals + CausalForest, but no PCI, no broad-dup, no clipping."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("surv_scalar_mode", "raw_surv")
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("observed_only", True)
        kwargs.setdefault("target", "RMST")
        kwargs.setdefault("horizon", None)
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 600)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("q_clip", 0.0)
        kwargs.setdefault("y_tilde_clip_quantile", None)
        kwargs.setdefault("y_res_clip_percentiles", (0.0, 100.0))
        kwargs.setdefault("n_estimators", 200)
        kwargs.setdefault("min_samples_leaf", 20)
        kwargs.setdefault("censoring_estimator", "nelson-aalen")
        kwargs.setdefault("event_survival_estimator", "cox")
        kwargs.setdefault("m_pred_mode", "survival")
        kwargs.setdefault("nuisance_feature_mode", "dup")
        kwargs.setdefault("n_jobs", 1)
        super().__init__(*args, **kwargs)


class FinalModelRawCensoredSurvivalForest(_BaseSinglePassBridgeFeatureCensoredSurvivalForest):
    """Finalized censored model with raw final-stage features only."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", False)
        kwargs.setdefault("surv_scalar_mode", "raw")
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("target", "RMST")
        kwargs.setdefault("horizon", None)
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 600)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("q_clip", 0.03)
        kwargs.setdefault("y_tilde_clip_quantile", 0.98)
        kwargs.setdefault("y_res_clip_percentiles", (2.0, 98.0))
        kwargs.setdefault("n_estimators", 200)
        kwargs.setdefault("min_samples_leaf", 20)
        kwargs.setdefault("censoring_estimator", "nelson-aalen")
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        kwargs.setdefault("n_jobs", 1)
        super().__init__(*args, **kwargs)


class BroadAugDMLCensoredSurvivalForest(OldC3AugmentedSurvDMLCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class BroadAugDMLObservedCensoredSurvivalForest(OldC3AugmentedSurvObservedDMLCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class BroadAugSPCensoredSurvivalForest(_BaseSinglePassBridgeFeatureCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class BroadAugSPObservedCensoredSurvivalForest(_BaseSinglePassBridgeFeatureCensoredSurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_surv_scalar", True)
        kwargs.setdefault("observed_only", True)
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class UnifiedBroadAugSPBaselineCensoredSurvivalForest(B2SummaryBaselineDMLCensoredSurvivalForest):
    """Matched baseline-family BroadAugSP alias."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("surv_scalar_mode", "full")
        kwargs.setdefault("censoring_estimator", "nelson-aalen")
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class UnifiedBroadAugSPMildShrinkCensoredSurvivalForest(BroadAugSPCensoredSurvivalForest):
    """Matched mild-shrink BroadAugSP family."""
