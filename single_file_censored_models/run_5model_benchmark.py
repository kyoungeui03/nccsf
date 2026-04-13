#!/usr/bin/env python3
"""Run the 5-model censored benchmark from the single-file model folder.

Models:
    1. Final Model
    2. Final Model Oracle
    3. Strict Baseline
    4. Strict Baseline Oracle
    5. R-CSF Baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from econml._ortho_learner import _OrthoLearner
from econml.dml import CausalForestDML

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_FILE = Path(__file__).resolve()
MODEL_DIR = THIS_FILE.parent
PROJECT_ROOT = MODEL_DIR.parent
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from grf.benchmarks.econml_8variant import (  # noqa: E402
    CASE_SPECS,
    _build_true_y_tilde,
    _compute_true_target_ipcw_3term_y_res,
    _evaluate_predictions,
    _true_survival_components,
    prepare_case,
    true_event_surv_on_grid,
    true_outcome_oracle,
    true_propensity_oracle,
)

try:  # pragma: no cover
    from .final_censored_model import (  # noqa: E402
        FinalModelCensoredSurvivalForest,
        _SinglePassBridgeFeatureCensoredSurvivalForest,
        _build_final_features_full,
        _compute_q_from_s,
        _compute_survival_probability_q_from_s,
        _ensure_2d,
        _prepare_target_inputs,
    )
    from .grf_censored_baseline import GRFCensoredBaseline  # noqa: E402
    from .strict_censored_baseline import StrictEconmlXWZCensoredSurvivalForest  # noqa: E402
except ImportError:  # pragma: no cover
    from single_file_censored_models.final_censored_model import (  # type: ignore  # noqa: E402
        FinalModelCensoredSurvivalForest,
        _SinglePassBridgeFeatureCensoredSurvivalForest,
        _build_final_features_full,
        _compute_q_from_s,
        _compute_survival_probability_q_from_s,
        _ensure_2d,
        _prepare_target_inputs,
    )
    from single_file_censored_models.grf_censored_baseline import GRFCensoredBaseline  # type: ignore  # noqa: E402
    from single_file_censored_models.strict_censored_baseline import StrictEconmlXWZCensoredSurvivalForest  # type: ignore  # noqa: E402


TARGETS = ["RMST", "survival.probability"]
MODEL_COUNT = 5


class StrictOracleCensoredSurvivalForest(StrictEconmlXWZCensoredSurvivalForest):
    """Strict baseline with oracle IPCW pseudo-outcomes from the synthetic DGP."""

    def __init__(self, *, cfg, dgp, **kwargs):
        self._cfg = cfg
        self._dgp = dgp
        super().__init__(**kwargs)

    def fit_oracle(self, X, A, time, event, Z, W, U):
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_full = self.stack_final_features(x, raw_w, raw_z)
        y_time = np.asarray(time, dtype=float).ravel()
        delta = np.asarray(event, dtype=float).ravel()
        a = np.asarray(A, dtype=float).ravel()
        u = np.asarray(U, dtype=float).ravel()

        y_tilde = _build_true_y_tilde(
            x,
            u,
            y_time,
            delta,
            self._cfg,
            self._dgp,
            target=self._target,
            horizon=self._horizon,
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


class _SingleFileFinalOracleNuisance:
    """Oracle nuisance layer for the single-file censored Final Model benchmark."""

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
        if self._target == "RMST":
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


class SingleFileFinalOracleCensoredSurvivalForest(_SinglePassBridgeFeatureCensoredSurvivalForest):
    """Oracle counterpart of the single-file censored Final Model."""

    def __init__(self, *, cfg, dgp, p_x, **kwargs):
        self._benchmark_cfg = cfg
        self._benchmark_dgp = dgp
        self._benchmark_p_x = int(p_x)
        self._oracle_feature_nuisance = None
        super().__init__(**kwargs)

    def _gen_ortho_learner_model_nuisance(self):
        return _SingleFileFinalOracleNuisance(
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

        self._oracle_feature_nuisance = _SingleFileFinalOracleNuisance(
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


MODEL_BUILDERS = {
    "final": (
        "Final Model",
        lambda target, horizon, random_state: FinalModelCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=random_state,
            surv_scalar_mode="full",
        ),
    ),
    "strict": (
        "Strict Baseline",
        lambda target, horizon, random_state: StrictEconmlXWZCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=random_state,
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 5-model censored benchmark from the single_file_censored_models folder."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int, default=None)
    parser.add_argument("--case-slugs", nargs="*", default=None)
    parser.add_argument("--target", choices=["RMST", "survival.probability", "both"], default="both")
    parser.add_argument("--horizon-quantile", type=float, default=0.60)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-trees-baseline", type=int, default=200)
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / "single_file_censored_models_5model").resolve()


def _selected_case_specs(case_ids: list[int] | None, case_slugs: list[str] | None) -> list[dict[str, object]]:
    if not case_ids and not case_slugs:
        return list(CASE_SPECS)
    id_set = set(case_ids or [])
    slug_set = set(case_slugs or [])
    selected = [spec for spec in CASE_SPECS if int(spec["case_id"]) in id_set or str(spec["slug"]) in slug_set]
    if not selected:
        raise ValueError("No synthetic cases matched the requested ids/slugs.")
    return selected


def _resolve_targets(target_arg: str) -> list[str]:
    if target_arg == "both":
        return list(TARGETS)
    return [target_arg]


def _normalize_metrics_row(row: dict[str, object]) -> dict[str, object]:
    normalized = dict(row)
    normalized["sign_precision"] = float(normalized.pop("sign_acc"))
    normalized["pearson_correlation"] = float(normalized.pop("pearson"))
    normalized.pop("pehe", None)
    return normalized


def summarize_results(combined_df: pd.DataFrame):
    summary = (
        combined_df.groupby("name", as_index=False)
        .agg(
            avg_pred_cate=("mean_pred", "mean"),
            avg_true_cate=("mean_true_cate", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson_correlation=("pearson_correlation", "mean"),
            avg_bias=("bias", "mean"),
            avg_abs_bias=("bias", lambda values: float(np.mean(np.abs(values)))),
            avg_sign_precision=("sign_precision", "mean"),
            avg_total_time=("total_time", "mean"),
        )
        .sort_values(
            ["avg_rmse", "avg_mae", "avg_abs_bias", "avg_pearson_correlation", "avg_sign_precision"],
            ascending=[True, True, True, False, False],
        )
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary = summary.drop(columns=["avg_abs_bias"])
    top5 = summary.head(5).copy()
    return summary, top5


def render_case_table_png(case_df: pd.DataFrame, output_path: Path):
    display = case_df.loc[
        :,
        ["name", "mean_pred", "mean_true_cate", "bias", "rmse", "mae", "pearson_correlation", "sign_precision", "total_time"],
    ].copy()
    display.columns = [
        "Variant",
        "Pred CATE",
        "True CATE",
        "Bias",
        "RMSE",
        "MAE",
        "Pearson Corr",
        "Sign Precision",
        "Time",
    ]
    for col in ["Pred CATE", "True CATE", "Bias", "RMSE", "MAE", "Pearson Corr"]:
        display[col] = display[col].map(lambda value: f"{value:.4f}")
    display["Sign Precision"] = display["Sign Precision"].map(lambda value: f"{100.0 * value:.1f}%")
    display["Time"] = display["Time"].map(lambda value: f"{value:.1f}s")

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.axis("off")
    title = case_df["case_title"].iloc[0]
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)
    ax.set_title("5-model benchmark", fontsize=13, color="#4b5563", pad=18)

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0.02, 1, 0.86],
        colWidths=[0.34, 0.09, 0.09, 0.08, 0.08, 0.08, 0.1, 0.09, 0.07],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.55)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("#d1d5db")
            cell.set_linewidth(1.5)
        else:
            cell.set_facecolor("#ffffff" if row % 2 == 1 else "#f3f4f6")
            cell.set_edgecolor("#d1d5db")
            cell.set_linewidth(0.8)
            if col == 0:
                cell.set_text_props(ha="left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_avg_summary_png(summary_df: pd.DataFrame, output_path: Path):
    display = summary_df.copy()
    display["Avg RMSE"] = display["avg_rmse"].map(lambda value: f"{value:.4f}")
    display["Avg MAE"] = display["avg_mae"].map(lambda value: f"{value:.4f}")
    display["Avg Pearson Corr"] = display["avg_pearson_correlation"].map(lambda value: f"{value:.4f}")
    display["Avg Bias"] = display["avg_bias"].map(lambda value: f"{value:+.4f}")
    display["Avg Sign Precision"] = display["avg_sign_precision"].map(lambda value: f"{100.0 * value:.1f}%")
    display["Avg Time"] = display["avg_total_time"].map(lambda value: f"{value:.1f}s")
    display = display.loc[:, ["rank", "name", "Avg RMSE", "Avg MAE", "Avg Pearson Corr", "Avg Bias", "Avg Sign Precision", "Avg Time"]]
    display.columns = ["Rank", "Variant", "Avg RMSE", "Avg MAE", "Avg Pearson Corr", "Avg Bias", "Avg Sign Precision", "Avg Time"]

    fig, ax = plt.subplots(figsize=(22, 8))
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.03, 0.98, 0.94],
        colWidths=[0.06, 0.33, 0.11, 0.11, 0.14, 0.09, 0.12, 0.09],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.7)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2b3440")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor("#19212c")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#0b1220")
            cell.set_text_props(color="white")
            if col == 1:
                cell.set_text_props(ha="left", color="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def render_top5_png(top5_df: pd.DataFrame, output_path: Path):
    display = top5_df.copy()
    display["Avg RMSE"] = display["avg_rmse"].map(lambda value: f"{value:.4f}")
    display["Avg MAE"] = display["avg_mae"].map(lambda value: f"{value:.4f}")
    display["Avg Pearson Corr"] = display["avg_pearson_correlation"].map(lambda value: f"{value:.4f}")
    display["Avg Bias"] = display["avg_bias"].map(lambda value: f"{value:+.4f}")
    display["Avg Sign Precision"] = display["avg_sign_precision"].map(lambda value: f"{100.0 * value:.1f}%")
    display["Avg Time"] = display["avg_total_time"].map(lambda value: f"{value:.1f}s")
    display = display.loc[:, ["rank", "name", "Avg RMSE", "Avg MAE", "Avg Pearson Corr", "Avg Bias", "Avg Sign Precision", "Avg Time"]]
    display.columns = ["Rank", "Variant", "Avg RMSE", "Avg MAE", "Avg Pearson Corr", "Avg Bias", "Avg Sign Precision", "Avg Time"]

    fig, ax = plt.subplots(figsize=(20, 6))
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.05, 0.96, 0.9],
        colWidths=[0.08, 0.34, 0.11, 0.11, 0.14, 0.09, 0.13, 0.1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.7)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2b3440")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor("#19212c")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#0b1220")
            cell.set_text_props(color="white")
            if col == 1:
                cell.set_text_props(ha="left", color="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _format_progress_line(step_idx: int, total_steps: int, row: dict[str, object], *, case_id: int, case_slug: str, target: str) -> str:
    return (
        f"[{step_idx}/{total_steps}] case={case_id:02d} slug={case_slug} "
        f"target={target} model={row['name']} "
        f"rmse={float(row['rmse']):.4f} mae={float(row['mae']):.4f} "
        f"corr={float(row['pearson_correlation']):.4f} bias={float(row['bias']):+.4f} "
        f"sign_precision={100.0 * float(row['sign_precision']):.1f}% "
        f"time={float(row['total_time']):.1f}s"
    )


def _evaluate_case_target(
    case,
    *,
    target: str,
    random_state: int,
    num_trees_baseline: int,
    progress_hook=None,
) -> list[dict[str, object]]:
    x = np.asarray(case.X, dtype=float)
    w = np.asarray(case.W, dtype=float)
    z = np.asarray(case.Z, dtype=float)
    a = np.asarray(case.A, dtype=float)
    time_obs = np.asarray(case.Y, dtype=float)
    event = np.asarray(case.delta, dtype=float)
    horizon = float(case.horizon)

    rows: list[dict[str, object]] = []

    final_name, final_builder = MODEL_BUILDERS["final"]
    final_model = final_builder(target, horizon, random_state)
    t0 = time.time()
    final_model.fit_components(x, a, time_obs, event, z, w)
    final_preds = np.asarray(final_model.effect_from_components(x, w, z), dtype=float).ravel()
    rows.append(
        _evaluate_predictions(
            final_name,
            final_preds,
            case.true_cate,
            time.time() - t0,
            backend=final_model.__class__.__name__,
        )
    )

    u = _ensure_2d(case.U)
    x_oracle = np.column_stack([x, u])
    oracle_model = SingleFileFinalOracleCensoredSurvivalForest(
        cfg=case.cfg,
        dgp=case.dgp,
        p_x=x.shape[1],
        target=target,
        horizon=horizon,
        random_state=random_state,
        q_kind="logit",
        h_kind="extra",
        h_n_estimators=600,
        h_min_samples_leaf=5,
        q_clip=0.03,
        y_tilde_clip_quantile=0.98,
        y_res_clip_percentiles=(2.0, 98.0),
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        censoring_estimator="nelson-aalen",
        event_survival_estimator="cox",
        m_pred_mode="bridge",
        nuisance_feature_mode="broad_dup",
    )
    t0 = time.time()
    oracle_model.fit_oracle(x_oracle, a, time_obs, event, u)
    oracle_preds = np.asarray(oracle_model.effect_oracle(x, u), dtype=float).ravel()
    rows.append(
        _evaluate_predictions(
            "Final Model Oracle",
            oracle_preds,
            case.true_cate,
            time.time() - t0,
            backend=oracle_model.__class__.__name__,
        )
    )

    strict_name, strict_builder = MODEL_BUILDERS["strict"]
    strict_model = strict_builder(target, horizon, random_state)
    t0 = time.time()
    strict_model.fit_components(x, a, time_obs, event, z, w)
    strict_preds = np.asarray(strict_model.effect_from_components(x, w, z), dtype=float).ravel()
    rows.append(
        _evaluate_predictions(
            strict_name,
            strict_preds,
            case.true_cate,
            time.time() - t0,
            backend=strict_model.__class__.__name__,
        )
    )

    strict_oracle_model = StrictOracleCensoredSurvivalForest(
        cfg=case.cfg,
        dgp=case.dgp,
        target=target,
        horizon=horizon,
        random_state=random_state,
    )
    t0 = time.time()
    strict_oracle_model.fit_oracle(x, a, time_obs, event, z, w, case.U)
    strict_oracle_preds = np.asarray(strict_oracle_model.effect_from_components(x, w, z), dtype=float).ravel()
    rows.append(
        _evaluate_predictions(
            "Strict Baseline Oracle",
            strict_oracle_preds,
            case.true_cate,
            time.time() - t0,
            backend=strict_oracle_model.__class__.__name__,
        )
    )

    r_model = GRFCensoredBaseline(
        target=target,
        horizon=horizon,
        n_estimators=num_trees_baseline,
        random_state=random_state,
    )
    try:
        t0 = time.time()
        r_model.fit_components(x, a, time_obs, event, z, w)
        r_preds = np.asarray(r_model.effect_from_components(x, w, z), dtype=float).ravel()
        rows.append(
            _evaluate_predictions(
                "R-CSF Baseline",
                r_preds,
                case.true_cate,
                time.time() - t0,
                backend=r_model.__class__.__name__,
            )
        )
    finally:
        r_model.cleanup()

    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        normalized = _normalize_metrics_row(row)
        normalized["target"] = target
        normalized_rows.append(normalized)
        if progress_hook is not None:
            progress_hook(normalized)
    return normalized_rows


def _write_target_summaries(results: pd.DataFrame, *, output_dir: Path, file_prefix: str) -> None:
    for target in TARGETS:
        target_df = results.loc[results["target"] == target].copy()
        if target_df.empty:
            continue
        summary_df, top5_df = summarize_results(target_df)
        slug = target.replace(".", "_")
        summary_df.to_csv(output_dir / f"{file_prefix}_{slug}_summary.csv", index=False)
        top5_df.to_csv(output_dir / f"{file_prefix}_{slug}_top5.csv", index=False)
        render_avg_summary_png(summary_df, output_dir / f"{file_prefix}_{slug}_summary.png")
        render_top5_png(top5_df, output_dir / f"{file_prefix}_{slug}_top5.png")


def main() -> int:
    args = parse_args()
    if args.list_cases:
        for spec in CASE_SPECS:
            print(f"{int(spec['case_id']):2d}  {spec['slug']}")
        return 0

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_specs = _selected_case_specs(args.case_ids, args.case_slugs)
    targets = _resolve_targets(args.target)
    total_steps = len(case_specs) * len(targets) * MODEL_COUNT
    completed_steps = 0

    frames: list[pd.DataFrame] = []
    for case_spec in case_specs:
        for target in targets:
            case = prepare_case(case_spec, target=target, horizon_quantile=args.horizon_quantile)

            def progress_hook(row, *, _case_id=int(case_spec["case_id"]), _case_slug=str(case_spec["slug"]), _target=target):
                nonlocal completed_steps
                completed_steps += 1
                print(
                    _format_progress_line(
                        completed_steps,
                        total_steps,
                        row,
                        case_id=_case_id,
                        case_slug=_case_slug,
                        target=_target,
                    ),
                    flush=True,
                )

            rows = _evaluate_case_target(
                case,
                target=target,
                random_state=args.random_state,
                num_trees_baseline=args.num_trees_baseline,
                progress_hook=progress_hook,
            )
            case_df = pd.DataFrame(rows)
            case_df["case_id"] = int(case_spec["case_id"])
            case_df["case_slug"] = str(case_spec["slug"])
            case_df["case_title"] = str(case_spec["title"])
            case_df["horizon"] = float(case.horizon)
            frames.append(case_df)

            target_slug = target.replace(".", "_")
            case_base = f"case_{int(case_spec['case_id']):02d}_{case_spec['slug']}_{target_slug}"
            case_df.to_csv(output_dir / f"{case_base}.csv", index=False)
            render_case_table_png(case_df, output_dir / f"{case_base}.png")

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(output_dir / "results_full.csv", index=False)
    _write_target_summaries(results, output_dir=output_dir, file_prefix="basic12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
