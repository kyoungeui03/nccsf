#!/usr/bin/env python3
"""Run selected censored benchmark models from the canonical censored folder.

Default models:
    1. Final Conditional
    2. Final Conditional Oracle
    3. Censored Baseline

Additional optional models include Naive Censored Baseline, R-CSF Baseline,
and the appendix final-feature ablation variants.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

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

from grf.benchmarks.econml_8variant import _evaluate_predictions, prepare_case  # noqa: E402

try:  # pragma: no cover
    from .final_censored_model import (  # noqa: E402
        _ensure_2d,
        FinalModelConditionalCensoredSurvivalForest,
        FinalModelConditionalPCIXOnlyCensoredSurvivalForest,
    )
    from .final_censored_oracle import FinalCensoredOracle  # noqa: E402
    from .gpu_backends import resolve_xgboost_gpu_mode  # noqa: E402
    from .censored_baseline import CensoredBaselineCensoredSurvivalForest  # noqa: E402
    from .grf_censored_baseline import GRFCensoredBaseline  # noqa: E402
    from .naive_censored_baseline import NaiveEconmlXWZCensoredSurvivalForest  # noqa: E402
except ImportError:  # pragma: no cover
    from censored.final_censored_model import (  # type: ignore  # noqa: E402
        _ensure_2d,
        FinalModelConditionalCensoredSurvivalForest,
        FinalModelConditionalPCIXOnlyCensoredSurvivalForest,
    )
    from censored.final_censored_oracle import FinalCensoredOracle  # type: ignore  # noqa: E402
    from censored.gpu_backends import resolve_xgboost_gpu_mode  # type: ignore  # noqa: E402
    from censored.censored_baseline import (  # type: ignore  # noqa: E402
        CensoredBaselineCensoredSurvivalForest,
    )
    from censored.grf_censored_baseline import GRFCensoredBaseline  # type: ignore  # noqa: E402
    from censored.naive_censored_baseline import (  # type: ignore  # noqa: E402
        NaiveEconmlXWZCensoredSurvivalForest,
    )


TARGETS = ["RMST", "survival.probability"]
DEFAULT_MODELS = ("final_conditional", "final_conditional_oracle", "censored_baseline")
CHECKPOINT_RESULTS_FILE = "results_incremental.csv"
RUN_METADATA_FILE = "run_metadata.json"

CASE_SPECS = [
    {
        "case_id": 1,
        "slug": "linear_linear_informative_strong_beneficial_large",
        "title": "Linear treatment / linear outcome DGP, informative proxies, strong confounding, beneficial treatment effect",
        "cfg": dict(
            beta_u_in_t=0.8,
            gamma_u_in_a=0.8,
            sigma_z=1.125,
            sigma_w=1.53,
            tau_log_hr=-0.6,
            linear_treatment=True,
            linear_outcome=True,
        ),
    },
    {
        "case_id": 2,
        "slug": "linear_linear_informative_weak_harmful_small",
        "title": "Linear treatment / linear outcome DGP, informative proxies, weak confounding, small beneficial treatment effect",
        "cfg": dict(
            beta_u_in_t=0.3,
            gamma_u_in_a=0.2,
            sigma_z=1.125,
            sigma_w=1.53,
            tau_log_hr=-0.25,
            linear_treatment=True,
            linear_outcome=True,
        ),
    },
    {
        "case_id": 3,
        "slug": "linear_linear_weakproxy_strong_harmful_moderate",
        "title": "Linear treatment / linear outcome DGP, weak proxies, strong confounding, beneficial treatment effect",
        "cfg": dict(
            beta_u_in_t=0.9,
            gamma_u_in_a=0.9,
            sigma_z=7.35,
            sigma_w=4.77,
            tau_log_hr=-0.5,
            linear_treatment=True,
            linear_outcome=True,
        ),
    },
    {
        "case_id": 4,
        "slug": "linear_linear_weakproxy_weak_nearnull_harmful",
        "title": "Linear treatment / linear outcome DGP, weak proxies, weak confounding, near-null beneficial treatment effect",
        "cfg": dict(
            beta_u_in_t=0.2,
            gamma_u_in_a=0.3,
            sigma_z=7.35,
            sigma_w=4.77,
            tau_log_hr=-0.12,
            linear_treatment=True,
            linear_outcome=True,
        ),
    },
    {
        "case_id": 5,
        "slug": "linear_nonlinear_informative_strong_beneficial_large",
        "title": "Linear treatment / nonlinear outcome DGP, informative proxies, strong confounding, beneficial treatment effect",
        "cfg": dict(
            beta_u_in_t=0.8,
            gamma_u_in_a=0.9,
            sigma_z=1.125,
            sigma_w=1.53,
            tau_log_hr=-0.7,
            linear_treatment=True,
            linear_outcome=False,
        ),
    },
    {
        "case_id": 6,
        "slug": "linear_nonlinear_informative_weak_harmful_small",
        "title": "Linear treatment / nonlinear outcome DGP, informative proxies, weak confounding, small beneficial treatment effect",
        "cfg": dict(
            beta_u_in_t=0.2,
            gamma_u_in_a=0.2,
            sigma_z=1.125,
            sigma_w=1.53,
            tau_log_hr=-0.3,
            linear_treatment=True,
            linear_outcome=False,
        ),
    },
    {
        "case_id": 7,
        "slug": "linear_nonlinear_weakproxy_strong_harmful_moderate",
        "title": "Linear treatment / nonlinear outcome DGP, weak proxies, strong confounding, beneficial treatment effect",
        "cfg": dict(
            beta_u_in_t=1.1,
            gamma_u_in_a=0.9,
            sigma_z=7.35,
            sigma_w=4.77,
            tau_log_hr=-0.5,
            linear_treatment=True,
            linear_outcome=False,
        ),
    },
    {
        "case_id": 8,
        "slug": "linear_nonlinear_weakproxy_weak_nearnull_harmful",
        "title": "Linear treatment / nonlinear outcome DGP, weak proxies, weak confounding, near-null beneficial treatment effect",
        "cfg": dict(
            beta_u_in_t=0.15,
            gamma_u_in_a=0.25,
            sigma_z=7.35,
            sigma_w=4.77,
            tau_log_hr=-0.12,
            linear_treatment=True,
            linear_outcome=False,
        ),
    },
    {
        "case_id": 9,
        "slug": "nonlinear_nonlinear_informative_strong_beneficial_large",
        "title": "Nonlinear treatment / nonlinear outcome DGP, informative proxies, strong confounding, beneficial treatment effect",
        "cfg": dict(
            beta_u_in_t=1.0,
            gamma_u_in_a=1.0,
            sigma_z=1.125,
            sigma_w=1.53,
            tau_log_hr=-0.7,
            linear_treatment=False,
            linear_outcome=False,
        ),
    },
    {
        "case_id": 10,
        "slug": "nonlinear_nonlinear_informative_weak_harmful_small",
        "title": "Nonlinear treatment / nonlinear outcome DGP, informative proxies, weak confounding, small beneficial treatment effect",
        "cfg": dict(
            beta_u_in_t=0.2,
            gamma_u_in_a=0.2,
            sigma_z=1.125,
            sigma_w=1.53,
            tau_log_hr=-0.3,
            linear_treatment=False,
            linear_outcome=False,
        ),
    },
    {
        "case_id": 11,
        "slug": "nonlinear_nonlinear_weakproxy_strong_harmful_moderate",
        "title": "Nonlinear treatment / nonlinear outcome DGP, weak proxies, strong confounding, beneficial treatment effect",
        "cfg": dict(
            beta_u_in_t=1.1,
            gamma_u_in_a=0.9,
            sigma_z=7.35,
            sigma_w=4.77,
            tau_log_hr=-0.5,
            linear_treatment=False,
            linear_outcome=False,
        ),
    },
    {
        "case_id": 12,
        "slug": "nonlinear_nonlinear_weakproxy_weak_nearnull_harmful",
        "title": "Nonlinear treatment / nonlinear outcome DGP, weak proxies, weak confounding, near-null beneficial treatment effect",
        "cfg": dict(
            beta_u_in_t=0.2,
            gamma_u_in_a=0.3,
            sigma_z=7.35,
            sigma_w=4.77,
            tau_log_hr=-0.12,
            linear_treatment=False,
            linear_outcome=False,
        ),
    },
]

SETTINGS = [
    {"setting_id": "S01", "n": 500, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S02", "n": 1000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S03", "n": 2000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S04", "n": 4000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S05", "n": 8000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S06", "n": 16000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S07", "n": 2000, "p_x": 2, "p_w": 1, "p_z": 1},
    {"setting_id": "S08", "n": 2000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S09", "n": 2000, "p_x": 10, "p_w": 1, "p_z": 1},
    {"setting_id": "S10", "n": 2000, "p_x": 20, "p_w": 1, "p_z": 1},
    {"setting_id": "S11", "n": 2000, "p_x": 40, "p_w": 1, "p_z": 1},
    {"setting_id": "S12", "n": 2000, "p_x": 5, "p_w": 2, "p_z": 2},
    {"setting_id": "S13", "n": 2000, "p_x": 5, "p_w": 3, "p_z": 3},
    {"setting_id": "S14", "n": 2000, "p_x": 5, "p_w": 5, "p_z": 5},
    {"setting_id": "S15", "n": 2000, "p_x": 5, "p_w": 10, "p_z": 10},
    {"setting_id": "S16", "n": 2000, "p_x": 5, "p_w": 20, "p_z": 20},
    {"setting_id": "S17", "n": 2000, "p_x": 5, "p_w": 1, "p_z": 10},
    {"setting_id": "S18", "n": 2000, "p_x": 5, "p_w": 10, "p_z": 1},
    {"setting_id": "S19", "n": 2000, "p_x": 10, "p_w": 5, "p_z": 5},
    {"setting_id": "S20", "n": 2000, "p_x": 20, "p_w": 5, "p_z": 5},
    {"setting_id": "S21", "n": 2000, "p_x": 40, "p_w": 10, "p_z": 10},
    {"setting_id": "S22", "n": 4000, "p_x": 20, "p_w": 10, "p_z": 10},
]


MODEL_NAMES = {
    "final_conditional": "Final Conditional",
    "final_conditional_oracle": "Final Conditional Oracle",
    "censored_baseline": "Censored Baseline",
    "naive_censored_baseline": "Naive Censored Baseline",
    "r_csf_baseline": "R-CSF Baseline",
    "final_conditional_x_only": "Final Conditional PCI X-Only",
    "final_conditional_broaddup_raw_only": "Final Conditional BroadDup Raw-Only",
    "final_conditional_dup_full": "Final Conditional Dup Full",
    "final_conditional_dup_raw_only": "Final Conditional Dup Raw-Only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected censored benchmark models from the censored folder."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int, default=None)
    parser.add_argument("--case-slugs", nargs="*", default=None)
    parser.add_argument("--setting-ids", nargs="*", default=None)
    parser.add_argument("--target", choices=["RMST", "survival.probability", "both"], default="both")
    parser.add_argument(
        "--rmst-horizon",
        type=float,
        default=3.0,
        help="Fixed RMST horizon used for every RMST benchmark cell.",
    )
    parser.add_argument(
        "--horizon-quantile",
        type=float,
        default=0.90,
        help="Survival-probability horizon quantile; q90 by default.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-trees", "--num-trees-baseline", dest="num_trees", type=int, default=200)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(MODEL_NAMES),
        default=list(DEFAULT_MODELS),
        help="Models to run. Defaults to Final Conditional, Oracle, and Censored Baseline.",
    )
    parser.add_argument("--grf-num-trees", type=int, default=200)
    parser.add_argument("--grf-min-node-size", type=int, default=20)
    parser.add_argument("--gpu", choices=["off", "auto", "xgboost"], default="auto")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-settings", action="store_true")
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / "censored_5model_conditional_suite").resolve()


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


def _selected_settings(setting_ids: list[str] | None) -> list[dict[str, int | str]]:
    if not setting_ids:
        return list(SETTINGS)
    wanted = {str(s).upper() for s in setting_ids}
    selected = [s for s in SETTINGS if str(s["setting_id"]).upper() in wanted]
    if not selected:
        raise ValueError("No settings matched the requested setting ids.")
    return selected


def _case_spec_with_setting(case_spec: dict[str, object], setting: dict[str, int | str]) -> dict[str, object]:
    case_copy = dict(case_spec)
    cfg = dict(case_spec["cfg"])
    cfg.update(
        {
            "n": int(setting["n"]),
            "p_x": int(setting["p_x"]),
            "p_w": int(setting["p_w"]),
            "p_z": int(setting["p_z"]),
        }
    )
    case_copy["cfg"] = cfg
    return case_copy


def _case_base_name(case_id: int, case_slug: str, setting_id: str, target: str) -> str:
    target_slug = target.replace(".", "_")
    return f"case_{int(case_id):02d}_{case_slug}_{setting_id}_{target_slug}"


def _make_run_metadata(
    *,
    args: argparse.Namespace,
    case_specs: list[dict[str, object]],
    settings: list[dict[str, int | str]],
    targets: list[str],
) -> dict[str, object]:
    return {
        "script": THIS_FILE.name,
        "runner_version": 4,
        "case_ids": [int(spec["case_id"]) for spec in case_specs],
        "case_slugs": [str(spec["slug"]) for spec in case_specs],
        "setting_ids": [str(setting["setting_id"]) for setting in settings],
        "targets": list(targets),
        "rmst_horizon": float(args.rmst_horizon),
        "horizon_quantile": float(args.horizon_quantile),
        "random_state": int(args.random_state),
        "num_trees": int(args.num_trees),
        "models": list(args.models),
        "grf_num_trees": int(args.grf_num_trees),
        "grf_min_node_size": int(args.grf_min_node_size),
        "gpu_mode": str(args.gpu),
        "model_count": len(args.models),
    }


def _metadata_path(output_dir: Path) -> Path:
    return output_dir / RUN_METADATA_FILE


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / CHECKPOINT_RESULTS_FILE


def _results_full_path(output_dir: Path) -> Path:
    return output_dir / "results_full.csv"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_checkpoint_results(output_dir: Path) -> pd.DataFrame:
    checkpoint_path = _checkpoint_path(output_dir)
    if checkpoint_path.exists():
        return pd.read_csv(checkpoint_path)

    results_full_path = _results_full_path(output_dir)
    if results_full_path.exists():
        return pd.read_csv(results_full_path)

    return pd.DataFrame()


def _validate_or_initialize_resume_state(
    *,
    output_dir: Path,
    metadata: dict[str, object],
) -> pd.DataFrame:
    meta_path = _metadata_path(output_dir)
    existing_results = _load_checkpoint_results(output_dir)

    if meta_path.exists():
        existing_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing_metadata != metadata:
            raise ValueError(
                "Existing checkpoint metadata does not match the requested run. "
                f"Use a new --output-dir or remove {meta_path} and saved results to restart cleanly."
            )
    else:
        _write_json(meta_path, metadata)

    return existing_results


def _sort_results_frame(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results.copy()
    sort_cols = ["case_id", "setting_id", "target", "name"]
    existing_cols = [col for col in sort_cols if col in results.columns]
    if not existing_cols:
        return results.reset_index(drop=True)
    return results.sort_values(existing_cols).reset_index(drop=True)


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
    ax.set_title("Conditional-censoring 5-model benchmark", fontsize=13, color="#4b5563", pad=18)

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0.02, 1, 0.86],
        colWidths=[0.38, 0.08, 0.08, 0.08, 0.08, 0.08, 0.1, 0.08, 0.06],
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
    case_id: int,
    case_slug: str,
    case_title: str,
    setting: dict[str, int | str],
    target: str,
    random_state: int,
    num_trees: int,
    selected_models: list[str],
    grf_num_trees: int,
    grf_min_node_size: int,
    gpu_config: dict[str, object],
    existing_rows_by_name: dict[str, dict[str, object]] | None = None,
    save_row_hook=None,
    progress_hook=None,
) -> list[dict[str, object]]:
    x = np.asarray(case.X, dtype=float)
    w = np.asarray(case.W, dtype=float)
    z = np.asarray(case.Z, dtype=float)
    a = np.asarray(case.A, dtype=float)
    time_obs = np.asarray(case.Y, dtype=float)
    event = np.asarray(case.delta, dtype=float)
    horizon = float(case.horizon)
    use_gpu = bool(gpu_config.get("use_gpu", False))
    q_kind = "xgb_gpu" if use_gpu else "logit"
    h_kind = "xgb_gpu" if use_gpu else "extra"
    propensity_kind = "xgb_gpu" if use_gpu else "logit"

    rows: list[dict[str, object]] = []

    def _attach_metadata(normalized: dict[str, object]) -> dict[str, object]:
        row = dict(normalized)
        row["target"] = target
        row["case_id"] = int(case_id)
        row["case_slug"] = str(case_slug)
        row["case_title"] = str(case_title)
        row["setting_id"] = str(setting["setting_id"])
        row["n"] = int(setting["n"])
        row["p_x"] = int(setting["p_x"])
        row["p_w"] = int(setting["p_w"])
        row["p_z"] = int(setting["p_z"])
        row["horizon"] = float(case.horizon)
        row["gpu_requested"] = str(gpu_config.get("mode", "off"))
        row["gpu_enabled"] = bool(gpu_config.get("use_gpu", False))
        return row

    existing_rows_by_name = existing_rows_by_name or {}

    def run_or_resume(model_name: str, build_row):
        if model_name in existing_rows_by_name:
            resumed_row = dict(existing_rows_by_name[model_name])
            rows.append(resumed_row)
            if progress_hook is not None:
                progress_hook(resumed_row, resumed=True)
            return

        raw = build_row()
        normalized = _normalize_metrics_row(raw)
        full_row = _attach_metadata(normalized)
        rows.append(full_row)
        if save_row_hook is not None:
            save_row_hook(full_row)
        if progress_hook is not None:
            progress_hook(full_row, resumed=False)

    def run_final_conditional():
        final_model = FinalModelConditionalCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=random_state,
            surv_scalar_mode="full",
            n_estimators=num_trees,
            censoring_estimator="cox",
            q_kind=q_kind,
            h_kind=h_kind,
        )
        t0 = time.time()
        final_model.fit_components(x, a, time_obs, event, z, w)
        final_preds = np.asarray(final_model.effect_from_components(x, w, z), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES["final_conditional"],
            final_preds,
            case.true_cate,
            time.time() - t0,
            backend=final_model.__class__.__name__,
        )

    def run_final_conditional_oracle():
        u = _ensure_2d(case.U)
        x_oracle = np.column_stack([x, u])
        oracle_model = FinalCensoredOracle(
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
            n_estimators=num_trees,
            min_samples_leaf=20,
            cv=5,
            censoring_estimator="cox",
            event_survival_estimator="cox",
            m_pred_mode="bridge",
            nuisance_feature_mode="broad_dup",
        )
        t0 = time.time()
        oracle_model.fit_oracle(x_oracle, a, time_obs, event, u)
        oracle_preds = np.asarray(oracle_model.effect_oracle(x, u), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES["final_conditional_oracle"],
            oracle_preds,
            case.true_cate,
            time.time() - t0,
            backend=oracle_model.__class__.__name__,
        )

    def run_censored_baseline():
        baseline_model = CensoredBaselineCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=random_state,
            n_estimators=num_trees,
            censoring_estimator="cox",
            propensity_kind=propensity_kind,
        )
        t0 = time.time()
        baseline_model.fit_components(x, a, time_obs, event, z, w)
        baseline_preds = np.asarray(baseline_model.effect_from_components(x, w, z), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES["censored_baseline"],
            baseline_preds,
            case.true_cate,
            time.time() - t0,
            backend=baseline_model.__class__.__name__,
        )

    def run_naive_censored_baseline():
        naive_model = NaiveEconmlXWZCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=random_state,
            n_estimators=num_trees,
            min_samples_leaf=20,
            censoring_estimator="cox",
        )
        t0 = time.time()
        naive_model.fit_components(x, a, time_obs, event, z, w)
        naive_preds = np.asarray(naive_model.effect_from_components(x, w, z), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES["naive_censored_baseline"],
            naive_preds,
            case.true_cate,
            time.time() - t0,
            backend=naive_model.__class__.__name__,
        )

    def run_r_csf_baseline():
        rcsf_model = GRFCensoredBaseline(
            target=target,
            horizon=horizon,
            random_state=random_state,
            n_estimators=grf_num_trees,
            min_samples_leaf=grf_min_node_size,
        )
        t0 = time.time()
        rcsf_model.fit_components(x, a, time_obs, event, z, w)
        rcsf_preds = np.asarray(rcsf_model.effect_from_components(x, w, z), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES["r_csf_baseline"],
            rcsf_preds,
            case.true_cate,
            time.time() - t0,
            backend=rcsf_model.__class__.__name__,
        )

    def _run_final_variant(name_key: str):
        common_kwargs = {
            "target": target,
            "horizon": horizon,
            "random_state": random_state,
            "surv_scalar_mode": "full",
            "n_estimators": num_trees,
            "censoring_estimator": "cox",
            "q_kind": q_kind,
            "h_kind": h_kind,
        }
        if name_key == "final_conditional_x_only":
            variant_model = FinalModelConditionalPCIXOnlyCensoredSurvivalForest(**common_kwargs)
        elif name_key == "final_conditional_broaddup_raw_only":
            variant_model = FinalModelConditionalCensoredSurvivalForest(
                **common_kwargs,
                nuisance_feature_mode="broad_dup",
                final_feature_mode="raw_only",
            )
        elif name_key == "final_conditional_dup_full":
            variant_model = FinalModelConditionalCensoredSurvivalForest(
                **common_kwargs,
                nuisance_feature_mode="dup",
                final_feature_mode="full",
            )
        elif name_key == "final_conditional_dup_raw_only":
            variant_model = FinalModelConditionalCensoredSurvivalForest(
                **common_kwargs,
                nuisance_feature_mode="dup",
                final_feature_mode="raw_only",
            )
        else:
            raise ValueError(f"Unsupported final variant {name_key!r}.")

        t0 = time.time()
        variant_model.fit_components(x, a, time_obs, event, z, w)
        variant_preds = np.asarray(variant_model.effect_from_components(x, w, z), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES[name_key],
            variant_preds,
            case.true_cate,
            time.time() - t0,
            backend=variant_model.__class__.__name__,
        )

    model_runners = {
        "final_conditional": run_final_conditional,
        "final_conditional_oracle": run_final_conditional_oracle,
        "censored_baseline": run_censored_baseline,
        "naive_censored_baseline": run_naive_censored_baseline,
        "r_csf_baseline": run_r_csf_baseline,
        "final_conditional_x_only": lambda: _run_final_variant("final_conditional_x_only"),
        "final_conditional_broaddup_raw_only": lambda: _run_final_variant("final_conditional_broaddup_raw_only"),
        "final_conditional_dup_full": lambda: _run_final_variant("final_conditional_dup_full"),
        "final_conditional_dup_raw_only": lambda: _run_final_variant("final_conditional_dup_raw_only"),
    }

    for model_key in selected_models:
        run_or_resume(MODEL_NAMES[model_key], model_runners[model_key])
    return rows


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


def _write_case_outputs(
    results: pd.DataFrame,
    *,
    output_dir: Path,
    case_id: int,
    case_slug: str,
    setting_id: str,
    target: str,
) -> None:
    case_df = results.loc[
        (results["case_id"] == int(case_id))
        & (results["case_slug"] == str(case_slug))
        & (results["setting_id"] == str(setting_id))
        & (results["target"] == target)
    ].copy()
    if case_df.empty:
        return
    case_df = case_df.sort_values(["name"]).reset_index(drop=True)
    case_base = _case_base_name(case_id, case_slug, setting_id, target)
    case_df.to_csv(output_dir / f"{case_base}.csv", index=False)
    render_case_table_png(case_df, output_dir / f"{case_base}.png")


def _persist_checkpoint(
    results: pd.DataFrame,
    *,
    output_dir: Path,
    latest_case_id: int | None = None,
    latest_case_slug: str | None = None,
    latest_setting_id: str | None = None,
    latest_target: str | None = None,
) -> None:
    results = _sort_results_frame(results)
    checkpoint_path = _checkpoint_path(output_dir)
    results_full_path = _results_full_path(output_dir)
    results.to_csv(checkpoint_path, index=False)
    results.to_csv(results_full_path, index=False)

    if results.empty:
        return

    if (
        latest_case_id is not None
        and latest_case_slug is not None
        and latest_setting_id is not None
        and latest_target is not None
    ):
        _write_case_outputs(
            results,
            output_dir=output_dir,
            case_id=latest_case_id,
            case_slug=latest_case_slug,
            setting_id=latest_setting_id,
            target=latest_target,
        )
    else:
        grouped = (
            results.loc[:, ["case_id", "case_slug", "setting_id", "target"]]
            .drop_duplicates()
            .sort_values(["case_id", "setting_id", "target"])
        )
        for row in grouped.to_dict("records"):
            _write_case_outputs(
                results,
                output_dir=output_dir,
                case_id=int(row["case_id"]),
                case_slug=str(row["case_slug"]),
                setting_id=str(row["setting_id"]),
                target=str(row["target"]),
            )

    _write_target_summaries(results, output_dir=output_dir, file_prefix="basic12_conditional_suite")


def main() -> int:
    args = parse_args()
    if args.list_cases:
        for spec in CASE_SPECS:
            print(f"{int(spec['case_id']):2d}  {spec['slug']}")
        return 0
    if args.list_settings:
        for setting in SETTINGS:
            print(
                f"{setting['setting_id']}: n={setting['n']} p_x={setting['p_x']} "
                f"p_w={setting['p_w']} p_z={setting['p_z']}"
            )
        return 0

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_gpu, gpu_reason = resolve_xgboost_gpu_mode(args.gpu)
    gpu_config = {
        "mode": str(args.gpu),
        "use_gpu": bool(use_gpu),
        "reason": str(gpu_reason),
    }
    print(f"[gpu] requested={args.gpu} enabled={use_gpu} reason={gpu_reason}", flush=True)

    case_specs = _selected_case_specs(args.case_ids, args.case_slugs)
    settings = _selected_settings(args.setting_ids)
    targets = _resolve_targets(args.target)
    run_metadata = _make_run_metadata(
        args=args,
        case_specs=case_specs,
        settings=settings,
        targets=targets,
    )
    checkpoint_results = _validate_or_initialize_resume_state(
        output_dir=output_dir,
        metadata=run_metadata,
    )
    checkpoint_results = _sort_results_frame(checkpoint_results)

    completed_lookup: dict[tuple[int, str, str, str], dict[str, object]] = {}
    for row in checkpoint_results.to_dict("records"):
        completed_lookup[
            (
                int(row["case_id"]),
                str(row["setting_id"]),
                str(row["target"]),
                str(row["name"]),
            )
        ] = dict(row)

    selected_models = list(args.models)
    total_steps = len(case_specs) * len(settings) * len(targets) * len(selected_models)
    completed_steps = len(completed_lookup)

    def save_row(row: dict[str, object]) -> None:
        nonlocal checkpoint_results
        row_df = pd.DataFrame([row])
        checkpoint_results = pd.concat([checkpoint_results, row_df], ignore_index=True)
        checkpoint_results = checkpoint_results.drop_duplicates(
            subset=["case_id", "setting_id", "target", "name"],
            keep="last",
        )
        checkpoint_results = _sort_results_frame(checkpoint_results)
        completed_lookup[
            (
                int(row["case_id"]),
                str(row["setting_id"]),
                str(row["target"]),
                str(row["name"]),
            )
        ] = dict(row)
        _persist_checkpoint(
            checkpoint_results,
            output_dir=output_dir,
            latest_case_id=int(row["case_id"]),
            latest_case_slug=str(row["case_slug"]),
            latest_setting_id=str(row["setting_id"]),
            latest_target=str(row["target"]),
        )

    try:
        for case_spec in case_specs:
            case_id = int(case_spec["case_id"])
            case_slug = str(case_spec["slug"])
            case_title = str(case_spec["title"])
            for setting in settings:
                setting_id = str(setting["setting_id"])
                case_with_setting = _case_spec_with_setting(case_spec, setting)
                for target in targets:
                    if target == "RMST":
                        case = prepare_case(case_with_setting, target=target, horizon=float(args.rmst_horizon))
                    else:
                        case = prepare_case(
                            case_with_setting,
                            target=target,
                            horizon_quantile=float(args.horizon_quantile),
                        )

                    existing_rows_by_name = {
                        name: row
                        for (row_case_id, row_setting_id, row_target, name), row in completed_lookup.items()
                        if row_case_id == case_id and row_setting_id == setting_id and row_target == target
                    }

                    def progress_hook(row, *, resumed: bool, _case_id=case_id, _case_slug=case_slug, _setting_id=setting_id, _target=target):
                        nonlocal completed_steps
                        line = _format_progress_line(
                            completed_steps,
                            total_steps,
                            row,
                            case_id=_case_id,
                            case_slug=_case_slug,
                            target=_target,
                        )
                        if resumed:
                            print(f"[resume] {line} setting={_setting_id}", flush=True)
                            return

                        completed_steps += 1
                        line = _format_progress_line(
                            completed_steps,
                            total_steps,
                            row,
                            case_id=_case_id,
                            case_slug=_case_slug,
                            target=_target,
                        )
                        print(f"{line} setting={_setting_id}", flush=True)

                    _evaluate_case_target(
                        case,
                        case_id=case_id,
                        case_slug=case_slug,
                        case_title=case_title,
                        setting=setting,
                        target=target,
                        random_state=args.random_state,
                        num_trees=args.num_trees,
                        selected_models=selected_models,
                        grf_num_trees=args.grf_num_trees,
                        grf_min_node_size=args.grf_min_node_size,
                        gpu_config=gpu_config,
                        existing_rows_by_name=existing_rows_by_name,
                        save_row_hook=save_row,
                        progress_hook=progress_hook,
                    )
    except KeyboardInterrupt:
        if not checkpoint_results.empty:
            _persist_checkpoint(checkpoint_results, output_dir=output_dir)
        print(
            "\nInterrupted. Completed model rows have been checkpointed. "
            f"Re-run the script with the same arguments to resume from {output_dir}.",
            file=sys.stderr,
            flush=True,
        )
        return 130

    results = _sort_results_frame(checkpoint_results)
    _persist_checkpoint(results, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
