#!/usr/bin/env python3
"""Sweep horizon quantiles for the three conditional-censoring models.

Models:
    1. Final Conditional Oracle
    2. Final Conditional
    3. Revised Conditional

This runner keeps the existing model implementations untouched and only varies
the synthetic benchmark horizon quantile. It supports checkpointing/resume, so
completed rows are saved after every model fit.
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
    from .final_censored_model import _ensure_2d  # noqa: E402
    from .final_censored_model_conditional import FinalModelConditionalCensoredSurvivalForest  # noqa: E402
    from .gpu_backends import make_xgb_classifier, make_xgb_regressor, resolve_xgboost_gpu_mode  # noqa: E402
    from .revised_censored_baseline import RevisedBaselineCensoredSurvivalForest  # noqa: E402
    from .run_5model_benchmark import (  # noqa: E402
        CASE_SPECS,
        SETTINGS,
        SingleFileFinalOracleCensoredSurvivalForest,
    )
except ImportError:  # pragma: no cover
    from single_file_censored_models.final_censored_model import _ensure_2d  # type: ignore  # noqa: E402
    from single_file_censored_models.final_censored_model_conditional import (  # type: ignore  # noqa: E402
        FinalModelConditionalCensoredSurvivalForest,
    )
    from single_file_censored_models.gpu_backends import (  # type: ignore  # noqa: E402
        make_xgb_classifier,
        make_xgb_regressor,
        resolve_xgboost_gpu_mode,
    )
    from single_file_censored_models.revised_censored_baseline import (  # type: ignore  # noqa: E402
        RevisedBaselineCensoredSurvivalForest,
    )
    from single_file_censored_models.run_5model_benchmark import (  # type: ignore  # noqa: E402
        CASE_SPECS,
        SETTINGS,
        SingleFileFinalOracleCensoredSurvivalForest,
    )


TARGETS = ["RMST", "survival.probability"]
DEFAULT_HORIZON_QUANTILES = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]
MODEL_NAMES = {
    "final_conditional_oracle": "Final Conditional Oracle",
    "final_conditional": "Final Conditional",
    "revised_conditional": "Revised Conditional",
}
MODEL_COUNT = len(MODEL_NAMES)
CHECKPOINT_RESULTS_FILE = "results_incremental.csv"
RUN_METADATA_FILE = "run_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the conditional-censoring horizon sweep for Final Conditional "
            "Oracle / Final Conditional / Revised Conditional."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int, default=None)
    parser.add_argument("--case-slugs", nargs="*", default=None)
    parser.add_argument("--setting-ids", nargs="*", default=None)
    parser.add_argument("--target", choices=["RMST", "survival.probability", "both"], default="both")
    parser.add_argument(
        "--horizon-quantiles",
        nargs="*",
        type=float,
        default=None,
        help="Space-separated horizon quantiles. Defaults to 0.1 0.2 ... 0.9.",
    )
    parser.add_argument(
        "--repeat-fixed-rmst",
        action="store_true",
        help=(
            "By default RMST is evaluated only once, because prepare_case fixes the RMST horizon "
            "to max(Y) regardless of horizon_quantile. Enable this flag only if you want duplicate "
            "RMST rows for every requested horizon quantile."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-trees", type=int, default=200)
    parser.add_argument("--gpu", choices=["off", "auto", "xgboost"], default="auto")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-settings", action="store_true")
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / "horizon_sweep_conditional_triplet").resolve()


def _selected_case_specs(case_ids: list[int] | None, case_slugs: list[str] | None) -> list[dict[str, object]]:
    if not case_ids and not case_slugs:
        return list(CASE_SPECS)
    id_set = set(case_ids or [])
    slug_set = set(case_slugs or [])
    selected = [spec for spec in CASE_SPECS if int(spec["case_id"]) in id_set or str(spec["slug"]) in slug_set]
    if not selected:
        raise ValueError("No synthetic cases matched the requested ids/slugs.")
    return selected


def _selected_settings(setting_ids: list[str] | None) -> list[dict[str, int | str]]:
    if not setting_ids:
        return list(SETTINGS)
    wanted = {str(s).upper() for s in setting_ids}
    selected = [setting for setting in SETTINGS if str(setting["setting_id"]).upper() in wanted]
    if not selected:
        raise ValueError("No settings matched the requested setting ids.")
    return selected


def _resolve_targets(target_arg: str) -> list[str]:
    if target_arg == "both":
        return list(TARGETS)
    return [target_arg]


def _resolve_horizon_quantiles(values: list[float] | None) -> list[float]:
    if not values:
        values = list(DEFAULT_HORIZON_QUANTILES)
    normalized = sorted({round(float(v), 4) for v in values})
    for value in normalized:
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Each horizon quantile must lie in (0, 1]. Received {value}.")
    return normalized


def _effective_horizon_quantiles_for_target(
    target: str,
    horizon_quantiles: list[float],
    *,
    repeat_fixed_rmst: bool,
) -> list[float]:
    if target == "RMST" and not repeat_fixed_rmst:
        return [float(horizon_quantiles[0])]
    return list(horizon_quantiles)


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


def _make_run_metadata(
    *,
    args: argparse.Namespace,
    case_specs: list[dict[str, object]],
    settings: list[dict[str, int | str]],
    targets: list[str],
    horizon_quantiles: list[float],
) -> dict[str, object]:
    return {
        "script": THIS_FILE.name,
        "runner_version": 1,
        "case_ids": [int(spec["case_id"]) for spec in case_specs],
        "case_slugs": [str(spec["slug"]) for spec in case_specs],
        "setting_ids": [str(setting["setting_id"]) for setting in settings],
        "targets": list(targets),
        "horizon_quantiles": [float(v) for v in horizon_quantiles],
        "repeat_fixed_rmst": bool(args.repeat_fixed_rmst),
        "random_state": int(args.random_state),
        "num_trees": int(args.num_trees),
        "gpu_mode": str(args.gpu),
        "model_count": MODEL_COUNT,
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


def _validate_or_initialize_resume_state(*, output_dir: Path, metadata: dict[str, object]) -> pd.DataFrame:
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
    sort_cols = ["horizon_quantile", "case_id", "setting_id", "target", "name"]
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


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["target", "horizon_quantile", "name"], as_index=False)
        .agg(
            avg_horizon=("horizon", "mean"),
            avg_pred_cate=("mean_pred", "mean"),
            avg_true_cate=("mean_true_cate", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson_correlation=("pearson_correlation", "mean"),
            avg_bias=("bias", "mean"),
            avg_abs_bias=("bias", lambda values: float(np.mean(np.abs(values)))),
            avg_sign_precision=("sign_precision", "mean"),
            avg_total_time=("total_time", "mean"),
            case_count=("case_id", "count"),
        )
        .sort_values(
            ["target", "horizon_quantile", "avg_rmse", "avg_mae", "avg_abs_bias", "avg_pearson_correlation"],
            ascending=[True, True, True, True, True, False],
        )
        .reset_index(drop=True)
    )
    return summary.drop(columns=["avg_abs_bias"])


def _render_target_metric_grid(summary_df: pd.DataFrame, output_path: Path, *, target: str) -> None:
    metric_specs = [
        ("avg_rmse", "RMSE"),
        ("avg_mae", "MAE"),
        ("avg_pearson_correlation", "Pearson"),
        ("avg_bias", "Bias"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle(f"Horizon Sweep: {target}", fontsize=16, fontweight="bold")

    names = list(MODEL_NAMES.values())
    colors = {
        "Final Conditional Oracle": "#111827",
        "Final Conditional": "#2563eb",
        "Revised Conditional": "#dc2626",
    }

    for ax, (metric_col, metric_title) in zip(axes.ravel(), metric_specs):
        for name in names:
            model_df = summary_df.loc[summary_df["name"] == name].sort_values("horizon_quantile")
            if model_df.empty:
                continue
            ax.plot(
                model_df["horizon_quantile"].to_numpy(),
                model_df[metric_col].to_numpy(),
                marker="o",
                linewidth=2,
                markersize=5,
                label=name,
                color=colors.get(name),
            )
        ax.set_title(metric_title)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_xlabel("Horizon Quantile")
        ax.set_ylabel(metric_title)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_summary_outputs(results: pd.DataFrame, *, output_dir: Path) -> None:
    if results.empty:
        return
    summary = summarize_results(results)
    summary.to_csv(output_dir / "horizon_sweep_summary.csv", index=False)

    for target in TARGETS:
        target_slug = target.replace(".", "_")
        target_df = summary.loc[summary["target"] == target].copy()
        if target_df.empty:
            continue
        target_df.to_csv(output_dir / f"horizon_sweep_{target_slug}_summary.csv", index=False)
        _render_target_metric_grid(
            target_df,
            output_dir / f"horizon_sweep_{target_slug}_metrics.png",
            target=target,
        )


def _persist_checkpoint(results: pd.DataFrame, *, output_dir: Path) -> None:
    results = _sort_results_frame(results)
    results.to_csv(_checkpoint_path(output_dir), index=False)
    results.to_csv(_results_full_path(output_dir), index=False)
    _write_summary_outputs(results, output_dir=output_dir)


def _format_progress_line(
    step_idx: int,
    total_steps: int,
    row: dict[str, object],
    *,
    case_id: int,
    case_slug: str,
    setting_id: str,
    target: str,
    horizon_quantile: float,
) -> str:
    return (
        f"[{step_idx}/{total_steps}] hq={horizon_quantile:.1f} "
        f"case={case_id:02d} slug={case_slug} setting={setting_id} target={target} "
        f"model={row['name']} rmse={float(row['rmse']):.4f} mae={float(row['mae']):.4f} "
        f"corr={float(row['pearson_correlation']):.4f} bias={float(row['bias']):+.4f} "
        f"sign_precision={100.0 * float(row['sign_precision']):.1f}% "
        f"time={float(row['total_time']):.1f}s"
    )


def _evaluate_case_target_horizon(
    case,
    *,
    case_id: int,
    case_slug: str,
    case_title: str,
    setting: dict[str, int | str],
    target: str,
    horizon_quantile: float,
    random_state: int,
    num_trees: int,
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
        row["target"] = str(target)
        row["horizon_quantile"] = float(horizon_quantile)
        row["case_id"] = int(case_id)
        row["case_slug"] = str(case_slug)
        row["case_title"] = str(case_title)
        row["setting_id"] = str(setting["setting_id"])
        row["n"] = int(setting["n"])
        row["p_x"] = int(setting["p_x"])
        row["p_w"] = int(setting["p_w"])
        row["p_z"] = int(setting["p_z"])
        row["horizon"] = float(horizon)
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
        preds = np.asarray(final_model.effect_from_components(x, w, z), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES["final_conditional"],
            preds,
            case.true_cate,
            time.time() - t0,
            backend=final_model.__class__.__name__,
        )

    def run_final_conditional_oracle():
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
        preds = np.asarray(oracle_model.effect_oracle(x, u), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES["final_conditional_oracle"],
            preds,
            case.true_cate,
            time.time() - t0,
            backend=oracle_model.__class__.__name__,
        )

    def run_revised_conditional():
        revised_model = RevisedBaselineCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=random_state,
            n_estimators=num_trees,
            censoring_estimator="cox",
            propensity_kind=propensity_kind,
        )
        t0 = time.time()
        revised_model.fit_components(x, a, time_obs, event, z, w)
        preds = np.asarray(revised_model.effect_from_components(x, w, z), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES["revised_conditional"],
            preds,
            case.true_cate,
            time.time() - t0,
            backend=revised_model.__class__.__name__,
        )

    run_or_resume(MODEL_NAMES["final_conditional_oracle"], run_final_conditional_oracle)
    run_or_resume(MODEL_NAMES["final_conditional"], run_final_conditional)
    run_or_resume(MODEL_NAMES["revised_conditional"], run_revised_conditional)
    return rows


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

    if int(args.num_trees) % 4 != 0:
        raise ValueError(
            "--num-trees must be divisible by 4 for the EconML causal forest backend "
            f"(received {args.num_trees})."
        )

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
    horizon_quantiles = _resolve_horizon_quantiles(args.horizon_quantiles)
    if "RMST" in targets and len(horizon_quantiles) > 1 and not args.repeat_fixed_rmst:
        print(
            "[note] RMST horizon is fixed to max(Y) inside prepare_case, so RMST results do not change with "
            "horizon_quantile in the current synthetic benchmark. Running RMST only at the first requested "
            f"quantile ({horizon_quantiles[0]:.1f}). Use --repeat-fixed-rmst to force duplicate RMST rows.",
            flush=True,
        )

    run_metadata = _make_run_metadata(
        args=args,
        case_specs=case_specs,
        settings=settings,
        targets=targets,
        horizon_quantiles=horizon_quantiles,
    )
    checkpoint_results = _validate_or_initialize_resume_state(
        output_dir=output_dir,
        metadata=run_metadata,
    )
    checkpoint_results = _sort_results_frame(checkpoint_results)

    completed_lookup: dict[tuple[int, str, str, str, str], dict[str, object]] = {}
    for row in checkpoint_results.to_dict("records"):
        hq_key = f"{float(row['horizon_quantile']):.4f}"
        completed_lookup[
            (
                int(row["case_id"]),
                str(row["setting_id"]),
                str(row["target"]),
                hq_key,
                str(row["name"]),
            )
        ] = dict(row)

    total_target_horizon_runs = sum(
        len(
            _effective_horizon_quantiles_for_target(
                target,
                horizon_quantiles,
                repeat_fixed_rmst=args.repeat_fixed_rmst,
            )
        )
        for target in targets
    )
    total_steps = len(case_specs) * len(settings) * total_target_horizon_runs * MODEL_COUNT
    completed_steps = len(completed_lookup)

    def save_row(row: dict[str, object]) -> None:
        nonlocal checkpoint_results
        row_df = pd.DataFrame([row])
        checkpoint_results = pd.concat([checkpoint_results, row_df], ignore_index=True)
        checkpoint_results = checkpoint_results.drop_duplicates(
            subset=["case_id", "setting_id", "target", "horizon_quantile", "name"],
            keep="last",
        )
        checkpoint_results = _sort_results_frame(checkpoint_results)
        completed_lookup[
            (
                int(row["case_id"]),
                str(row["setting_id"]),
                str(row["target"]),
                f"{float(row['horizon_quantile']):.4f}",
                str(row["name"]),
            )
        ] = dict(row)
        _persist_checkpoint(checkpoint_results, output_dir=output_dir)

    try:
        for case_spec in case_specs:
            case_id = int(case_spec["case_id"])
            case_slug = str(case_spec["slug"])
            case_title = str(case_spec["title"])
            for setting in settings:
                setting_id = str(setting["setting_id"])
                case_with_setting = _case_spec_with_setting(case_spec, setting)
                for target in targets:
                    target_horizon_quantiles = _effective_horizon_quantiles_for_target(
                        target,
                        horizon_quantiles,
                        repeat_fixed_rmst=args.repeat_fixed_rmst,
                    )
                    for horizon_quantile in target_horizon_quantiles:
                        case = prepare_case(case_with_setting, target=target, horizon_quantile=horizon_quantile)
                        hq_key = f"{float(horizon_quantile):.4f}"
                        existing_rows_by_name = {
                            name: row
                            for (row_case_id, row_setting_id, row_target, row_hq_key, name), row in completed_lookup.items()
                            if row_case_id == case_id
                            and row_setting_id == setting_id
                            and row_target == target
                            and row_hq_key == hq_key
                        }

                        def progress_hook(
                            row,
                            *,
                            resumed: bool,
                            _case_id=case_id,
                            _case_slug=case_slug,
                            _setting_id=setting_id,
                            _target=target,
                            _horizon_quantile=horizon_quantile,
                        ):
                            nonlocal completed_steps
                            line = _format_progress_line(
                                completed_steps,
                                total_steps,
                                row,
                                case_id=_case_id,
                                case_slug=_case_slug,
                                setting_id=_setting_id,
                                target=_target,
                                horizon_quantile=_horizon_quantile,
                            )
                            if resumed:
                                print(f"[resume] {line}", flush=True)
                                return

                            completed_steps += 1
                            line = _format_progress_line(
                                completed_steps,
                                total_steps,
                                row,
                                case_id=_case_id,
                                case_slug=_case_slug,
                                setting_id=_setting_id,
                                target=_target,
                                horizon_quantile=_horizon_quantile,
                            )
                            print(line, flush=True)

                        _evaluate_case_target_horizon(
                            case,
                            case_id=case_id,
                            case_slug=case_slug,
                            case_title=case_title,
                            setting=setting,
                            target=target,
                            horizon_quantile=horizon_quantile,
                            random_state=args.random_state,
                            num_trees=args.num_trees,
                            gpu_config=gpu_config,
                            existing_rows_by_name=existing_rows_by_name,
                            save_row_hook=save_row,
                            progress_hook=progress_hook,
                        )
    except KeyboardInterrupt:
        if not checkpoint_results.empty:
            _persist_checkpoint(checkpoint_results, output_dir=output_dir)
        print(
            "\nInterrupted. Completed rows have been checkpointed. "
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
