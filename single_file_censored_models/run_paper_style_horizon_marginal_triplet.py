#!/usr/bin/env python3
"""Paper-style horizon benchmark for the marginal-censoring triplet.

Models:
    1. Final Marginal Oracle
    2. Final Marginal
    3. Revised Marginal

RMST is evaluated at the four paper-style fixed horizons:
    1.5, 2.0, 15.0, 3.0

Survival probability is evaluated at the paper-style rule:
    horizon = q-th percentile of observed follow-up time (default q = 0.90)

This runner checkpoints every completed fit so interrupted runs can resume.
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
    from .final_censored_model import FinalModelCensoredSurvivalForest, _ensure_2d  # noqa: E402
    from .gpu_backends import resolve_xgboost_gpu_mode  # noqa: E402
    from .revised_censored_baseline import RevisedBaselineCensoredSurvivalForest  # noqa: E402
    from .run_5model_benchmark import CASE_SPECS, SETTINGS, SingleFileFinalOracleCensoredSurvivalForest  # noqa: E402
except ImportError:  # pragma: no cover
    from single_file_censored_models.final_censored_model import FinalModelCensoredSurvivalForest, _ensure_2d  # type: ignore  # noqa: E402
    from single_file_censored_models.gpu_backends import resolve_xgboost_gpu_mode  # type: ignore  # noqa: E402
    from single_file_censored_models.revised_censored_baseline import RevisedBaselineCensoredSurvivalForest  # type: ignore  # noqa: E402
    from single_file_censored_models.run_5model_benchmark import CASE_SPECS, SETTINGS, SingleFileFinalOracleCensoredSurvivalForest  # type: ignore  # noqa: E402


TARGETS = ["RMST", "survival.probability"]
MODEL_NAMES = {
    "final_marginal_oracle": "Final Marginal Oracle",
    "final_marginal": "Final Marginal",
    "revised_marginal": "Revised Marginal",
}
MODEL_COUNT = len(MODEL_NAMES)
CHECKPOINT_RESULTS_FILE = "results_incremental.csv"
RUN_METADATA_FILE = "run_metadata.json"
RMST_PAPER_HORIZONS = [
    {"label": "RMST-H1", "order": 1, "horizon": 1.5, "description": "Paper fixed RMST horizon 1.5"},
    {"label": "RMST-H2", "order": 2, "horizon": 2.0, "description": "Paper fixed RMST horizon 2.0"},
    {"label": "RMST-H3", "order": 3, "horizon": 15.0, "description": "Paper fixed RMST horizon 15.0"},
    {"label": "RMST-H4", "order": 4, "horizon": 3.0, "description": "Paper fixed RMST horizon 3.0"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper-style horizon benchmark for Final Marginal Oracle / "
            "Final Marginal / Revised Marginal."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int, default=None)
    parser.add_argument("--case-slugs", nargs="*", default=None)
    parser.add_argument("--setting-ids", nargs="*", default=None)
    parser.add_argument("--target", choices=["RMST", "survival.probability", "both"], default="both")
    parser.add_argument(
        "--horizon-quantile",
        type=float,
        default=0.90,
        help="Observed follow-up quantile used for survival.probability. Default: 0.90.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-trees", type=int, default=200, help="Trees for Final/Revised Marginal.")
    parser.add_argument("--grf-num-trees", type=int, default=200, help="Deprecated compatibility flag; ignored.")
    parser.add_argument("--grf-min-node-size", type=int, default=20, help="Deprecated compatibility flag; ignored.")
    parser.add_argument("--gpu", choices=["off", "auto", "xgboost"], default="auto")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-settings", action="store_true")
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / "paper_style_horizon_marginal_triplet").resolve()


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


def _target_horizon_specs(target: str, *, survival_quantile: float) -> list[dict[str, object]]:
    if target == "RMST":
        return [dict(spec) for spec in RMST_PAPER_HORIZONS]
    return [
        {
            "label": f"SURV-Q{int(round(100 * survival_quantile)):02d}",
            "order": 1,
            "horizon": None,
            "description": f"Observed follow-up quantile q={survival_quantile:.2f}",
            "horizon_quantile": float(survival_quantile),
        }
    ]


def _make_run_metadata(
    *,
    args: argparse.Namespace,
    case_specs: list[dict[str, object]],
    settings: list[dict[str, int | str]],
    targets: list[str],
) -> dict[str, object]:
    return {
        "script": THIS_FILE.name,
        "runner_version": 2,
        "case_ids": [int(spec["case_id"]) for spec in case_specs],
        "case_slugs": [str(spec["slug"]) for spec in case_specs],
        "setting_ids": [str(setting["setting_id"]) for setting in settings],
        "targets": list(targets),
        "rmst_paper_horizons": [float(spec["horizon"]) for spec in RMST_PAPER_HORIZONS],
        "survival_horizon_quantile": float(args.horizon_quantile),
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
    sort_cols = ["target", "horizon_order", "case_id", "setting_id", "name"]
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
        results.groupby(["target", "horizon_label", "horizon_order", "name"], as_index=False)
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
            ["target", "horizon_order", "avg_rmse", "avg_mae", "avg_abs_bias", "avg_pearson_correlation"],
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
    fig.suptitle(f"Paper-Style Horizon Benchmark: {target}", fontsize=16, fontweight="bold")

    names = list(MODEL_NAMES.values())
    colors = {
        "Final Marginal Oracle": "#111827",
        "Final Marginal": "#2563eb",
        "Revised Marginal": "#dc2626",
    }
    x_order = summary_df[["horizon_label", "horizon_order"]].drop_duplicates().sort_values("horizon_order")
    x_positions = np.arange(len(x_order))

    for ax, (metric_col, metric_title) in zip(axes.ravel(), metric_specs):
        for name in names:
            model_df = summary_df.loc[summary_df["name"] == name].copy()
            if model_df.empty:
                continue
            model_df = x_order.merge(model_df, on=["horizon_label", "horizon_order"], how="left")
            ax.plot(
                x_positions,
                model_df[metric_col].to_numpy(),
                marker="o",
                linewidth=2,
                markersize=5,
                label=name,
                color=colors.get(name),
            )
        ax.set_title(metric_title)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_order["horizon_label"].tolist(), rotation=25, ha="right")
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
    summary.to_csv(output_dir / "paper_style_horizon_summary.csv", index=False)

    for target in TARGETS:
        target_slug = target.replace(".", "_")
        target_df = summary.loc[summary["target"] == target].copy()
        if target_df.empty:
            continue
        target_df.to_csv(output_dir / f"paper_style_horizon_{target_slug}_summary.csv", index=False)
        _render_target_metric_grid(
            target_df,
            output_dir / f"paper_style_horizon_{target_slug}_metrics.png",
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
    horizon_label: str,
) -> str:
    return (
        f"[{step_idx}/{total_steps}] horizon={horizon_label} "
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
    horizon_spec: dict[str, object],
    random_state: int,
    num_trees: int,
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
        row["target"] = str(target)
        row["case_id"] = int(case_id)
        row["case_slug"] = str(case_slug)
        row["case_title"] = str(case_title)
        row["setting_id"] = str(setting["setting_id"])
        row["n"] = int(setting["n"])
        row["p_x"] = int(setting["p_x"])
        row["p_w"] = int(setting["p_w"])
        row["p_z"] = int(setting["p_z"])
        row["horizon"] = float(horizon)
        row["horizon_label"] = str(horizon_spec["label"])
        row["horizon_order"] = int(horizon_spec["order"])
        row["horizon_description"] = str(horizon_spec["description"])
        if target == "RMST":
            row["horizon_rule"] = "paper_fixed_rmst"
            row["horizon_quantile"] = np.nan
        else:
            row["horizon_rule"] = f"observed_followup_q{float(horizon_spec['horizon_quantile']):.2f}"
            row["horizon_quantile"] = float(horizon_spec["horizon_quantile"])
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

    def run_final_marginal():
        model = FinalModelCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=random_state,
            surv_scalar_mode="full",
            n_estimators=num_trees,
            q_kind=q_kind,
            h_kind=h_kind,
        )
        t0 = time.time()
        model.fit_components(x, a, time_obs, event, z, w)
        preds = np.asarray(model.effect_from_components(x, w, z), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES["final_marginal"],
            preds,
            case.true_cate,
            time.time() - t0,
            backend=model.__class__.__name__,
        )

    def run_final_marginal_oracle():
        _ = (grf_num_trees, grf_min_node_size)
        u = _ensure_2d(case.U)
        x_oracle = np.column_stack([x, u])
        model = SingleFileFinalOracleCensoredSurvivalForest(
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
            censoring_estimator="nelson-aalen",
            event_survival_estimator="cox",
            m_pred_mode="bridge",
            nuisance_feature_mode="broad_dup",
        )
        t0 = time.time()
        model.fit_oracle(x_oracle, a, time_obs, event, u)
        preds = np.asarray(model.effect_oracle(x, u), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES["final_marginal_oracle"],
            preds,
            case.true_cate,
            time.time() - t0,
            backend=model.__class__.__name__,
        )

    def run_revised_marginal():
        model = RevisedBaselineCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=random_state,
            n_estimators=num_trees,
            propensity_kind=propensity_kind,
            censoring_estimator="nelson-aalen",
        )
        t0 = time.time()
        model.fit_components(x, a, time_obs, event, z, w)
        preds = np.asarray(model.effect_from_components(x, w, z), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES["revised_marginal"],
            preds,
            case.true_cate,
            time.time() - t0,
            backend=model.__class__.__name__,
        )

    run_or_resume(MODEL_NAMES["final_marginal_oracle"], run_final_marginal_oracle)
    run_or_resume(MODEL_NAMES["final_marginal"], run_final_marginal)
    run_or_resume(MODEL_NAMES["revised_marginal"], run_revised_marginal)
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
    if not (0.0 < float(args.horizon_quantile) <= 1.0):
        raise ValueError("--horizon-quantile must lie in (0, 1].")

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
    checkpoint_results = _validate_or_initialize_resume_state(output_dir=output_dir, metadata=run_metadata)
    checkpoint_results = _sort_results_frame(checkpoint_results)

    completed_lookup: dict[tuple[int, str, str, str, str], dict[str, object]] = {}
    for row in checkpoint_results.to_dict("records"):
        completed_lookup[
            (
                int(row["case_id"]),
                str(row["setting_id"]),
                str(row["target"]),
                str(row["horizon_label"]),
                str(row["name"]),
            )
        ] = dict(row)

    total_target_horizon_runs = sum(len(_target_horizon_specs(target, survival_quantile=args.horizon_quantile)) for target in targets)
    total_steps = len(case_specs) * len(settings) * total_target_horizon_runs * MODEL_COUNT
    completed_steps = len(completed_lookup)

    def save_row(row: dict[str, object]) -> None:
        nonlocal checkpoint_results
        row_df = pd.DataFrame([row])
        checkpoint_results = pd.concat([checkpoint_results, row_df], ignore_index=True)
        checkpoint_results = checkpoint_results.drop_duplicates(
            subset=["case_id", "setting_id", "target", "horizon_label", "name"],
            keep="last",
        )
        checkpoint_results = _sort_results_frame(checkpoint_results)
        completed_lookup[
            (
                int(row["case_id"]),
                str(row["setting_id"]),
                str(row["target"]),
                str(row["horizon_label"]),
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
                    for horizon_spec in _target_horizon_specs(target, survival_quantile=args.horizon_quantile):
                        if target == "RMST":
                            case = prepare_case(case_with_setting, target=target, horizon=float(horizon_spec["horizon"]))
                        else:
                            case = prepare_case(
                                case_with_setting,
                                target=target,
                                horizon_quantile=float(horizon_spec["horizon_quantile"]),
                            )

                        existing_rows_by_name = {
                            name: row
                            for (row_case_id, row_setting_id, row_target, row_horizon_label, name), row in completed_lookup.items()
                            if row_case_id == case_id
                            and row_setting_id == setting_id
                            and row_target == target
                            and row_horizon_label == str(horizon_spec["label"])
                        }

                        def progress_hook(
                            row,
                            *,
                            resumed: bool,
                            _case_id=case_id,
                            _case_slug=case_slug,
                            _setting_id=setting_id,
                            _target=target,
                            _horizon_label=str(horizon_spec["label"]),
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
                                horizon_label=_horizon_label,
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
                                horizon_label=_horizon_label,
                            )
                            print(line, flush=True)

                        _evaluate_case_target_horizon(
                            case,
                            case_id=case_id,
                            case_slug=case_slug,
                            case_title=case_title,
                            setting=setting,
                            target=target,
                            horizon_spec=horizon_spec,
                            random_state=args.random_state,
                            num_trees=args.num_trees,
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
