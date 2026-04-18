#!/usr/bin/env python3
"""Run the missing marginal full-22 benchmark pieces.

Models:
    1. Final Marginal
    2. R-CSF Baseline

This runner mirrors the existing 22-setting synthetic benchmark convention:

    - RMST uses the benchmark default horizon h = max(Y)
    - survival.probability uses the observed follow-up q-quantile
      (default q = 0.60, to match the existing full-22 Final Conditional
      comparison outputs)

It checkpoints after every completed fit so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


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
    from .final_censored_model import FinalModelCensoredSurvivalForest  # noqa: E402
    from .gpu_backends import resolve_xgboost_gpu_mode  # noqa: E402
    from .grf_censored_baseline import GRFCensoredBaseline  # noqa: E402
    from .run_5model_benchmark import CASE_SPECS, SETTINGS  # noqa: E402
except ImportError:  # pragma: no cover
    from single_file_censored_models.final_censored_model import FinalModelCensoredSurvivalForest  # type: ignore  # noqa: E402
    from single_file_censored_models.gpu_backends import resolve_xgboost_gpu_mode  # type: ignore  # noqa: E402
    from single_file_censored_models.grf_censored_baseline import GRFCensoredBaseline  # type: ignore  # noqa: E402
    from single_file_censored_models.run_5model_benchmark import CASE_SPECS, SETTINGS  # type: ignore  # noqa: E402


TARGETS = ["RMST", "survival.probability"]
MODEL_NAMES = {
    "final_marginal": "Final Marginal",
    "r_csf": "R-CSF Baseline",
}
MODEL_COUNT = len(MODEL_NAMES)
CHECKPOINT_RESULTS_FILE = "results_incremental.csv"
RUN_METADATA_FILE = "run_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full 22-setting synthetic benchmark for Final Marginal and the R-CSF baseline."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int, default=None)
    parser.add_argument("--case-slugs", nargs="*", default=None)
    parser.add_argument("--setting-ids", nargs="*", default=None)
    parser.add_argument("--target", choices=["RMST", "survival.probability", "both"], default="both")
    parser.add_argument(
        "--horizon-quantile",
        type=float,
        default=0.60,
        help="Observed follow-up quantile used when target=survival.probability.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-trees", type=int, default=200, help="Trees for Final Marginal.")
    parser.add_argument("--grf-num-trees", type=int, default=200, help="Trees for the R-CSF baseline.")
    parser.add_argument("--grf-min-node-size", type=int, default=20, help="Minimum node size for the R-CSF baseline.")
    parser.add_argument("--gpu", choices=["off", "auto", "xgboost"], default="auto")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-settings", action="store_true")
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / "final_marginal_rcsf_full22").resolve()


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


def _make_run_metadata(
    *,
    args: argparse.Namespace,
    case_specs: list[dict[str, object]],
    settings: list[dict[str, int | str]],
    targets: list[str],
) -> dict[str, object]:
    return {
        "script": THIS_FILE.name,
        "runner_version": 1,
        "case_ids": [int(spec["case_id"]) for spec in case_specs],
        "case_slugs": [str(spec["slug"]) for spec in case_specs],
        "setting_ids": [str(setting["setting_id"]) for setting in settings],
        "targets": list(targets),
        "horizon_quantile": float(args.horizon_quantile),
        "random_state": int(args.random_state),
        "num_trees": int(args.num_trees),
        "grf_num_trees": int(args.grf_num_trees),
        "grf_min_node_size": int(args.grf_min_node_size),
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


def _summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["target", "name"], as_index=False)
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
            cell_count=("case_id", "count"),
        )
        .sort_values(
            ["target", "avg_rmse", "avg_mae", "avg_abs_bias", "avg_pearson_correlation"],
            ascending=[True, True, True, True, False],
        )
        .reset_index(drop=True)
    )
    return summary.drop(columns=["avg_abs_bias"])


def _write_summary_outputs(results: pd.DataFrame, *, output_dir: Path) -> None:
    if results.empty:
        return
    summary = _summarize_results(results)
    summary.to_csv(output_dir / "summary_full.csv", index=False)
    for target in TARGETS:
        target_df = summary.loc[summary["target"] == target].copy()
        if target_df.empty:
            continue
        target_slug = target.replace(".", "_")
        target_df.to_csv(output_dir / f"summary_{target_slug}.csv", index=False)


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
) -> str:
    return (
        f"[{step_idx}/{total_steps}] case={case_id:02d} slug={case_slug} setting={setting_id} target={target} "
        f"model={row['name']} rmse={float(row['rmse']):.4f} mae={float(row['mae']):.4f} "
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

    def run_r_csf():
        model = GRFCensoredBaseline(
            target=target,
            horizon=horizon,
            n_estimators=grf_num_trees,
            min_samples_leaf=grf_min_node_size,
            random_state=random_state,
        )
        t0 = time.time()
        model.fit_components(x, a, time_obs, event, z, w)
        preds = np.asarray(model.effect_from_components(x, w, z), dtype=float).ravel()
        elapsed = time.time() - t0
        model.cleanup()
        return _evaluate_predictions(
            MODEL_NAMES["r_csf"],
            preds,
            case.true_cate,
            elapsed,
            backend=model.__class__.__name__,
        )

    run_or_resume(MODEL_NAMES["final_marginal"], run_final_marginal)
    run_or_resume(MODEL_NAMES["r_csf"], run_r_csf)
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

    total_steps = len(case_specs) * len(settings) * len(targets) * MODEL_COUNT
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
                    case = prepare_case(case_with_setting, target=target, horizon_quantile=args.horizon_quantile)

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
                            setting_id=_setting_id,
                            target=_target,
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
                        )
                        print(line, flush=True)

                    _evaluate_case_target(
                        case,
                        case_id=case_id,
                        case_slug=case_slug,
                        case_title=case_title,
                        setting=setting,
                        target=target,
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
