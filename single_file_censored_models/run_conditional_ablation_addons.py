#!/usr/bin/env python3
"""Run the missing conditional-censoring PCI ablation models.

This runner is intentionally narrow: it only evaluates the conditional
final-stage ablations that are not already present in our saved result suites.

Models:
    1. Final Conditional PCI Surv-Only
    2. Final Conditional PCI X-Only
    3. Proper No PCI Conditional

It reuses the same synthetic case grid, target handling, and horizon rule as
the existing benchmark runners so the outputs can be merged with the current
conditional result tables.
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
    from .final_censored_model_conditional import (  # noqa: E402
        FinalModelConditionalPCISurvOnlyCensoredSurvivalForest,
        FinalModelConditionalPCIXOnlyCensoredSurvivalForest,
    )
    from .proper_censored_baseline_conditional import ProperNoPCIConditionalCensoredSurvivalForest  # noqa: E402
    from .run_5model_benchmark import CASE_SPECS, SETTINGS  # noqa: E402
except ImportError:  # pragma: no cover
    from single_file_censored_models.final_censored_model_conditional import (  # type: ignore  # noqa: E402
        FinalModelConditionalPCISurvOnlyCensoredSurvivalForest,
        FinalModelConditionalPCIXOnlyCensoredSurvivalForest,
    )
    from single_file_censored_models.proper_censored_baseline_conditional import (  # type: ignore  # noqa: E402
        ProperNoPCIConditionalCensoredSurvivalForest,
    )
    from single_file_censored_models.run_5model_benchmark import CASE_SPECS, SETTINGS  # type: ignore  # noqa: E402


TARGETS = ["RMST", "survival.probability"]
MODEL_COUNT = 3
RUN_METADATA_FILE = "run_metadata.json"
CHECKPOINT_RESULTS_FILE = "results_incremental.csv"

MODEL_NAMES = {
    "final_conditional_surv_only": "Final Conditional PCI Surv-Only",
    "final_conditional_x_only": "Final Conditional PCI X-Only",
    "proper_conditional": "Proper No PCI Conditional",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int, default=None)
    parser.add_argument("--case-slugs", nargs="*", default=None)
    parser.add_argument("--setting-ids", nargs="*", default=None)
    parser.add_argument("--target", choices=["RMST", "survival.probability", "both"], default="both")
    parser.add_argument("--horizon-quantile", type=float, default=0.90)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-trees", type=int, default=200)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-settings", action="store_true")
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / "single_file_censored_models_conditional_ablation_addons").resolve()


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
    selected = [s for s in SETTINGS if str(s["setting_id"]).upper() in wanted]
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


def _persist_checkpoint(results: pd.DataFrame, *, output_dir: Path) -> None:
    results = _sort_results_frame(results)
    checkpoint_path = _checkpoint_path(output_dir)
    results_full_path = _results_full_path(output_dir)
    results.to_csv(checkpoint_path, index=False)
    results.to_csv(results_full_path, index=False)
    if results.empty:
        return
    for target, target_df in results.groupby("target"):
        summary = (
            target_df.groupby("name", as_index=False)
            .agg(
                avg_rmse=("rmse", "mean"),
                avg_mae=("mae", "mean"),
                avg_pearson=("pearson_correlation", "mean"),
                avg_bias=("bias", "mean"),
                avg_sign_precision=("sign_precision", "mean"),
                avg_total_time=("total_time", "mean"),
            )
            .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
            .reset_index(drop=True)
        )
        target_slug = str(target).replace(".", "_")
        summary.to_csv(output_dir / f"conditional_ablation_addons_{target_slug}_summary.csv", index=False)


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
        row["horizon"] = horizon
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

    def run_surv_only():
        model = FinalModelConditionalPCISurvOnlyCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=random_state,
            surv_scalar_mode="full",
            n_estimators=num_trees,
            censoring_estimator="cox",
        )
        t0 = time.time()
        model.fit_components(x, a, time_obs, event, z, w)
        preds = np.asarray(model.effect_from_components(x, w, z), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES["final_conditional_surv_only"],
            preds,
            case.true_cate,
            time.time() - t0,
            backend=model.__class__.__name__,
        )

    run_or_resume(MODEL_NAMES["final_conditional_surv_only"], run_surv_only)

    def run_x_only():
        model = FinalModelConditionalPCIXOnlyCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=random_state,
            surv_scalar_mode="full",
            n_estimators=num_trees,
            censoring_estimator="cox",
        )
        t0 = time.time()
        model.fit_components(x, a, time_obs, event, z, w)
        preds = np.asarray(model.effect_from_components(x, w, z), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES["final_conditional_x_only"],
            preds,
            case.true_cate,
            time.time() - t0,
            backend=model.__class__.__name__,
        )

    run_or_resume(MODEL_NAMES["final_conditional_x_only"], run_x_only)

    def run_proper_conditional():
        model = ProperNoPCIConditionalCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=random_state,
            n_estimators=num_trees,
            censoring_estimator="cox",
        )
        t0 = time.time()
        model.fit_components(x, a, time_obs, event, z, w)
        preds = np.asarray(model.effect_from_components(x, w, z), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES["proper_conditional"],
            preds,
            case.true_cate,
            time.time() - t0,
            backend=model.__class__.__name__,
        )

    run_or_resume(MODEL_NAMES["proper_conditional"], run_proper_conditional)
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

    case_specs = _selected_case_specs(args.case_ids, args.case_slugs)
    settings = _selected_settings(args.setting_ids)
    targets = _resolve_targets(args.target)
    run_metadata = _make_run_metadata(args=args, case_specs=case_specs, settings=settings, targets=targets)
    checkpoint_results = _validate_or_initialize_resume_state(output_dir=output_dir, metadata=run_metadata)
    checkpoint_results = _sort_results_frame(checkpoint_results)

    completed_lookup: dict[tuple[int, str, str, str], dict[str, object]] = {}
    for row in checkpoint_results.to_dict("records"):
        completed_lookup[
            (int(row["case_id"]), str(row["setting_id"]), str(row["target"]), str(row["name"]))
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
            (int(row["case_id"]), str(row["setting_id"]), str(row["target"]), str(row["name"]))
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

                    def progress_hook(row, *, resumed: bool, _case_id=case_id, _case_slug=case_slug, _target=target):
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
                            print(f"[resume] {line} setting={setting_id}", flush=True)
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
                        print(f"{line} setting={setting_id}", flush=True)

                    _evaluate_case_target(
                        case,
                        case_id=case_id,
                        case_slug=case_slug,
                        case_title=case_title,
                        setting=setting,
                        target=target,
                        random_state=args.random_state,
                        num_trees=args.num_trees,
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
