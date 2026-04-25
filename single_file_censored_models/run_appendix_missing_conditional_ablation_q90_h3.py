#!/usr/bin/env python3
"""Run the missing conditional ablation models for appendix tables.

This runner evaluates only the four conditional-censoring ablation models that
do not already have results under the standardized appendix horizon rules:

  - RMST at fixed horizon h = 3.0
  - survival.probability at observed follow-up q90

Together with the existing q90 / h=3.0 result suites for

  - Final Conditional Oracle
  - Final Conditional
  - Revised Conditional
  - Final Marginal Oracle
  - Final Marginal
  - Revised Marginal

these four models are sufficient to reproduce the three appendix tables:

  1. Final model ablation
  2. Conditional vs marginal
  3. Proxy-enriched vs proxy-specific
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

for path in (PROJECT_ROOT, PYTHON_PACKAGE_ROOT, MODEL_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from grf.benchmarks.econml_8variant import _evaluate_predictions, prepare_case  # noqa: E402

try:  # pragma: no cover
    from . import final_censored_model as final_mod  # noqa: E402
    from . import final_censored_model_conditional as conditional_mod  # noqa: E402
    from .run_5model_benchmark import CASE_SPECS, SETTINGS  # noqa: E402
except ImportError:  # pragma: no cover
    from single_file_censored_models import final_censored_model as final_mod  # type: ignore  # noqa: E402
    from single_file_censored_models import final_censored_model_conditional as conditional_mod  # type: ignore  # noqa: E402
    from single_file_censored_models.run_5model_benchmark import CASE_SPECS, SETTINGS  # type: ignore  # noqa: E402


TARGETS = ["RMST", "survival.probability"]
MODEL_NAMES = {
    "final_conditional_x_only": "Final Conditional PCI X-Only",
    "final_conditional_broaddup_raw_only": "Final Conditional BroadDup Raw-Only",
    "final_conditional_dup_full": "Final Conditional Dup Full",
    "final_conditional_dup_raw_only": "Final Conditional Dup Raw-Only",
}
MODEL_COUNT = len(MODEL_NAMES)
RUN_METADATA_FILE = "run_metadata.json"
CHECKPOINT_RESULTS_FILE = "results_incremental.csv"

RMST_H4 = {
    "label": "RMST-H4",
    "order": 4,
    "horizon": 3.0,
    "description": "Fixed appendix RMST horizon 3.0",
}


def _install_raw_only_feature_mode() -> None:
    """Patch the local single-file Final Model to support final_feature_mode='raw_only'."""

    supported = tuple(final_mod.SUPPORTED_FINAL_FEATURE_MODES)
    if "raw_only" not in supported:
        final_mod.SUPPORTED_FINAL_FEATURE_MODES = supported + ("raw_only",)

    if getattr(final_mod, "_codex_raw_only_patch_installed", False):
        return

    original_build = final_mod._build_final_features
    ensure_2d = final_mod._ensure_2d

    def _patched_build_final_features(mode, x, raw_w, raw_z, bridge):
        if mode == "raw_only":
            del bridge
            return np.hstack(
                [
                    ensure_2d(x).astype(float),
                    ensure_2d(raw_w).astype(float),
                    ensure_2d(raw_z).astype(float),
                ]
            )
        return original_build(mode, x, raw_w, raw_z, bridge)

    final_mod._build_final_features = _patched_build_final_features
    final_mod._codex_raw_only_patch_installed = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int, default=None)
    parser.add_argument("--case-slugs", nargs="*", default=None)
    parser.add_argument("--setting-ids", nargs="*", default=None)
    parser.add_argument("--target", choices=["RMST", "survival.probability", "both"], default="both")
    parser.add_argument(
        "--horizon-quantile",
        type=float,
        default=0.90,
        help="Observed follow-up quantile for survival.probability (default: q90).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-trees", type=int, default=200)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-settings", action="store_true")
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (PROJECT_ROOT / "outputs" / "appendix_missing_conditional_ablation_q90_h3").resolve()


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
        return [dict(RMST_H4)]
    return [
        {
            "label": f"SURV-Q{int(round(100 * survival_quantile)):02d}",
            "order": int(round(100 * survival_quantile)),
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
        "runner_version": 1,
        "case_ids": [int(spec["case_id"]) for spec in case_specs],
        "case_slugs": [str(spec["slug"]) for spec in case_specs],
        "setting_ids": [str(setting["setting_id"]) for setting in settings],
        "targets": list(targets),
        "horizon_quantile": float(args.horizon_quantile),
        "rmst_horizon": float(RMST_H4["horizon"]),
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
    if results.empty:
        return pd.DataFrame()
    return (
        results.groupby(["target", "horizon_label", "horizon_order", "name"], as_index=False)
        .agg(
            avg_horizon=("horizon", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_bias=("bias", "mean"),
            avg_abs_bias=("bias", lambda values: float(np.mean(np.abs(values)))),
            avg_pearson_correlation=("pearson_correlation", "mean"),
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


def _persist_checkpoint(results: pd.DataFrame, *, output_dir: Path) -> None:
    results = _sort_results_frame(results)
    results.to_csv(_checkpoint_path(output_dir), index=False)
    results.to_csv(_results_full_path(output_dir), index=False)
    summary = summarize_results(results)
    if not summary.empty:
        summary.to_csv(output_dir / "appendix_missing_conditional_ablation_q90_h3_summary.csv", index=False)


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


def _build_variant_model(
    *,
    name_key: str,
    target: str,
    horizon: float,
    random_state: int,
    num_trees: int,
):
    _install_raw_only_feature_mode()

    common_kwargs = {
        "target": target,
        "horizon": horizon,
        "random_state": random_state,
        "surv_scalar_mode": "full",
        "n_estimators": num_trees,
        "censoring_estimator": "cox",
    }

    if name_key == "final_conditional_x_only":
        return conditional_mod.FinalModelConditionalPCIXOnlyCensoredSurvivalForest(**common_kwargs)
    if name_key == "final_conditional_broaddup_raw_only":
        return conditional_mod.FinalModelConditionalCensoredSurvivalForest(
            **common_kwargs,
            nuisance_feature_mode="broad_dup",
            final_feature_mode="raw_only",
        )
    if name_key == "final_conditional_dup_full":
        return conditional_mod.FinalModelConditionalCensoredSurvivalForest(
            **common_kwargs,
            nuisance_feature_mode="dup",
            final_feature_mode="full",
        )
    if name_key == "final_conditional_dup_raw_only":
        return conditional_mod.FinalModelConditionalCensoredSurvivalForest(
            **common_kwargs,
            nuisance_feature_mode="dup",
            final_feature_mode="raw_only",
        )
    raise ValueError(f"Unsupported name_key={name_key!r}")


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
            row["horizon_rule"] = "fixed_rmst_h3"
            row["horizon_quantile"] = np.nan
        else:
            row["horizon_rule"] = f"observed_followup_q{float(horizon_spec['horizon_quantile']):.2f}"
            row["horizon_quantile"] = float(horizon_spec["horizon_quantile"])
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

    def _run_variant(name_key: str):
        model = _build_variant_model(
            name_key=name_key,
            target=target,
            horizon=horizon,
            random_state=random_state,
            num_trees=num_trees,
        )
        t0 = time.time()
        model.fit_components(x, a, time_obs, event, z, w)
        preds = np.asarray(model.effect_from_components(x, w, z), dtype=float).ravel()
        return _evaluate_predictions(
            MODEL_NAMES[name_key],
            preds,
            case.true_cate,
            time.time() - t0,
            backend=model.__class__.__name__,
        )

    for name_key in (
        "final_conditional_x_only",
        "final_conditional_broaddup_raw_only",
        "final_conditional_dup_full",
        "final_conditional_dup_raw_only",
    ):
        run_or_resume(MODEL_NAMES[name_key], lambda _name_key=name_key: _run_variant(_name_key))

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

    if not (0.0 < float(args.horizon_quantile) <= 1.0):
        raise ValueError(f"--horizon-quantile must lie in (0, 1]. Received {args.horizon_quantile}.")
    if int(args.num_trees) % 4 != 0:
        raise ValueError(
            "--num-trees must be divisible by 4 for the EconML causal forest backend "
            f"(received {args.num_trees})."
        )

    case_specs = _selected_case_specs(args.case_ids, args.case_slugs)
    settings = _selected_settings(args.setting_ids)
    targets = _resolve_targets(args.target)
    run_metadata = _make_run_metadata(args=args, case_specs=case_specs, settings=settings, targets=targets)
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

    total_target_horizon_runs = sum(
        len(_target_horizon_specs(target, survival_quantile=float(args.horizon_quantile)))
        for target in targets
    )
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
                    for horizon_spec in _target_horizon_specs(target, survival_quantile=float(args.horizon_quantile)):
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
                            existing_rows_by_name=existing_rows_by_name,
                            save_row_hook=save_row,
                            progress_hook=progress_hook,
                        )
    except KeyboardInterrupt:
        if not checkpoint_results.empty:
            _persist_checkpoint(checkpoint_results, output_dir=output_dir)
        print(
            "\nInterrupted. Completed model rows have been checkpointed. "
            f"Resume by rerunning the same command with --output-dir {output_dir}.",
            flush=True,
        )
        return 130

    _persist_checkpoint(checkpoint_results, output_dir=output_dir)
    print(f"[done] wrote outputs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
