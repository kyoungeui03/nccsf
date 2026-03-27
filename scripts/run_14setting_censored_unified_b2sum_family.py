#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks.econml_8variant import (  # noqa: E402
    CASE_SPECS,
    _evaluate_predictions,
    evaluate_r_csf_variant,
    prepare_case,
)
from grf.censored import (  # noqa: E402
    UnifiedB2SumBaselineCensoredSurvivalForest,
    UnifiedB2SumMildShrinkCensoredSurvivalForest,
)


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

MODEL_SPECS = [
    {
        "name": "UnifiedB2SumBaseline (Cens)",
        "kind": "python",
        "cls": UnifiedB2SumBaselineCensoredSurvivalForest,
    },
    {
        "name": "UnifiedB2SumMildShrink (Cens)",
        "kind": "python",
        "cls": UnifiedB2SumMildShrinkCensoredSurvivalForest,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the structured 14-setting censored benchmark for the unified B2Sum family."
    )
    parser.add_argument("--setting-ids", nargs="*")
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benchmark_structured_14settings_unified_b2sum_final5",
    )
    parser.add_argument("--target", choices=["RMST", "survival.probability"], default="RMST")
    parser.add_argument("--horizon-quantile", type=float, default=0.60)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


def _setting_slug(setting: dict[str, int | str]) -> str:
    return (
        f"{setting['setting_id']}_n{setting['n']}_px{setting['p_x']}"
        f"_pw{setting['p_w']}_pz{setting['p_z']}"
    )


def _mean_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("name", as_index=False)
        .agg(
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
            avg_sign_acc=("sign_acc", "mean"),
            avg_time=("total_time", "mean"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def _surv_case_with_overrides(case_spec, *, n, p_x, p_w, p_z):
    case_copy = dict(case_spec)
    cfg_updates = dict(case_spec["cfg"])
    cfg_updates.update({"n": int(n), "p_x": int(p_x), "p_w": int(p_w), "p_z": int(p_z)})
    case_copy["cfg"] = cfg_updates
    return case_copy


def _evaluate_candidate(case, *, spec: dict[str, object], target: str) -> dict[str, object]:
    name = str(spec["name"])
    if spec["kind"] == "b2":
        feature_cols = case.x_cols + case.w_cols + case.z_cols
        return evaluate_r_csf_variant(
            name,
            case.obs_df,
            feature_cols,
            case.true_cate,
            case.horizon,
            num_trees=200,
            target=target,
        )

    start = time.time()
    horizon = None if target == "RMST" else float(case.horizon)
    model = spec["cls"](
        target=target,
        horizon=horizon,
        surv_scalar_mode="pair",
        censoring_estimator="nelson-aalen",
    )
    model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
    preds = model.effect_from_components(case.X, case.W, case.Z).ravel()
    return _evaluate_predictions(name, preds, case.true_cate, time.time() - start, backend=model.__class__.__name__)


def _consolidate_outputs(output_dir: Path) -> None:
    frames: list[pd.DataFrame] = []
    for setting in SETTINGS:
        path = output_dir / _setting_slug(setting) / "results.csv"
        if not path.exists():
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        return
    all_results = pd.concat(frames, ignore_index=True)
    all_results.to_csv(output_dir / "all_settings_results.csv", index=False)
    _mean_summary(all_results).to_csv(output_dir / "all_settings_summary.csv", index=False)


def run(
    settings: list[dict[str, int | str]],
    *,
    output_dir: Path,
    target: str,
    horizon_quantile: float,
    case_ids: set[int] | None,
    skip_existing: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for setting in settings:
        setting_dir = output_dir / _setting_slug(setting)
        results_path = setting_dir / "results.csv"
        summary_path = setting_dir / "summary.csv"
        if skip_existing and results_path.exists() and summary_path.exists():
            print(f"[censored] skip {setting['setting_id']} (existing)", flush=True)
            continue

        print(
            f"[censored] {setting['setting_id']} "
            f"n={setting['n']} p_x={setting['p_x']} p_w={setting['p_w']} p_z={setting['p_z']}",
            flush=True,
        )
        setting_dir.mkdir(parents=True, exist_ok=True)
        case_frames = []
        for case_spec in CASE_SPECS:
            if case_ids is not None and int(case_spec["case_id"]) not in case_ids:
                continue
            print(f"  [case {int(case_spec['case_id']):02d}] {case_spec['slug']}", flush=True)
            case_spec_override = _surv_case_with_overrides(
                case_spec,
                n=setting["n"],
                p_x=setting["p_x"],
                p_w=setting["p_w"],
                p_z=setting["p_z"],
            )
            case = prepare_case(case_spec_override, target=target, horizon_quantile=horizon_quantile)

            rows = []
            for spec in MODEL_SPECS:
                rows.append(_evaluate_candidate(case, spec=spec, target=target))

            case_df = pd.DataFrame(rows)
            case_df.insert(0, "setting_id", str(setting["setting_id"]))
            case_df.insert(1, "setting_slug_full", _setting_slug(setting))
            case_df.insert(2, "case_id", int(case_spec_override["case_id"]))
            case_df.insert(3, "case_slug", str(case_spec_override["slug"]))
            case_df.insert(4, "case_title", str(case_spec_override["title"]))
            case_df["n"] = int(setting["n"])
            case_df["p_x"] = int(setting["p_x"])
            case_df["p_w"] = int(setting["p_w"])
            case_df["p_z"] = int(setting["p_z"])
            case_frames.append(case_df)

        results = pd.concat(case_frames, ignore_index=True)
        results.to_csv(results_path, index=False)
        _mean_summary(results).to_csv(summary_path, index=False)
        _consolidate_outputs(output_dir)
        print(f"  [done] saved {setting_dir}", flush=True)

    _consolidate_outputs(output_dir)


def main() -> int:
    args = parse_args()
    selected_ids = set(args.setting_ids) if args.setting_ids else None
    selected_case_ids = set(args.case_ids) if args.case_ids else None
    settings = [s for s in SETTINGS if selected_ids is None or s["setting_id"] in selected_ids]
    run(
        settings,
        output_dir=args.output_dir.resolve(),
        target=args.target,
        horizon_quantile=args.horizon_quantile,
        case_ids=selected_case_ids,
        skip_existing=args.skip_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
