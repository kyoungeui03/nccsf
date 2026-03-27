#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks.econml_8variant import CASE_SPECS as SURV_CASE_SPECS  # noqa: E402
from grf.benchmarks.econml_8variant import _evaluate_predictions, prepare_case  # noqa: E402
from grf.methods import (  # noqa: E402
    OldC3AugmentedSurvDMLCensoredSurvivalForest,
    OldC3AugmentedSurvObservedDMLCensoredSurvivalForest,
    OldC3SummarySurvDMLCensoredSurvivalForest,
    OldC3SummarySurvObservedDMLCensoredSurvivalForest,
)
from grf.non_censored.benchmarks import CASE_SPECS as NC_CASE_SPECS  # noqa: E402
from grf.non_censored.benchmarks import _build_case, _make_cfg, _metric_row  # noqa: E402
from grf.non_censored.models import (  # noqa: E402
    OldC3AugmentedDMLNCCausalForest,
    OldC3AugmentedObservedDMLNCCausalForest,
    OldC3SummaryDMLNCCausalForest,
    OldC3SummaryObservedDMLNCCausalForest,
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


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

NC_MODEL_SPECS = [
    ("Summary-DML (PCI)", OldC3SummaryDMLNCCausalForest),
    ("Summary-DML (no PCI)", OldC3SummaryObservedDMLNCCausalForest),
    ("Augmented-DML (PCI)", OldC3AugmentedDMLNCCausalForest),
    ("Augmented-DML (no PCI)", OldC3AugmentedObservedDMLNCCausalForest),
]

SURV_MODEL_SPECS = [
    ("SummarySurv-DML (PCI)", OldC3SummarySurvDMLCensoredSurvivalForest),
    ("SummarySurv-DML (no PCI)", OldC3SummarySurvObservedDMLCensoredSurvivalForest),
    ("AugmentedSurv-DML (PCI)", OldC3AugmentedSurvDMLCensoredSurvivalForest),
    ("AugmentedSurv-DML (no PCI)", OldC3AugmentedSurvObservedDMLCensoredSurvivalForest),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run structured 14-setting benchmarks for the 8 DML simplification candidates.")
    parser.add_argument(
        "--domains",
        nargs="*",
        choices=["non_censored", "censored"],
        default=["non_censored", "censored"],
    )
    parser.add_argument("--setting-ids", nargs="*")
    parser.add_argument(
        "--nc-output-dir",
        type=Path,
        default=PROJECT_ROOT / "non_censored" / "outputs" / "benchmark_structured_14settings_dml_candidates",
    )
    parser.add_argument(
        "--surv-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benchmark_structured_14settings_dml_candidates",
    )
    parser.add_argument("--target", choices=["RMST"], default="RMST")
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
            avg_time=("time_sec", "mean") if "time_sec" in df.columns else ("total_time", "mean"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def _nc_case_with_overrides(case_spec, *, n, p_x, p_w, p_z):
    cfg = _make_cfg(case_spec)
    cfg.n = int(n)
    cfg.p_x = int(p_x)
    cfg.p_w = int(p_w)
    cfg.p_z = int(p_z)
    return cfg, _build_case(cfg, case_spec)


def _surv_case_with_overrides(case_spec, *, n, p_x, p_w, p_z, target, horizon_quantile):
    case_copy = dict(case_spec)
    cfg_updates = dict(case_spec["cfg"])
    cfg_updates.update({"n": int(n), "p_x": int(p_x), "p_w": int(p_w), "p_z": int(p_z)})
    case_copy["cfg"] = cfg_updates
    return prepare_case(case_copy, target=target, horizon_quantile=horizon_quantile)


def _evaluate_nc_candidate(case, *, name: str, model_cls) -> dict[str, object]:
    start = time.time()
    model = model_cls()
    model.fit_components(case["X"], case["A"], case["Y"], case["Z"], case["W"])
    preds = model.effect_from_components(case["X"], case["W"], case["Z"]).ravel()
    return _metric_row(name, preds, case["true_cate"], time.time() - start)


def _evaluate_surv_candidate(case, case_spec, *, name: str, model_cls, target: str) -> dict[str, object]:
    start = time.time()
    model = model_cls(target=target, horizon=None)
    model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
    preds = model.effect_from_components(case.X, case.W, case.Z).ravel()
    row = _evaluate_predictions(name, preds, case.true_cate, time.time() - start, backend=model.__class__.__name__)
    row.update(
        case_id=int(case_spec["case_id"]),
        case_slug=str(case_spec["slug"]),
        case_title=str(case_spec["title"]),
        target=str(target),
        estimand_horizon=float(case.horizon),
        horizon_quantile=None,
        n=int(case.cfg.n),
        p_x=int(case.cfg.p_x),
        p_w=int(case.cfg.p_w),
        p_z=int(case.cfg.p_z),
        seed=int(case.cfg.seed),
        target_censor_rate=float(case.cfg.target_censor_rate),
        actual_censor_rate=float(1.0 - case.delta.mean()),
        linear_treatment=bool(case.cfg.linear_treatment),
        linear_outcome=bool(case.cfg.linear_outcome),
        tau_log_hr=float(case.cfg.tau_log_hr),
    )
    return row


def run_non_censored(settings: list[dict[str, int | str]], output_dir: Path, *, skip_existing: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for setting in settings:
        setting_dir = output_dir / _setting_slug(setting)
        results_path = setting_dir / "results.csv"
        summary_path = setting_dir / "summary.csv"
        if skip_existing and results_path.exists() and summary_path.exists():
            print(f"[non_censored] skip {setting['setting_id']} (existing)", flush=True)
            continue

        print(
            f"[non_censored] {setting['setting_id']} "
            f"n={setting['n']} p_x={setting['p_x']} p_w={setting['p_w']} p_z={setting['p_z']}",
            flush=True,
        )
        setting_dir.mkdir(parents=True, exist_ok=True)
        case_frames = []
        for case_spec in NC_CASE_SPECS:
            print(f"  [case {case_spec['case_id']:02d}] {case_spec['slug']}", flush=True)
            cfg, case = _nc_case_with_overrides(
                case_spec,
                n=setting["n"],
                p_x=setting["p_x"],
                p_w=setting["p_w"],
                p_z=setting["p_z"],
            )
            rows = []
            for name, model_cls in NC_MODEL_SPECS:
                rows.append(_evaluate_nc_candidate(case, name=name, model_cls=model_cls))
            case_df = pd.DataFrame(rows)
            case_df.insert(0, "case_id", case_spec["case_id"])
            case_df.insert(1, "case_slug", case_spec["slug"])
            case_df.insert(2, "case_title", case_spec["title"])
            case_df["n"] = cfg.n
            case_df["p_x"] = cfg.p_x
            case_df["p_w"] = cfg.p_w
            case_df["p_z"] = cfg.p_z
            case_frames.append(case_df)

        results = pd.concat(case_frames, ignore_index=True)
        results.to_csv(results_path, index=False)
        _mean_summary(results).to_csv(summary_path, index=False)
        print(f"  [done] saved {setting_dir}", flush=True)


def run_censored(
    settings: list[dict[str, int | str]],
    output_dir: Path,
    *,
    target: str,
    horizon_quantile: float,
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
        for case_spec in SURV_CASE_SPECS:
            print(f"  [case {int(case_spec['case_id']):02d}] {case_spec['slug']}", flush=True)
            case = _surv_case_with_overrides(
                case_spec,
                n=setting["n"],
                p_x=setting["p_x"],
                p_w=setting["p_w"],
                p_z=setting["p_z"],
                target=target,
                horizon_quantile=horizon_quantile,
            )
            rows = []
            for name, model_cls in SURV_MODEL_SPECS:
                rows.append(_evaluate_surv_candidate(case, case_spec, name=name, model_cls=model_cls, target=target))
            case_frames.append(pd.DataFrame(rows))

        results = pd.concat(case_frames, ignore_index=True)
        results.to_csv(results_path, index=False)
        _mean_summary(results).to_csv(summary_path, index=False)
        print(f"  [done] saved {setting_dir}", flush=True)


def main() -> int:
    args = parse_args()
    selected_ids = set(args.setting_ids) if args.setting_ids else None
    settings = [s for s in SETTINGS if selected_ids is None or s["setting_id"] in selected_ids]
    if "non_censored" in args.domains:
        run_non_censored(settings, args.nc_output_dir.resolve(), skip_existing=args.skip_existing)
    if "censored" in args.domains:
        run_censored(
            settings,
            args.surv_output_dir.resolve(),
            target=args.target,
            horizon_quantile=args.horizon_quantile,
            skip_existing=args.skip_existing,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
