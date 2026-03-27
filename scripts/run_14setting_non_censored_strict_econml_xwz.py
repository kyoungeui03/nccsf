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

from grf.non_censored.benchmarks import CASE_SPECS as NC_CASE_SPECS  # noqa: E402
from grf.non_censored.benchmarks import _build_case, _make_cfg, _metric_row  # noqa: E402
from grf.non_censored.models import StrictEconmlXWZNCCausalForest  # noqa: E402


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


MODEL_NAME = "StrictEconML (X+W+Z)"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the strict EconML X+W+Z baseline on the non-censored structured 14-setting benchmark.")
    parser.add_argument("--setting-ids", nargs="*")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "non_censored" / "outputs" / "benchmark_structured_14settings_strict_econml_xwz",
    )
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
            avg_time=("time_sec", "mean"),
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


def _evaluate_candidate(case) -> dict[str, object]:
    start = time.time()
    model = StrictEconmlXWZNCCausalForest()
    model.fit_components(case["X"], case["A"], case["Y"], case["Z"], case["W"])
    preds = model.effect_from_components(case["X"], case["W"], case["Z"]).ravel()
    return _metric_row(MODEL_NAME, preds, case["true_cate"], time.time() - start)


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


def run(settings: list[dict[str, int | str]], output_dir: Path, *, skip_existing: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for setting in settings:
        setting_dir = output_dir / _setting_slug(setting)
        results_path = setting_dir / "results.csv"
        summary_path = setting_dir / "summary.csv"
        if skip_existing and results_path.exists() and summary_path.exists():
            print(f"[strict_econml_xwz] skip {setting['setting_id']} (existing)", flush=True)
            continue

        print(
            f"[strict_econml_xwz] {setting['setting_id']} "
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
            case_df = pd.DataFrame([_evaluate_candidate(case)])
            case_df.insert(0, "setting_id", str(setting["setting_id"]))
            case_df.insert(1, "setting_slug_full", _setting_slug(setting))
            case_df.insert(2, "case_id", case_spec["case_id"])
            case_df.insert(3, "case_slug", case_spec["slug"])
            case_df.insert(4, "case_title", case_spec["title"])
            case_df["n"] = cfg.n
            case_df["p_x"] = cfg.p_x
            case_df["p_w"] = cfg.p_w
            case_df["p_z"] = cfg.p_z
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
    settings = [s for s in SETTINGS if selected_ids is None or s["setting_id"] in selected_ids]
    run(settings, args.output_dir.resolve(), skip_existing=args.skip_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
