#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.non_censored.benchmarks import (  # noqa: E402
    CASE_SPECS,
    SUMMARY_COLUMNS,
    SUMMARY_KEYS,
    TABLE_COLUMNS,
    TABLE_KEYS,
    TOP5_COLUMNS,
    TOP5_KEYS,
    _build_case,
    _make_cfg,
    _metric_row,
    _render_table_png,
)
from grf.non_censored.models import (  # noqa: E402
    FinalModelNCCausalForest,
    FinalModelNoPCINCCausalForest,
    FinalModelRawNCCausalForest,
)

matplotlib.use("Agg")


ABLATION_SPECS = [
    ("Final Model (PCI)", FinalModelNCCausalForest),
    ("Final Model (No PCI)", FinalModelNoPCINCCausalForest),
    ("Final Model (Raw)", FinalModelRawNCCausalForest),
]

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the non-censored Final Model PCI / no PCI / raw ablation on basic12 or structured14."
    )
    parser.add_argument("--suite", choices=["basic12", "structured14"], default="basic12")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--setting-ids", nargs="*")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p-x", type=int, default=None)
    parser.add_argument("--p-w", type=int, default=1)
    parser.add_argument("--p-z", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


def _resolve_output_dir(args) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    if args.suite == "structured14":
        return (PROJECT_ROOT / "non_censored" / "outputs" / "benchmark_structured_14settings_final_model_ablation").resolve()
    return (PROJECT_ROOT / "non_censored" / "outputs" / "benchmark_final_model_ablation_12case").resolve()


def _case_with_overrides(case_spec, *, n=None, p_x=None, p_w=None, p_z=None):
    case_copy = dict(case_spec)
    cfg_updates = dict(case_spec["cfg"])
    if n is not None:
        cfg_updates["n"] = int(n)
    if p_x is not None:
        cfg_updates["p_x"] = int(p_x)
    if p_w is not None:
        cfg_updates["p_w"] = int(p_w)
    if p_z is not None:
        cfg_updates["p_z"] = int(p_z)
    case_copy["cfg"] = cfg_updates
    return case_copy


def _setting_slug(setting: dict[str, int | str]) -> str:
    return (
        f"{setting['setting_id']}_n{setting['n']}_px{setting['p_x']}"
        f"_pw{setting['p_w']}_pz{setting['p_z']}"
    )


def _format_case_title(case_spec, cfg):
    base = str(case_spec["title"]).split(", n=", 1)[0]
    return (
        f"{base}, n={cfg.n}, p_x={cfg.p_x}, p_w={cfg.p_w}, "
        f"p_z={cfg.p_z}, seed={cfg.seed}, censoring rate=0%"
    )


def _summarize(combined_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        combined_df.groupby("name", as_index=False)
        .agg(
            avg_pred_cate=("mean_pred", "mean"),
            avg_true_cate=("mean_true_cate", "mean"),
            avg_acc=("sign_acc", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_pehe=("pehe", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
            avg_time=("time_sec", "mean"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary, summary.head(5).copy()


def _evaluate_ablation(case, name: str, model_cls):
    x = np.asarray(case["X"], dtype=float)
    w = np.asarray(case["W"], dtype=float)
    z = np.asarray(case["Z"], dtype=float)
    a = np.asarray(case["A"], dtype=float)
    y = np.asarray(case["Y"], dtype=float)
    true_cate = np.asarray(case["true_cate"], dtype=float)
    model = model_cls()
    t0 = time.time()
    model.fit_components(x, a, y, z, w)
    preds = model.effect_from_components(x, w, z).ravel()
    return _metric_row(name, preds, true_cate, time.time() - t0)


def _run_basic12(case_specs: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_frames = []
    for case_spec in case_specs:
        cfg = _make_cfg(case_spec)
        case = _build_case(cfg, case_spec)
        case_title = _format_case_title(case_spec, cfg)
        rows = [_evaluate_ablation(case, name, model_cls) for name, model_cls in ABLATION_SPECS]
        case_df = pd.DataFrame(rows)
        case_df.insert(0, "case_id", case_spec["case_id"])
        case_df.insert(1, "case_slug", case_spec["slug"])
        case_df.insert(2, "case_title", case_title)
        case_frames.append(case_df)

        case_csv = output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.csv"
        case_png = output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.png"
        case_df.to_csv(case_csv, index=False)
        _render_table_png(
            case_title,
            case_df.to_dict("records"),
            case_png,
            TABLE_COLUMNS,
            TABLE_KEYS,
            meta="Final Model PCI / no PCI / raw ablation",
        )

    combined_df = pd.concat(case_frames, ignore_index=True)
    summary_df, top5_df = _summarize(combined_df)
    combined_df.to_csv(output_dir / "all_12case_final_model_ablation_results.csv", index=False)
    summary_df.to_csv(output_dir / "all_12case_final_model_ablation_summary.csv", index=False)
    top5_df.to_csv(output_dir / "all_12case_final_model_ablation_top5.csv", index=False)

    _render_table_png(
        "Non-censored Final Model ablation average summary",
        summary_df.to_dict("records"),
        output_dir / "all_12case_final_model_ablation_summary.png",
        SUMMARY_COLUMNS,
        SUMMARY_KEYS,
        dark=True,
        meta="PCI vs no PCI vs raw",
        col_widths=[90, 420, 150, 150, 120, 140, 140, 120, 150, 120],
    )
    _render_table_png(
        "Non-censored Final Model ablation top 5",
        top5_df.to_dict("records"),
        output_dir / "all_12case_final_model_ablation_top5.png",
        TOP5_COLUMNS,
        TOP5_KEYS,
        dark=True,
        meta="Top 5 by avg RMSE",
        col_widths=[90, 440, 140, 140, 140, 160, 120],
    )


def _consolidate_structured_outputs(output_dir: Path) -> None:
    frames: list[pd.DataFrame] = []
    for setting in SETTINGS:
        path = output_dir / _setting_slug(setting) / "results.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    summary_df, top5_df = _summarize(combined)
    combined.to_csv(output_dir / "all_settings_results.csv", index=False)
    summary_df.to_csv(output_dir / "all_settings_summary.csv", index=False)
    top5_df.to_csv(output_dir / "all_settings_top5.csv", index=False)


def _run_structured14(
    settings: list[dict[str, int | str]],
    *,
    case_ids: set[int] | None,
    output_dir: Path,
    skip_existing: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for setting in settings:
        setting_dir = output_dir / _setting_slug(setting)
        results_path = setting_dir / "results.csv"
        summary_path = setting_dir / "summary.csv"
        if skip_existing and results_path.exists() and summary_path.exists():
            continue

        setting_dir.mkdir(parents=True, exist_ok=True)
        case_frames = []
        for raw_case_spec in CASE_SPECS:
            if case_ids is not None and int(raw_case_spec["case_id"]) not in case_ids:
                continue
            case_spec = _case_with_overrides(
                raw_case_spec,
                n=setting["n"],
                p_x=setting["p_x"],
                p_w=setting["p_w"],
                p_z=setting["p_z"],
            )
            cfg = _make_cfg(case_spec)
            case = _build_case(cfg, case_spec)
            case_title = _format_case_title(case_spec, cfg)
            rows = [_evaluate_ablation(case, name, model_cls) for name, model_cls in ABLATION_SPECS]
            case_df = pd.DataFrame(rows)
            case_df.insert(0, "setting_id", str(setting["setting_id"]))
            case_df.insert(1, "setting_slug_full", _setting_slug(setting))
            case_df.insert(2, "case_id", case_spec["case_id"])
            case_df.insert(3, "case_slug", case_spec["slug"])
            case_df.insert(4, "case_title", case_title)
            case_df["n"] = cfg.n
            case_df["p_x"] = cfg.p_x
            case_df["p_w"] = cfg.p_w
            case_df["p_z"] = cfg.p_z
            case_frames.append(case_df)

        results = pd.concat(case_frames, ignore_index=True)
        summary_df, top5_df = _summarize(results)
        results.to_csv(results_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        top5_df.to_csv(setting_dir / "top5.csv", index=False)
        _consolidate_structured_outputs(output_dir)

    _consolidate_structured_outputs(output_dir)


def main() -> int:
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    selected_case_ids = set(args.case_ids) if args.case_ids else None

    if args.suite == "basic12":
        case_specs = [
            _case_with_overrides(case, n=args.n, p_x=args.p_x, p_w=args.p_w, p_z=args.p_z)
            for case in CASE_SPECS
            if selected_case_ids is None or int(case["case_id"]) in selected_case_ids
        ]
        _run_basic12(case_specs, output_dir)
        return 0

    selected_setting_ids = set(args.setting_ids) if args.setting_ids else None
    settings = [s for s in SETTINGS if selected_setting_ids is None or s["setting_id"] in selected_setting_ids]
    _run_structured14(settings, case_ids=selected_case_ids, output_dir=output_dir, skip_existing=args.skip_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
