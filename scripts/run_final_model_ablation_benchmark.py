#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks.econml_8variant import (  # noqa: E402
    CASE_SPECS,
    _evaluate_predictions,
    prepare_case,
    render_avg_summary_png,
    render_case_table_png,
    render_top5_png,
    summarize_results,
)
from grf.censored import (  # noqa: E402
    FinalModelCensoredSurvivalForest,
    FinalModelNoPCICensoredSurvivalForest,
    FinalModelRawCensoredSurvivalForest,
)

matplotlib.use("Agg")


ABLATION_SPECS = [
    ("Final Model (PCI)", FinalModelCensoredSurvivalForest),
    ("Final Model (No PCI)", FinalModelNoPCICensoredSurvivalForest),
    ("Final Model (Raw)", FinalModelRawCensoredSurvivalForest),
]
ABLATION_KEY_MAP = {
    "pci": ABLATION_SPECS[0],
    "no_pci": ABLATION_SPECS[1],
    "raw": ABLATION_SPECS[2],
}

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
        description="Run the censored Final Model PCI / no PCI / raw ablation on basic12 or structured14."
    )
    parser.add_argument("--suite", choices=["basic12", "structured14"], default="basic12")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--setting-ids", nargs="*")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p-x", type=int, default=None)
    parser.add_argument("--p-w", type=int, default=1)
    parser.add_argument("--p-z", type=int, default=1)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(ABLATION_KEY_MAP),
        default=None,
        help="Subset of ablation models to run. Default runs pci, no_pci, and raw.",
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


def _resolve_output_dir(args) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    if args.suite == "structured14":
        return (PROJECT_ROOT / "outputs" / "benchmark_structured_14settings_final_model_ablation").resolve()
    return (PROJECT_ROOT / "outputs" / "benchmark_final_model_ablation_12case").resolve()


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
    censor_pct = int(round(100 * float(cfg.target_censor_rate)))
    return (
        f"{base}, n={cfg.n}, p_x={cfg.p_x}, p_w={cfg.p_w}, "
        f"p_z={cfg.p_z}, seed={cfg.seed}, censoring rate={censor_pct}%"
    )


def _evaluate_case_model(case, name: str, model_cls):
    model = model_cls()
    t0 = time.time()
    model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
    preds = model.effect_from_components(case.X, case.W, case.Z).ravel()
    elapsed = time.time() - t0
    return _evaluate_predictions(name, preds, case.true_cate, elapsed, backend=model.__class__.__name__)


def _consolidate_structured_outputs(output_dir: Path) -> None:
    frames: list[pd.DataFrame] = []
    for setting in SETTINGS:
        path = output_dir / _setting_slug(setting) / "results.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    summary_df, top5_df = summarize_results(combined)
    combined.to_csv(output_dir / "all_settings_results.csv", index=False)
    summary_df.to_csv(output_dir / "all_settings_summary.csv", index=False)
    top5_df.to_csv(output_dir / "all_settings_top5.csv", index=False)


def _run_basic12(case_specs: list[dict[str, object]], output_dir: Path, ablation_specs) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_frames = []
    for case_spec in case_specs:
        case = prepare_case(case_spec, target="RMST", horizon_quantile=0.60)
        case_title = _format_case_title(case_spec, case.cfg)
        rows = []
        for name, model_cls in ablation_specs:
            row = _evaluate_case_model(case, name, model_cls)
            row["case_id"] = case_spec["case_id"]
            row["case_slug"] = case_spec["slug"]
            row["case_title"] = case_title
            rows.append(row)

        case_df = pd.DataFrame(rows)
        case_frames.append(case_df)
        case_csv = output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.csv"
        case_png = output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.png"
        case_df.to_csv(case_csv, index=False)
        render_case_table_png(case_df, case_png)

    combined_df = pd.concat(case_frames, ignore_index=True)
    summary_df, top5_df = summarize_results(combined_df)
    combined_df.to_csv(output_dir / "all_12case_final_model_ablation_results.csv", index=False)
    summary_df.to_csv(output_dir / "all_12case_final_model_ablation_summary.csv", index=False)
    top5_df.to_csv(output_dir / "all_12case_final_model_ablation_top5.csv", index=False)
    render_avg_summary_png(summary_df, output_dir / "all_12case_final_model_ablation_summary.png")
    render_top5_png(top5_df, output_dir / "all_12case_final_model_ablation_top5.png")


def _run_structured14(
    settings: list[dict[str, int | str]],
    *,
    case_ids: set[int] | None,
    output_dir: Path,
    skip_existing: bool,
    ablation_specs,
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
            case = prepare_case(case_spec, target="RMST", horizon_quantile=0.60)
            case_title = _format_case_title(case_spec, case.cfg)
            rows = []
            for name, model_cls in ablation_specs:
                row = _evaluate_case_model(case, name, model_cls)
                row["setting_id"] = str(setting["setting_id"])
                row["setting_slug_full"] = _setting_slug(setting)
                row["case_id"] = int(case_spec["case_id"])
                row["case_slug"] = str(case_spec["slug"])
                row["case_title"] = case_title
                row["n"] = int(case.cfg.n)
                row["p_x"] = int(case.cfg.p_x)
                row["p_w"] = int(case.cfg.p_w)
                row["p_z"] = int(case.cfg.p_z)
                rows.append(row)
            case_frames.append(pd.DataFrame(rows))

        results = pd.concat(case_frames, ignore_index=True)
        summary_df, top5_df = summarize_results(results)
        results.to_csv(results_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        top5_df.to_csv(setting_dir / "top5.csv", index=False)
        _consolidate_structured_outputs(output_dir)

    _consolidate_structured_outputs(output_dir)


def main() -> int:
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    selected_case_ids = set(args.case_ids) if args.case_ids else None
    ablation_specs = [ABLATION_KEY_MAP[k] for k in (args.models or ["pci", "no_pci", "raw"])]

    if args.suite == "basic12":
        case_specs = [
            _case_with_overrides(case, n=args.n, p_x=args.p_x, p_w=args.p_w, p_z=args.p_z)
            for case in CASE_SPECS
            if selected_case_ids is None or int(case["case_id"]) in selected_case_ids
        ]
        _run_basic12(case_specs, output_dir, ablation_specs)
        return 0

    selected_setting_ids = set(args.setting_ids) if args.setting_ids else None
    settings = [s for s in SETTINGS if selected_setting_ids is None or s["setting_id"] in selected_setting_ids]
    _run_structured14(
        settings,
        case_ids=selected_case_ids,
        output_dir=output_dir,
        skip_existing=args.skip_existing,
        ablation_specs=ablation_specs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
