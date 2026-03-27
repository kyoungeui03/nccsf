#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks.econml_8variant import CASE_SPECS as SURV_CASE_SPECS  # noqa: E402
from grf.benchmarks.econml_8variant import evaluate_r_csf_variant, prepare_case  # noqa: E402
from grf.methods import (
    EconmlMildShrinkNCSurvivalForest,
    EconmlMildShrinkObservedSurvivalForestMatched,
)  # noqa: E402
from grf.non_censored.benchmarks import CASE_SPECS as NC_CASE_SPECS  # noqa: E402
from grf.non_censored.benchmarks import _build_case, _make_cfg, _metric_row  # noqa: E402
from grf.non_censored.models import MildShrinkNCCausalForestDML  # noqa: E402


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


NC_BESTCURVE = _load_module(
    "nc_bestcurve_runner",
    PROJECT_ROOT / "non_censored" / "scripts" / "run_12case_bestcurve_8variant_benchmark.py",
)
SURV_BESTCURVE = _load_module(
    "surv_bestcurve_runner",
    PROJECT_ROOT / "scripts" / "run_12case_bestcurve_8variant_benchmark.py",
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

OLD_C3_NC_NAME = "Old C3  Legacy NC-CSF"
OLD_E2_NC_NAME = "Old E2  Legacy no-PCI baseline"
OLD_C3_SURV_NAME = "Old C3  Legacy NC-CSF"
OLD_E2_SURV_NAME = "Old E2  Legacy no-PCI baseline"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a 14-setting structured benchmark suite with New C3 8-variant benchmark plus Old C3 and Old E2."
    )
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
        default=PROJECT_ROOT / "non_censored" / "outputs" / "benchmark_structured_14settings_extended",
    )
    parser.add_argument(
        "--surv-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benchmark_structured_14settings_extended",
    )
    parser.add_argument(
        "--combined-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benchmark_structured_14settings_combined",
    )
    parser.add_argument("--target", choices=["RMST"], default="RMST")
    parser.add_argument("--horizon-quantile", type=float, default=0.60)
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


def _beats_b2_casewise(df: pd.DataFrame, b2_name: str) -> pd.DataFrame:
    rows = []
    for case_id in sorted(df["case_id"].unique()):
        case_rows = df[df["case_id"] == case_id].set_index("name")
        b2_rmse = float(case_rows.loc[b2_name, "rmse"])
        b2_mae = float(case_rows.loc[b2_name, "mae"])
        for name in case_rows.index:
            if name == b2_name:
                continue
            rows.append(
                {
                    "case_id": int(case_id),
                    "name": name,
                    "beats_b2_rmse": int(float(case_rows.loc[name, "rmse"]) < b2_rmse),
                    "beats_b2_mae": int(float(case_rows.loc[name, "mae"]) < b2_mae),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_beats_b2(df: pd.DataFrame, b2_name: str) -> pd.DataFrame:
    beats = _beats_b2_casewise(df, b2_name)
    if beats.empty:
        return beats
    return (
        beats.groupby("name", as_index=False)
        .agg(rmse_wins_vs_b2=("beats_b2_rmse", "sum"), mae_wins_vs_b2=("beats_b2_mae", "sum"))
        .sort_values(["rmse_wins_vs_b2", "mae_wins_vs_b2", "name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def _nc_case_with_overrides(case_spec, *, n, p_x, p_w, p_z):
    cfg = _make_cfg(case_spec)
    cfg.n = int(n)
    cfg.p_x = int(p_x)
    cfg.p_w = int(p_w)
    cfg.p_z = int(p_z)
    return cfg, _build_case(cfg, case_spec)


def _surv_case_with_overrides(case_spec, *, n, p_x, p_w, p_z):
    case_copy = dict(case_spec)
    cfg_updates = dict(case_spec["cfg"])
    cfg_updates.update({"n": int(n), "p_x": int(p_x), "p_w": int(p_w), "p_z": int(p_z)})
    case_copy["cfg"] = cfg_updates
    return case_copy


def _evaluate_old_c3_nc(case):
    x_final = MildShrinkNCCausalForestDML.stack_final_features(case["X"], case["W"], case["Z"])
    start = time.time()
    model = MildShrinkNCCausalForestDML(
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=42,
        x_core_dim=case["X"].shape[1],
    )
    model.fit_nc(x_final, case["A"], case["Y"], case["Z"], case["W"])
    preds = model.effect(x_final).ravel()
    return _metric_row(OLD_C3_NC_NAME, preds, case["true_cate"], time.time() - start)


def _evaluate_old_e2_nc(case):
    x_final = MildShrinkNCCausalForestDML.stack_final_features(case["X"], case["W"], case["Z"])
    z_dummy = np.zeros_like(case["Z"], dtype=float)
    w_dummy = np.zeros_like(case["W"], dtype=float)
    start = time.time()
    model = MildShrinkNCCausalForestDML(
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=42,
        x_core_dim=case["X"].shape[1],
    )
    model.fit_nc(x_final, case["A"], case["Y"], z_dummy, w_dummy)
    preds = model.effect(x_final).ravel()
    return _metric_row(OLD_E2_NC_NAME, preds, case["true_cate"], time.time() - start)


def _evaluate_old_c3_surv(case, *, target):
    model = EconmlMildShrinkNCSurvivalForest(target=target, horizon=case.horizon if target != "RMST" else None)
    start = time.time()
    model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
    preds = model.effect_from_components(case.X, case.W, case.Z).ravel()
    elapsed = time.time() - start
    return SURV_BESTCURVE._evaluate_predictions(
        OLD_C3_SURV_NAME,
        preds,
        case.true_cate,
        elapsed,
        backend=model.__class__.__name__,
    )


def _evaluate_old_e2_surv(case, *, target):
    model = EconmlMildShrinkObservedSurvivalForestMatched(
        target=target,
        horizon=case.horizon if target != "RMST" else None,
    )
    start = time.time()
    model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
    preds = model.effect_from_components(case.X, case.W, case.Z).ravel()
    elapsed = time.time() - start
    return SURV_BESTCURVE._evaluate_predictions(
        OLD_E2_SURV_NAME,
        preds,
        case.true_cate,
        elapsed,
        backend=model.__class__.__name__,
    )


def run_non_censored(settings: list[dict[str, int | str]], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []

    for setting in settings:
        print(
            f"[non_censored] {setting['setting_id']} "
            f"n={setting['n']} p_x={setting['p_x']} p_w={setting['p_w']} p_z={setting['p_z']}",
            flush=True,
        )
        setting_dir = output_dir / _setting_slug(setting)
        setting_dir.mkdir(parents=True, exist_ok=True)
        case_frames = []

        for case_spec in NC_CASE_SPECS:
            print(
                f"  [case {case_spec['case_id']:02d}] {case_spec['slug']}",
                flush=True,
            )
            cfg, case = _nc_case_with_overrides(
                case_spec,
                n=setting["n"],
                p_x=setting["p_x"],
                p_w=setting["p_w"],
                p_z=setting["p_z"],
            )
            rows = []
            for spec in NC_BESTCURVE.VARIANT_SPECS:
                if spec["kind"] == "baseline_xwz":
                    rows.append(NC_BESTCURVE._evaluate_b2(case))
                elif spec["kind"] == "oracle":
                    rows.append(NC_BESTCURVE._evaluate_oracle(case, spec))
                elif spec["kind"] == "nc":
                    rows.append(NC_BESTCURVE._evaluate_nc(case, spec))
                elif spec["kind"] == "d2":
                    rows.append(NC_BESTCURVE._evaluate_d2(case))
                else:
                    raise ValueError(f"Unknown non-censored variant kind: {spec['kind']}")

            rows.append(_evaluate_old_c3_nc(case))
            rows.append(_evaluate_old_e2_nc(case))

            case_df = pd.DataFrame(rows)
            case_df.insert(0, "case_id", case_spec["case_id"])
            case_df.insert(1, "case_slug", case_spec["slug"])
            case_df.insert(2, "setting_id", setting["setting_id"])
            case_df.insert(3, "domain", "non_censored")
            case_df["n"] = int(setting["n"])
            case_df["p_x"] = int(setting["p_x"])
            case_df["p_w"] = int(setting["p_w"])
            case_df["p_z"] = int(setting["p_z"])
            case_frames.append(case_df)

        setting_results = pd.concat(case_frames, ignore_index=True)
        setting_results.to_csv(setting_dir / "results.csv", index=False)
        setting_summary = _mean_summary(setting_results)
        setting_summary.to_csv(setting_dir / "summary.csv", index=False)
        _aggregate_beats_b2(setting_results, "EconML Baseline").to_csv(
            setting_dir / "wins_vs_b2.csv",
            index=False,
        )
        print(f"  [done] saved {setting_dir}", flush=True)
        all_frames.append(setting_results)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(output_dir / "all_settings_results.csv", index=False)
    overall = (
        combined.groupby(["setting_id", "domain", "n", "p_x", "p_w", "p_z", "name"], as_index=False)
        .agg(
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
        )
        .sort_values(["setting_id", "avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )
    overall.to_csv(output_dir / "all_settings_summary.csv", index=False)
    return combined


def run_censored(settings: list[dict[str, int | str]], output_dir: Path, *, target: str, horizon_quantile: float) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []

    for setting in settings:
        print(
            f"[censored] {setting['setting_id']} "
            f"n={setting['n']} p_x={setting['p_x']} p_w={setting['p_w']} p_z={setting['p_z']}",
            flush=True,
        )
        setting_dir = output_dir / _setting_slug(setting)
        setting_dir.mkdir(parents=True, exist_ok=True)
        case_frames = []

        for case_spec in SURV_CASE_SPECS:
            print(
                f"  [case {case_spec['case_id']:02d}] {case_spec['slug']}",
                flush=True,
            )
            case_spec_override = _surv_case_with_overrides(
                case_spec,
                n=setting["n"],
                p_x=setting["p_x"],
                p_w=setting["p_w"],
                p_z=setting["p_z"],
            )
            case = prepare_case(case_spec_override, target=target, horizon_quantile=horizon_quantile)

            rows = []
            for name, kind, opts in SURV_BESTCURVE.VARIANT_SPECS:
                if kind == "b2":
                    feature_cols = case.x_cols + case.w_cols + case.z_cols
                    rows.append(
                        evaluate_r_csf_variant(
                            name,
                            case.obs_df,
                            feature_cols,
                            case.true_cate,
                            case.horizon,
                            num_trees=200,
                            target=target,
                        )
                    )
                elif kind == "oracle":
                    rows.append(SURV_BESTCURVE._evaluate_oracle_variant(name, case, true_surv=opts["true_surv"], true_qh=opts["true_qh"], target=target))
                elif kind == "nc":
                    rows.append(SURV_BESTCURVE._evaluate_nc_variant(name, case, true_surv=opts["true_surv"], true_qh=opts["true_qh"], target=target))
                elif kind == "d2":
                    rows.append(SURV_BESTCURVE._evaluate_d2_variant(name, case, target=target))
                else:
                    raise ValueError(f"Unknown censored variant kind: {kind}")

            rows.append(_evaluate_old_c3_surv(case, target=target))
            rows.append(_evaluate_old_e2_surv(case, target=target))

            case_df = pd.DataFrame(rows)
            case_df.insert(0, "case_id", int(case_spec_override["case_id"]))
            case_df.insert(1, "case_slug", str(case_spec_override["slug"]))
            case_df.insert(2, "setting_id", setting["setting_id"])
            case_df.insert(3, "domain", "censored")
            case_df["n"] = int(setting["n"])
            case_df["p_x"] = int(setting["p_x"])
            case_df["p_w"] = int(setting["p_w"])
            case_df["p_z"] = int(setting["p_z"])
            case_frames.append(case_df)

        setting_results = pd.concat(case_frames, ignore_index=True)
        setting_results.to_csv(setting_dir / "results.csv", index=False)
        setting_summary = _mean_summary(setting_results.rename(columns={"total_time": "time_sec"}))
        setting_summary.to_csv(setting_dir / "summary.csv", index=False)
        _aggregate_beats_b2(setting_results, "R-CSF Baseline").to_csv(
            setting_dir / "wins_vs_b2.csv",
            index=False,
        )
        print(f"  [done] saved {setting_dir}", flush=True)
        all_frames.append(setting_results)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(output_dir / "all_settings_results.csv", index=False)
    overall = (
        combined.groupby(["setting_id", "domain", "n", "p_x", "p_w", "p_z", "name"], as_index=False)
        .agg(
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
        )
        .sort_values(["setting_id", "avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )
    overall.to_csv(output_dir / "all_settings_summary.csv", index=False)
    return combined


def write_combined_outputs(
    combined_output_dir: Path,
    nc_df: pd.DataFrame | None,
    surv_df: pd.DataFrame | None,
) -> None:
    combined_output_dir.mkdir(parents=True, exist_ok=True)
    frames = [df for df in [nc_df, surv_df] if df is not None and not df.empty]
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(combined_output_dir / "all_domains_all_settings_results.csv", index=False)
    summary = (
        combined.groupby(["domain", "setting_id", "n", "p_x", "p_w", "p_z", "name"], as_index=False)
        .agg(
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
        )
        .sort_values(["domain", "setting_id", "avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, True, True, False])
        .reset_index(drop=True)
    )
    summary.to_csv(combined_output_dir / "all_domains_all_settings_summary.csv", index=False)

    b2_rows = []
    for (domain, setting_id), group in combined.groupby(["domain", "setting_id"]):
        b2_name = "EconML Baseline" if domain == "non_censored" else "R-CSF Baseline"
        beats = _aggregate_beats_b2(group, b2_name)
        if beats.empty:
            continue
        setting_meta = group.iloc[0][["n", "p_x", "p_w", "p_z"]].to_dict()
        for _, row in beats.iterrows():
            b2_rows.append(
                {
                    "domain": domain,
                    "setting_id": setting_id,
                    **setting_meta,
                    "name": row["name"],
                    "rmse_wins_vs_b2": int(row["rmse_wins_vs_b2"]),
                    "mae_wins_vs_b2": int(row["mae_wins_vs_b2"]),
                }
            )
    pd.DataFrame(b2_rows).to_csv(combined_output_dir / "wins_vs_b2_all_domains.csv", index=False)


def main() -> int:
    args = parse_args()
    selected_ids = set(args.setting_ids) if args.setting_ids else None
    settings = [s for s in SETTINGS if selected_ids is None or s["setting_id"] in selected_ids]

    nc_df = None
    surv_df = None

    if "non_censored" in args.domains:
        print(f"[non_censored] running {len(settings)} settings")
        nc_df = run_non_censored(settings, args.nc_output_dir.resolve())
    if "censored" in args.domains:
        print(f"[censored] running {len(settings)} settings")
        surv_df = run_censored(settings, args.surv_output_dir.resolve(), target=args.target, horizon_quantile=args.horizon_quantile)

    write_combined_outputs(args.combined_output_dir.resolve(), nc_df, surv_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
