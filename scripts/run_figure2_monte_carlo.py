#!/usr/bin/env python3
"""Paired, resumable Monte Carlo experiment for the Figure 2 benchmark.

The original Figure 2 inputs contain one generated dataset for each
case/setting/target cell.  This runner repeats those cells with independent
data-generating seeds while evaluating every method on the same dataset within
a replication.  It writes replication-level results, Monte Carlo uncertainty
summaries, paired method contrasts, and uncertainty plots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Prevent nested BLAS/OpenMP pools from multiplying worker-level parallelism.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
for _path in (PROJECT_ROOT, PYTHON_PACKAGE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


DEFAULT_MODELS = (
    "final_conditional",
    "final_conditional_oracle",
    "censored_baseline",
)
METRICS = (
    "rmse",
    "mae",
    "bias",
    "pearson_correlation",
    "sign_precision",
    "mean_pred",
)
CHECKPOINT_COLUMNS = (
    "replication",
    "case_id",
    "setting_id",
    "target",
    "name",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired Monte Carlo replications for the Figure 2 simulation."
    )
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=20260725)
    parser.add_argument("--case-ids", nargs="*", type=int, default=None)
    parser.add_argument("--setting-ids", nargs="*", default=None)
    parser.add_argument(
        "--target",
        choices=["RMST", "survival.probability", "both"],
        default="both",
    )
    parser.add_argument("--rmst-horizon", type=float, default=3.0)
    parser.add_argument("--horizon-quantile", type=float, default=0.90)
    parser.add_argument("--num-trees", type=int, default=200)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=[
            "final_conditional",
            "final_conditional_oracle",
            "censored_baseline",
        ],
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument("--flush-every", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reruns" / "figure2_monte_carlo",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Do not fit models; regenerate summaries from the checkpoint.",
    )
    return parser.parse_args()


def _load_benchmark_definitions():
    from censored import run_5model_benchmark as benchmark

    return benchmark


def _selected_definitions(args: argparse.Namespace):
    benchmark = _load_benchmark_definitions()
    cases = benchmark._selected_case_specs(args.case_ids, None)
    settings = benchmark._selected_settings(args.setting_ids)
    targets = benchmark._resolve_targets(args.target)
    return benchmark, cases, settings, targets


def _seed_pair(
    base_seed: int,
    replication: int,
    case_id: int,
    setting_id: str,
) -> tuple[int, int]:
    setting_number = int("".join(ch for ch in str(setting_id) if ch.isdigit()) or 0)
    sequence = np.random.SeedSequence(
        [int(base_seed), int(replication), int(case_id), setting_number]
    )
    data_seed, learner_seed = sequence.generate_state(2, dtype=np.uint32)
    return int(data_seed), int(learner_seed)


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run one replicated case/setting/target cell in a worker process."""

    started = time.time()
    try:
        benchmark = _load_benchmark_definitions()
        from grf.benchmarks.econml_8variant import prepare_case

        case_spec = dict(task["case_spec"])
        case_spec["cfg"] = dict(case_spec["cfg"])
        setting = dict(task["setting"])
        case_with_setting = benchmark._case_spec_with_setting(case_spec, setting)
        case_with_setting["cfg"] = dict(case_with_setting["cfg"])
        case_with_setting["cfg"]["seed"] = int(task["data_seed"])

        if task["target"] == "RMST":
            case = prepare_case(
                case_with_setting,
                target="RMST",
                horizon=float(task["rmst_horizon"]),
            )
        else:
            case = prepare_case(
                case_with_setting,
                target="survival.probability",
                horizon_quantile=float(task["horizon_quantile"]),
            )

        rows = benchmark._evaluate_case_target(
            case,
            case_id=int(case_spec["case_id"]),
            case_slug=str(case_spec["slug"]),
            case_title=str(case_spec["title"]),
            setting=setting,
            target=str(task["target"]),
            random_state=int(task["learner_seed"]),
            num_trees=int(task["num_trees"]),
            selected_models=list(task["models"]),
            grf_num_trees=int(task["num_trees"]),
            grf_min_node_size=20,
            gpu_config={"requested": "off", "enabled": False, "use_gpu": False},
        )
        for row in rows:
            row["replication"] = int(task["replication"])
            row["data_seed"] = int(task["data_seed"])
            row["learner_seed"] = int(task["learner_seed"])
            row["task_wall_time"] = float(time.time() - started)
        return {"ok": True, "rows": rows}
    except Exception as exc:
        return {
            "ok": False,
            "failure": {
                "replication": int(task["replication"]),
                "case_id": int(task["case_spec"]["case_id"]),
                "setting_id": str(task["setting"]["setting_id"]),
                "target": str(task["target"]),
                "data_seed": int(task["data_seed"]),
                "learner_seed": int(task["learner_seed"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "task_wall_time": float(time.time() - started),
            },
        }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _key_set(frame: pd.DataFrame) -> set[tuple[Any, ...]]:
    if frame.empty or not set(CHECKPOINT_COLUMNS).issubset(frame.columns):
        return set()
    return {
        tuple(row)
        for row in frame.loc[:, CHECKPOINT_COLUMNS].itertuples(index=False, name=None)
    }


def _task_key(task: dict[str, Any], model_name: str) -> tuple[Any, ...]:
    return (
        int(task["replication"]),
        int(task["case_spec"]["case_id"]),
        str(task["setting"]["setting_id"]),
        str(task["target"]),
        str(model_name),
    )


def _model_display_names() -> dict[str, str]:
    return {
        "final_conditional": "Final Conditional",
        "final_conditional_oracle": "Final Conditional Oracle",
        "censored_baseline": "Censored Baseline",
    }


def _build_tasks(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    settings: list[dict[str, Any]],
    targets: list[str],
    completed: set[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    display_names = _model_display_names()
    tasks: list[dict[str, Any]] = []
    for replication in range(1, int(args.replications) + 1):
        for case_spec in cases:
            for setting in settings:
                data_seed, learner_seed = _seed_pair(
                    args.base_seed,
                    replication,
                    int(case_spec["case_id"]),
                    str(setting["setting_id"]),
                )
                for target in targets:
                    task = {
                        "replication": replication,
                        "case_spec": case_spec,
                        "setting": setting,
                        "target": target,
                        "data_seed": data_seed,
                        "learner_seed": learner_seed,
                        "rmst_horizon": args.rmst_horizon,
                        "horizon_quantile": args.horizon_quantile,
                        "num_trees": args.num_trees,
                        "models": list(args.models),
                    }
                    expected = [
                        _task_key(task, display_names[model_key])
                        for model_key in args.models
                    ]
                    if not all(key in completed for key in expected):
                        tasks.append(task)
    return tasks


def _summarize_group(group: pd.DataFrame, metrics: tuple[str, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {"n_reps": int(group["replication"].nunique())}
    for metric in metrics:
        values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
        n = len(values)
        mean = float(np.mean(values)) if n else math.nan
        sd = float(np.std(values, ddof=1)) if n > 1 else math.nan
        mcse = sd / math.sqrt(n) if n > 1 else math.nan
        critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else math.nan
        output[f"{metric}_mean"] = mean
        output[f"{metric}_sd"] = sd
        output[f"{metric}_mcse"] = mcse
        output[f"{metric}_ci_low"] = mean - critical * mcse if n > 1 else math.nan
        output[f"{metric}_ci_high"] = mean + critical * mcse if n > 1 else math.nan
        output[f"{metric}_q025"] = float(np.quantile(values, 0.025)) if n else math.nan
        output[f"{metric}_q975"] = float(np.quantile(values, 0.975)) if n else math.nan
    return output


def _group_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: tuple[str, ...] = METRICS,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouper: Any = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in frame.groupby(grouper, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        record.update(_summarize_group(group, metrics))
        records.append(record)
    return pd.DataFrame(records)


def _paired_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_name = "Final Conditional"
    keys = [
        "replication",
        "case_id",
        "case_slug",
        "setting_id",
        "target",
    ]
    base = frame.loc[frame["name"] == base_name, keys + ["rmse", "mae"]].copy()
    pair_frames: list[pd.DataFrame] = []
    for comparator in sorted(set(frame["name"]) - {base_name}):
        other = frame.loc[
            frame["name"] == comparator,
            keys + ["rmse", "mae"],
        ].copy()
        merged = base.merge(other, on=keys, suffixes=("_final", "_comparator"))
        merged["comparator"] = comparator
        for metric in ("rmse", "mae"):
            merged[f"delta_{metric}"] = (
                merged[f"{metric}_final"] - merged[f"{metric}_comparator"]
            )
        pair_frames.append(merged)
    if not pair_frames:
        return pd.DataFrame(), pd.DataFrame()

    paired = pd.concat(pair_frames, ignore_index=True)
    cell = _group_summary(
        paired,
        ["case_id", "case_slug", "setting_id", "target", "comparator"],
        ("delta_rmse", "delta_mae"),
    )
    suite_rep = (
        paired.groupby(
            ["replication", "setting_id", "target", "comparator"],
            as_index=False,
        )[["delta_rmse", "delta_mae"]]
        .mean()
    )
    suite = _group_summary(
        suite_rep,
        ["setting_id", "target", "comparator"],
        ("delta_rmse", "delta_mae"),
    )
    return cell, suite


def _setting_sort_key(value: str) -> tuple[int, str]:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits), text) if digits else (10**9, text)


def _plot_suite_uncertainty(
    suite_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    colors = {
        "Final Conditional Oracle": "#4e79a7",
        "Final Conditional": "#f28e2b",
        "Censored Baseline": "#59a14f",
        "Revised Conditional": "#59a14f",
    }
    for target in sorted(suite_summary["target"].unique()):
        target_data = suite_summary.loc[suite_summary["target"] == target]
        settings = sorted(
            target_data["setting_id"].astype(str).unique(),
            key=_setting_sort_key,
        )
        x = np.arange(len(settings), dtype=float)
        for metric in ("rmse", "mae"):
            fig, ax = plt.subplots(figsize=(18, 7))
            for name in target_data["name"].drop_duplicates():
                color = colors.get(str(name), "#767676")
                method = (
                    target_data.loc[target_data["name"] == name]
                    .set_index("setting_id")
                    .reindex(settings)
                )
                mean = method[f"{metric}_mean"].to_numpy(float)
                lower = method[f"{metric}_ci_low"].to_numpy(float)
                upper = method[f"{metric}_ci_high"].to_numpy(float)
                yerr = np.vstack([mean - lower, upper - mean])
                ax.errorbar(
                    x,
                    mean,
                    yerr=yerr,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.4,
                    capsize=2.5,
                    label=name,
                    color=color,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(settings)
            ax.set_xlabel("Setting")
            ax.set_ylabel(metric.upper())
            ax.set_title(
                f"Figure 2 Monte Carlo uncertainty: {target} {metric.upper()}"
            )
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.legend(frameon=True)
            fig.tight_layout()
            slug = target.replace(".", "_")
            fig.savefig(
                output_dir / f"figure2_mc_uncertainty_{slug}_{metric}.png",
                dpi=200,
            )
            plt.close(fig)


def generate_summaries(frame: pd.DataFrame, output_dir: Path) -> None:
    if frame.empty:
        raise ValueError("No replication-level results are available to summarize.")

    cell_columns = [
        "case_id",
        "case_slug",
        "setting_id",
        "target",
        "name",
    ]
    cell_summary = _group_summary(frame, cell_columns)
    _atomic_csv(cell_summary, output_dir / "cell_mc_summary.csv")

    suite_rep = (
        frame.groupby(
            ["replication", "setting_id", "target", "name"],
            as_index=False,
        )[list(METRICS)]
        .mean()
    )
    _atomic_csv(suite_rep, output_dir / "suite_replication_means.csv")
    suite_summary = _group_summary(
        suite_rep,
        ["setting_id", "target", "name"],
    )
    _atomic_csv(suite_summary, output_dir / "suite_mc_summary.csv")

    paired_cell, paired_suite = _paired_summaries(frame)
    _atomic_csv(paired_cell, output_dir / "paired_cell_mc_summary.csv")
    _atomic_csv(paired_suite, output_dir / "paired_suite_mc_summary.csv")
    _plot_suite_uncertainty(suite_summary, output_dir)


def main() -> int:
    args = parse_args()
    if args.replications < 2:
        raise ValueError("--replications must be at least 2 for uncertainty summaries.")
    if args.workers < 1:
        raise ValueError("--workers must be positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "replication_results.csv"
    failures_path = output_dir / "failures.csv"
    metadata_path = output_dir / "run_metadata.json"

    _, cases, settings, targets = _selected_definitions(args)
    metadata = {
        "script": THIS_FILE.name,
        "runner_version": 1,
        "replications": int(args.replications),
        "workers_requested": int(args.workers),
        "base_seed": int(args.base_seed),
        "case_ids": [int(case["case_id"]) for case in cases],
        "setting_ids": [str(setting["setting_id"]) for setting in settings],
        "targets": list(targets),
        "models": list(args.models),
        "rmst_horizon": float(args.rmst_horizon),
        "horizon_quantile": float(args.horizon_quantile),
        "num_trees": int(args.num_trees),
        "seed_design": (
            "Independent DGP seed per replication/case/setting; identical DGP "
            "and learner seeds across methods and targets within a matched cell."
        ),
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        comparable_existing.pop("workers_requested", None)
        comparable_new = dict(metadata)
        comparable_new.pop("workers_requested", None)
        if comparable_existing != comparable_new:
            raise ValueError(
                f"Run metadata does not match existing output at {output_dir}. "
                "Choose a new --output-dir."
            )
    else:
        _atomic_json(metadata, metadata_path)

    results = (
        pd.read_csv(checkpoint_path)
        if checkpoint_path.exists()
        else pd.DataFrame()
    )
    failures = (
        pd.read_csv(failures_path)
        if failures_path.exists()
        else pd.DataFrame()
    )

    if args.summarize_only:
        generate_summaries(results, output_dir)
        print(f"Regenerated summaries in output directory {output_dir.name!r}.", flush=True)
        return 0

    completed = _key_set(results)
    tasks = _build_tasks(args, cases, settings, targets, completed)
    total_cells = (
        int(args.replications) * len(cases) * len(settings) * len(targets)
    )
    print(
        f"Expected cells={total_cells}; completed result rows={len(completed)}; "
        f"pending cells={len(tasks)}; workers={args.workers}",
        flush=True,
    )
    if not tasks:
        generate_summaries(results, output_dir)
        print("Monte Carlo run already complete.", flush=True)
        return 0

    buffered_rows: list[dict[str, Any]] = []
    buffered_failures: list[dict[str, Any]] = []
    finished = 0
    started = time.time()

    def flush() -> None:
        nonlocal results, failures, buffered_rows, buffered_failures
        if buffered_rows:
            results = pd.concat(
                [results, pd.DataFrame(buffered_rows)],
                ignore_index=True,
            )
            results = (
                results.drop_duplicates(
                    subset=list(CHECKPOINT_COLUMNS),
                    keep="last",
                )
                .sort_values(list(CHECKPOINT_COLUMNS))
                .reset_index(drop=True)
            )
            _atomic_csv(results, checkpoint_path)
            buffered_rows = []
        if buffered_failures:
            failures = pd.concat(
                [failures, pd.DataFrame(buffered_failures)],
                ignore_index=True,
            )
            failures = failures.drop_duplicates(
                subset=["replication", "case_id", "setting_id", "target"],
                keep="last",
            )
            _atomic_csv(failures, failures_path)
            buffered_failures = []

    try:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            future_map = {executor.submit(_run_task, task): task for task in tasks}
            for future in as_completed(future_map):
                result = future.result()
                finished += 1
                if result["ok"]:
                    buffered_rows.extend(result["rows"])
                    status = "ok"
                else:
                    buffered_failures.append(result["failure"])
                    status = f"FAILED {result['failure']['error_type']}"
                elapsed = time.time() - started
                rate = finished / elapsed if elapsed > 0 else 0.0
                remaining = (len(tasks) - finished) / rate if rate > 0 else math.nan
                task = future_map[future]
                print(
                    f"[{finished}/{len(tasks)}] {status} "
                    f"rep={task['replication']} case={task['case_spec']['case_id']} "
                    f"setting={task['setting']['setting_id']} target={task['target']} "
                    f"ETA={remaining / 3600:.2f}h",
                    flush=True,
                )
                if finished % int(args.flush_every) == 0:
                    flush()
    except KeyboardInterrupt:
        flush()
        print("Interrupted; checkpoint saved.", file=sys.stderr, flush=True)
        return 130

    flush()
    generate_summaries(results, output_dir)

    expected_rows = total_cells * len(args.models)
    successful_rows = len(_key_set(results))
    print(
        f"Finished with {successful_rows}/{expected_rows} expected result rows "
        f"and {len(failures)} failed cells. "
        f"Output directory: {output_dir.name!r}",
        flush=True,
    )
    return 0 if successful_rows == expected_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
