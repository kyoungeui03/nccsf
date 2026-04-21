#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

BENCHMARK_SCRIPT = PROJECT_ROOT / "single_file_censored_models" / "run_5model_benchmark.py"
BOXPLOT_SCRIPT = PROJECT_ROOT / "single_file_censored_models" / "plot_grouped_boxplot.py"
SUBGROUP_SCRIPT = THIS_FILE.parent / "evaluate_subgroup_bias_censored.py"
SUBGROUP_PLOT_SCRIPT = PROJECT_ROOT / "outputs" / "censored" / "plot_grouped_subgroup_bias_boxplots.py"

MODEL_ORDER = [
    "Final Conditional Oracle",
    "Final Conditional",
    "Revised Conditional",
]
TARGET_ORDER = ["RMST", "survival.probability"]
PAIRWISE_METRICS_BENCHMARK = ["rmse", "mae", "abs_bias", "abs_relative_bias"]
PAIRWISE_METRICS_SUBGROUP = ["subgroup_abs_bias", "subgroup_abs_relative_bias"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified runner for the censored slide-reproduction workflow: "
            "3-model benchmark, RMSE/MAE grouped boxplots, subgroup-bias evaluation, "
            "subgroup boxplots, relative-bias summaries, signed-bias cell counting, "
            "and pairwise win-rate/margin tables."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "slide_reproduction_conditional",
        help="Directory that will contain all generated benchmark, subgroup, and diagnostic outputs.",
    )
    parser.add_argument(
        "--benchmark-results",
        type=Path,
        default=None,
        help="Optional existing benchmark results_full.csv. If omitted, the benchmark is run first.",
    )
    parser.add_argument(
        "--target",
        choices=["RMST", "survival.probability", "both"],
        default="both",
        help="Target subset to run.",
    )
    parser.add_argument(
        "--rmst-horizon",
        type=float,
        default=3.0,
        help="Fixed RMST horizon forwarded to the benchmark/subgroup runners.",
    )
    parser.add_argument(
        "--horizon-quantile",
        type=float,
        default=0.90,
        help="Survival-probability horizon quantile passed through to the benchmark/subgroup runners.",
    )
    parser.add_argument(
        "--num-trees",
        type=int,
        default=200,
        help="Number of trees used by the final forest in the three conditional models.",
    )
    parser.add_argument(
        "--gpu",
        choices=["off", "auto", "force"],
        default="off",
        help="GPU mode forwarded to the benchmark runner.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of synthetic case ids.",
    )
    parser.add_argument(
        "--setting-ids",
        nargs="*",
        default=None,
        help="Optional subset of synthetic setting ids such as S01 S02.",
    )
    parser.add_argument(
        "--subgroup-test-n",
        type=int,
        default=20000,
        help="Independent test size used by subgroup-bias evaluation.",
    )
    parser.add_argument(
        "--min-subgroup-size",
        type=int,
        default=200,
        help="Minimum subgroup size retained in subgroup-bias evaluation.",
    )
    parser.add_argument(
        "--x-quantiles",
        type=int,
        default=3,
        help="Number of quantile bins used for X-based subgroup families.",
    )
    parser.add_argument(
        "--tau-quantiles",
        type=int,
        default=5,
        help="Number of quantile bins used for true-CATE subgroup families.",
    )
    parser.add_argument(
        "--x-subgroup-cols",
        nargs="*",
        type=int,
        default=[0, 1],
        help="Indices of X columns used for subgrouping.",
    )
    parser.add_argument(
        "--skip-subgroup",
        action="store_true",
        help="Skip subgroup-bias evaluation and subgroup plots.",
    )
    parser.add_argument(
        "--relative-bias-eps",
        type=float,
        default=1e-8,
        help="Minimum |truth| denominator used when computing relative bias.",
    )
    parser.add_argument(
        "--zero-bias-tol",
        type=float,
        default=1e-10,
        help="Tolerance used to count a bias value as zero in signed-bias summaries.",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    print(message, flush=True)


def _run_python(script: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script), *args]
    _log(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def _extend_flag(args: list[str], flag: str, values) -> None:
    if values:
        args.append(flag)
        args.extend(str(v) for v in values)


def _sort_target_frame(df: pd.DataFrame, *, model_col: str) -> pd.DataFrame:
    out = df.copy()
    if "target" in out.columns:
        out["target"] = pd.Categorical(out["target"], TARGET_ORDER, ordered=True)
    if model_col in out.columns:
        out[model_col] = pd.Categorical(out[model_col], MODEL_ORDER, ordered=True)
    return out.sort_values([c for c in ["target", model_col] if c in out.columns]).reset_index(drop=True)


def _safe_relative(numer: pd.Series, denom: pd.Series, eps: float) -> np.ndarray:
    numer_values = np.asarray(numer, dtype=float)
    denom_values = np.asarray(denom, dtype=float)
    out = np.full_like(numer_values, np.nan, dtype=float)
    mask = np.abs(denom_values) > float(eps)
    out[mask] = numer_values[mask] / denom_values[mask]
    return out


def _filter_models(df: pd.DataFrame, model_col: str) -> pd.DataFrame:
    return df.loc[df[model_col].isin(MODEL_ORDER)].copy()


def _prepare_benchmark_results(args: argparse.Namespace, benchmark_dir: Path) -> Path:
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark_results is None:
        cli_args = [
            "--target",
            str(args.target),
            "--rmst-horizon",
            str(args.rmst_horizon),
            "--horizon-quantile",
            str(args.horizon_quantile),
            "--num-trees",
            str(args.num_trees),
            "--gpu",
            str(args.gpu),
            "--output-dir",
            str(benchmark_dir),
        ]
        _extend_flag(cli_args, "--case-ids", args.case_ids)
        _extend_flag(cli_args, "--setting-ids", args.setting_ids)
        _run_python(BENCHMARK_SCRIPT, cli_args)
        raw_path = benchmark_dir / "results_full.csv"
    else:
        raw_path = args.benchmark_results.resolve()

    raw_df = pd.read_csv(raw_path)
    filtered = _filter_models(raw_df, "name")
    filtered = _sort_target_frame(filtered, model_col="name")
    filtered_path = benchmark_dir / "results_full_three_models.csv"
    filtered.to_csv(filtered_path, index=False)
    _log(f"[done] filtered benchmark results -> {filtered_path}")
    return filtered_path


def _generate_benchmark_boxplots(results_path: Path, benchmark_dir: Path) -> None:
    df = pd.read_csv(results_path)
    targets = [target for target in TARGET_ORDER if target in set(df["target"].astype(str))]
    for target, metric in itertools.product(targets, ["rmse", "mae"]):
        output_png = benchmark_dir / f"grouped_{metric}_boxplot_{target.replace('.', '_')}.png"
        cli_args = [
            "--input-csv",
            str(results_path),
            "--target",
            target,
            "--metric",
            metric,
            "--output-png",
            str(output_png),
            "--models",
            *MODEL_ORDER,
        ]
        _run_python(BOXPLOT_SCRIPT, cli_args)


def _run_subgroup_suite(args: argparse.Namespace, subgroup_dir: Path) -> tuple[Path, Path]:
    subgroup_dir.mkdir(parents=True, exist_ok=True)
    cli_args = [
        "--output-dir",
        str(subgroup_dir),
        "--target",
        str(args.target),
        "--rmst-horizon",
        str(args.rmst_horizon),
        "--horizon-quantile",
        str(args.horizon_quantile),
        "--num-trees",
        str(args.num_trees),
        "--test-n",
        str(args.subgroup_test_n),
        "--min-subgroup-size",
        str(args.min_subgroup_size),
        "--x-quantiles",
        str(args.x_quantiles),
        "--tau-quantiles",
        str(args.tau_quantiles),
        "--models",
        *MODEL_ORDER,
    ]
    _extend_flag(cli_args, "--case-ids", args.case_ids)
    _extend_flag(cli_args, "--setting-ids", args.setting_ids)
    _extend_flag(cli_args, "--x-subgroup-cols", args.x_subgroup_cols)
    _run_python(SUBGROUP_SCRIPT, cli_args)

    detailed_path = subgroup_dir / "subgroup_bias_detailed.csv"
    plot_output = subgroup_dir / "grouped_subgroup_abs_bias_boxplots.png"
    plot_csv = subgroup_dir / "grouped_subgroup_abs_bias_per_case.csv"
    _run_python(
        SUBGROUP_PLOT_SCRIPT,
        [
            "--input",
            str(detailed_path),
            "--output",
            str(plot_output),
            "--csv-output",
            str(plot_csv),
            "--models",
            *MODEL_ORDER,
        ],
    )
    return detailed_path, plot_csv


def _benchmark_relative_tables(results_path: Path, diagnostics_dir: Path, eps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(results_path)
    df["abs_bias"] = df["bias"].abs()
    df["relative_bias"] = _safe_relative(df["bias"], df["mean_true_cate"], eps)
    df["abs_relative_bias"] = np.abs(df["relative_bias"])
    cells_path = diagnostics_dir / "benchmark_relative_bias_cells.csv"
    df.to_csv(cells_path, index=False)

    summary = (
        df.groupby(["target", "name"], as_index=False)
        .agg(
            cells=("bias", "size"),
            mean_signed_bias=("bias", "mean"),
            mean_abs_bias=("abs_bias", "mean"),
            mean_relative_bias=("relative_bias", "mean"),
            median_relative_bias=("relative_bias", "median"),
            mean_abs_relative_bias=("abs_relative_bias", "mean"),
        )
        .rename(columns={"name": "model"})
    )
    summary = _sort_target_frame(summary, model_col="model")
    summary_path = diagnostics_dir / "benchmark_relative_bias_summary.csv"
    summary.to_csv(summary_path, index=False)
    return df, summary


def _subgroup_relative_tables(detailed_path: Path, diagnostics_dir: Path, eps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(detailed_path)
    df = _filter_models(df, "model")
    df["subgroup_abs_bias"] = df["subgroup_bias"].abs()
    df["subgroup_relative_bias"] = _safe_relative(df["subgroup_bias"], df["true_subgroup_ate"], eps)
    df["subgroup_abs_relative_bias"] = np.abs(df["subgroup_relative_bias"])
    cells_path = diagnostics_dir / "subgroup_relative_bias_cells.csv"
    df.to_csv(cells_path, index=False)

    summary = (
        df.groupby(["target", "model"], as_index=False)
        .agg(
            cells=("subgroup_bias", "size"),
            mean_signed_bias=("subgroup_bias", "mean"),
            mean_abs_bias=("subgroup_abs_bias", "mean"),
            mean_relative_bias=("subgroup_relative_bias", "mean"),
            median_relative_bias=("subgroup_relative_bias", "median"),
            mean_abs_relative_bias=("subgroup_abs_relative_bias", "mean"),
        )
    )
    summary = _sort_target_frame(summary, model_col="model")
    summary_path = diagnostics_dir / "subgroup_relative_bias_summary.csv"
    summary.to_csv(summary_path, index=False)
    return df, summary


def _signed_bias_summary(
    df: pd.DataFrame,
    *,
    bias_col: str,
    model_col: str,
    output_path: Path,
    zero_tol: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, model_name), gdf in df.groupby(["target", model_col]):
        bias = pd.to_numeric(gdf[bias_col], errors="coerce").dropna()
        if bias.empty:
            continue
        neg = int((bias < -zero_tol).sum())
        pos = int((bias > zero_tol).sum())
        zero = int((bias.abs() <= zero_tol).sum())
        total = int(len(bias))
        rows.append(
            {
                "target": target,
                "model": model_name,
                "cells": total,
                "negative_cells": neg,
                "positive_cells": pos,
                "zero_cells": zero,
                "negative_pct": 100.0 * neg / total,
                "positive_pct": 100.0 * pos / total,
                "zero_pct": 100.0 * zero / total,
            }
        )
    out = pd.DataFrame(rows)
    out = _sort_target_frame(out, model_col="model")
    out.to_csv(output_path, index=False)
    return out


def _pairwise_summary(
    df: pd.DataFrame,
    *,
    model_col: str,
    key_cols: list[str],
    metrics: list[str],
    output_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    present_models = [model for model in MODEL_ORDER if model in set(df[model_col].astype(str))]
    for target in TARGET_ORDER:
        target_df = df.loc[df["target"] == target].copy()
        if target_df.empty:
            continue
        for metric, model_a, model_b in itertools.product(metrics, present_models, present_models):
            if model_a >= model_b:
                continue
            left = target_df.loc[target_df[model_col] == model_a, key_cols + [metric]].rename(columns={metric: f"{metric}_a"})
            right = target_df.loc[target_df[model_col] == model_b, key_cols + [metric]].rename(columns={metric: f"{metric}_b"})
            merged = left.merge(right, on=key_cols, how="inner")
            if merged.empty:
                continue
            value_a = pd.to_numeric(merged[f"{metric}_a"], errors="coerce")
            value_b = pd.to_numeric(merged[f"{metric}_b"], errors="coerce")
            valid = value_a.notna() & value_b.notna()
            value_a = value_a[valid]
            value_b = value_b[valid]
            if value_a.empty:
                continue
            margin_a = value_b.to_numpy(dtype=float) - value_a.to_numpy(dtype=float)
            win_a = int(np.sum(margin_a > 0))
            win_b = int(np.sum(margin_a < 0))
            tie = int(np.sum(margin_a == 0))
            total = int(len(margin_a))
            rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "cells": total,
                    "win_count_a": win_a,
                    "win_count_b": win_b,
                    "tie_count": tie,
                    "win_rate_a_pct": 100.0 * win_a / total,
                    "win_rate_b_pct": 100.0 * win_b / total,
                    "tie_rate_pct": 100.0 * tie / total,
                    "avg_margin_a": float(np.mean(margin_a)),
                    "median_margin_a": float(np.median(margin_a)),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["target"] = pd.Categorical(out["target"], TARGET_ORDER, ordered=True)
        out["metric"] = pd.Categorical(out["metric"], metrics, ordered=True)
        out = out.sort_values(["target", "metric", "model_a", "model_b"]).reset_index(drop=True)
    out.to_csv(output_path, index=False)
    return out


def _format_value(value) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        magnitude = abs(float(value))
        if magnitude >= 100:
            return f"{float(value):.1f}"
        if magnitude >= 1:
            return f"{float(value):.3f}"
        return f"{float(value):.4f}"
    return str(value)


def _render_table_png(df: pd.DataFrame, output_path: Path, title: str) -> None:
    if df.empty:
        return

    render_df = df.copy()
    wrapped_columns = [textwrap.fill(str(col), width=18) for col in render_df.columns]
    cell_text = [[_format_value(value) for value in row] for row in render_df.itertuples(index=False, name=None)]

    fig_width = max(12.0, 1.45 * len(render_df.columns))
    fig_height = max(2.8, 0.45 * (len(render_df) + 3))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)

    table = ax.table(
        cellText=cell_text,
        colLabels=wrapped_columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.02, 0.98, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#B7C0CC")
        if row == 0:
            cell.set_facecolor("#223A5E")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#F6F8FB" if row % 2 else "#EAF0F8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _render_target_tables(df: pd.DataFrame, stem: str, output_dir: Path) -> None:
    if "target" not in df.columns:
        _render_table_png(df, output_dir / f"{stem}.png", stem.replace("_", " ").title())
        return

    for target in TARGET_ORDER:
        target_df = df.loc[df["target"] == target].copy()
        if target_df.empty:
            continue
        suffix = target.replace(".", "_")
        _render_table_png(
            target_df,
            output_dir / f"{stem}_{suffix}.png",
            f"{stem.replace('_', ' ').title()} ({target})",
        )


def _write_manifest(output_dir: Path, benchmark_dir: Path, subgroup_dir: Path | None, diagnostics_dir: Path) -> None:
    lines = [
        "slide_reproduction_outputs",
        f"benchmark_dir={benchmark_dir.resolve()}",
        f"diagnostics_dir={diagnostics_dir.resolve()}",
    ]
    if subgroup_dir is not None:
        lines.append(f"subgroup_dir={subgroup_dir.resolve()}")
    (output_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    benchmark_dir = output_dir / "benchmark"
    diagnostics_dir = output_dir / "diagnostics"
    subgroup_dir = output_dir / "subgroup_bias"

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    _log("[step 1/4] preparing benchmark results")
    benchmark_results = _prepare_benchmark_results(args, benchmark_dir)

    _log("[step 2/4] generating RMSE/MAE grouped boxplots")
    _generate_benchmark_boxplots(benchmark_results, benchmark_dir)

    _log("[step 3/4] computing benchmark diagnostics")
    benchmark_cells, benchmark_relative = _benchmark_relative_tables(
        benchmark_results,
        diagnostics_dir,
        eps=float(args.relative_bias_eps),
    )
    benchmark_signed = _signed_bias_summary(
        benchmark_cells,
        bias_col="bias",
        model_col="name",
        output_path=diagnostics_dir / "benchmark_signed_bias_summary.csv",
        zero_tol=float(args.zero_bias_tol),
    )
    benchmark_pairwise = _pairwise_summary(
        benchmark_cells,
        model_col="name",
        key_cols=["target", "setting_id", "case_id"],
        metrics=PAIRWISE_METRICS_BENCHMARK,
        output_path=diagnostics_dir / "benchmark_pairwise_win_margin.csv",
    )
    _render_target_tables(benchmark_relative, "benchmark_relative_bias_summary", diagnostics_dir)
    _render_target_tables(benchmark_signed, "benchmark_signed_bias_summary", diagnostics_dir)
    _render_target_tables(benchmark_pairwise, "benchmark_pairwise_win_margin", diagnostics_dir)

    subgroup_dir_for_manifest: Path | None = None
    if not args.skip_subgroup:
        _log("[step 4/4] running subgroup-bias suite")
        subgroup_detailed, _ = _run_subgroup_suite(args, subgroup_dir)
        subgroup_cells, subgroup_relative = _subgroup_relative_tables(
            subgroup_detailed,
            diagnostics_dir,
            eps=float(args.relative_bias_eps),
        )
        subgroup_signed = _signed_bias_summary(
            subgroup_cells,
            bias_col="subgroup_bias",
            model_col="model",
            output_path=diagnostics_dir / "subgroup_signed_bias_summary.csv",
            zero_tol=float(args.zero_bias_tol),
        )
        subgroup_pairwise = _pairwise_summary(
            subgroup_cells,
            model_col="model",
            key_cols=["target", "setting_id", "case_id", "subgroup_family", "subgroup"],
            metrics=PAIRWISE_METRICS_SUBGROUP,
            output_path=diagnostics_dir / "subgroup_pairwise_win_margin.csv",
        )
        _render_target_tables(subgroup_relative, "subgroup_relative_bias_summary", diagnostics_dir)
        _render_target_tables(subgroup_signed, "subgroup_signed_bias_summary", diagnostics_dir)
        _render_target_tables(subgroup_pairwise, "subgroup_pairwise_win_margin", diagnostics_dir)
        subgroup_dir_for_manifest = subgroup_dir
    else:
        _log("[step 4/4] subgroup suite skipped")

    _write_manifest(output_dir, benchmark_dir, subgroup_dir_for_manifest, diagnostics_dir)
    _log(f"[done] all outputs written under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
