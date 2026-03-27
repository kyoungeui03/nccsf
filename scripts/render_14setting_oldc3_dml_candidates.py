#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate structured 14-setting DML candidate results.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Structured benchmark output directory containing per-setting subdirectories.",
    )
    return parser.parse_args()


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
            n_settings=("case_id", lambda s: int(s.count() / 12)),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    setting_results = []
    for setting_dir in sorted(input_dir.iterdir()):
        if not setting_dir.is_dir() or not setting_dir.name.startswith("S"):
            continue
        results_path = setting_dir / "results.csv"
        if results_path.exists():
            setting_results.append(pd.read_csv(results_path))
    if not setting_results:
        raise SystemExit(f"No structured results found in {input_dir}")

    all_results = pd.concat(setting_results, ignore_index=True)
    all_results_path = input_dir / "all_settings_results.csv"
    all_summary_path = input_dir / "all_settings_summary.csv"
    all_results.to_csv(all_results_path, index=False)
    _mean_summary(all_results).to_csv(all_summary_path, index=False)
    print(f"Saved {all_results_path}")
    print(f"Saved {all_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
