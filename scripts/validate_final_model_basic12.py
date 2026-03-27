#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PATH = PROJECT_ROOT / "docs" / "FINAL_MODEL_EXPECTED_BASIC12.json"
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_final_model_bundle_benchmark.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical basic12 bundle and verify final-model metrics against the pre-cleanup expectation."
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=PROJECT_ROOT / ".mmenv311" / "bin" / "python",
        help="Python interpreter used to run the canonical bundle benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "validation_postcleanup_final_model_bundle_basic12",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Absolute tolerance for metric equality checks.",
    )
    return parser.parse_args()


def _load_expected() -> dict:
    with EXPECTED_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _compare_summary(actual: pd.DataFrame, expected_rows: list[dict[str, float]], *, tolerance: float, domain: str) -> None:
    expected_by_name = {row["name"]: row for row in expected_rows}
    actual_by_name = {row["name"]: row for row in actual.to_dict("records")}
    if set(actual_by_name) != set(expected_by_name):
        raise AssertionError(
            f"{domain}: model name mismatch. actual={sorted(actual_by_name)} expected={sorted(expected_by_name)}"
        )
    for name, expected in expected_by_name.items():
        actual_row = actual_by_name[name]
        for metric_col, expected_value in expected.items():
            if metric_col == "name":
                continue
            actual_value = float(actual_row[metric_col])
            if abs(actual_value - float(expected_value)) > tolerance:
                raise AssertionError(
                    f"{domain}: {name} mismatch for {metric_col}. actual={actual_value} expected={expected_value}"
                )


def main() -> int:
    args = parse_args()
    expected = _load_expected()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.python),
        str(RUNNER_PATH),
        "--dataset",
        "basic12",
        "--domain",
        "both",
        "--output-dir",
        str(args.output_dir),
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Bundle validation run failed.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    nc_summary = pd.read_csv(args.output_dir / "non_censored" / "summary_full.csv")
    c_summary = pd.read_csv(args.output_dir / "censored" / "summary_full.csv")
    _compare_summary(nc_summary, expected["non_censored"], tolerance=args.tolerance, domain="non_censored")
    _compare_summary(c_summary, expected["censored"], tolerance=args.tolerance, domain="censored")
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
