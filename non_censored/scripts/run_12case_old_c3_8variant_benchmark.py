#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.non_censored import run_all_12case_benchmarks  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the legacy Old C3 non-censored 12-case 8-variant benchmark."
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "benchmark_old_c3_8variant_12case"),
        help="Directory where CSV and PNG outputs will be written.",
    )
    parser.add_argument(
        "--case-ids",
        help="Optional comma-separated subset of case IDs, e.g. 1,3,12",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_ids = None
    if args.case_ids:
        case_ids = [int(part.strip()) for part in args.case_ids.split(",") if part.strip()]
    run_all_12case_benchmarks(output_dir, case_ids=case_ids)


if __name__ == "__main__":
    main()
