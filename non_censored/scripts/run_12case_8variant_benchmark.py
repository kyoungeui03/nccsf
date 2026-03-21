#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(
        str(PROJECT_ROOT / "non_censored" / "scripts" / "run_12case_bestcurve_8variant_benchmark.py"),
        run_name="__main__",
    )
