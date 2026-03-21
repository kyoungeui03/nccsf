#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(
        str(PROJECT_ROOT / "non_censored" / "scripts" / "run_rhc_b2_vs_best_curve.py"),
        run_name="__main__",
    )
