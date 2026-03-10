from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

import json

from grf.experiments import run_reference_comparison  # noqa: E402


def main() -> int:
    metrics = run_reference_comparison(PROJECT_ROOT)
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
