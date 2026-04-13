from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _ensure_namespace_package(name: str, path: Path) -> None:
    """Register a lightweight namespace package without executing package __init__.

    The main package exports inside the repo are currently noisy for archival
    experiments because some convenience imports have drifted over time. For the
    archive scripts we want direct, pinned imports from the source files
    themselves, so we create minimal namespace packages for only the broken
    levels (`grf`, `grf.methods`, `grf.synthetic`, `grf.benchmarks`) and then
    import the exact submodules we need.
    """

    if name in sys.modules:
        module = sys.modules[name]
        if hasattr(module, "__path__"):
            module.__path__ = [str(path)]
        return

    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def bootstrap_grf_source_imports(project_root: Path) -> Path:
    """Prepare direct source imports for archive experiments."""

    project_root = Path(project_root).resolve()
    python_package_root = project_root / "python-package"
    grf_root = python_package_root / "grf"

    if str(python_package_root) not in sys.path:
        sys.path.insert(0, str(python_package_root))

    _ensure_namespace_package("grf", grf_root)
    _ensure_namespace_package("grf.methods", grf_root / "methods")
    _ensure_namespace_package("grf.synthetic", grf_root / "synthetic")
    _ensure_namespace_package("grf.benchmarks", grf_root / "benchmarks")
    return grf_root


def import_grf_module(project_root: Path, module_name: str):
    """Import a grf source module by exact path, bypassing package re-export noise."""

    bootstrap_grf_source_imports(project_root)
    return importlib.import_module(module_name)

