"""Optional GPU-capable model builders for the single-file benchmark suite.

The censored single-file models rely heavily on sklearn, lifelines, and
EconML. Those stacks are primarily CPU-based, so we expose GPU support only for
the nuisance learners that can realistically be swapped to XGBoost.

This module is intentionally lazy:

- importing it does not require `xgboost`
- callers can ask whether GPU is available
- if `xgboost` is missing, the caller can cleanly fall back to CPU
"""

from __future__ import annotations

import importlib.util
import os
import shutil


def xgboost_available() -> bool:
    """Return True when the `xgboost` Python package is importable."""

    return importlib.util.find_spec("xgboost") is not None


def xgboost_cuda_build_available() -> bool:
    """Return True when the installed XGBoost build appears to support CUDA."""

    if not xgboost_available():
        return False
    try:
        xgb = _import_xgboost()
        build_info = getattr(xgb, "build_info", lambda: {})()
        if isinstance(build_info, dict):
            for key in ("USE_CUDA", "CUDA_SUPPORT", "USE_NCCL"):
                value = build_info.get(key)
                if isinstance(value, str):
                    if value.lower() in {"true", "1", "on", "yes"}:
                        return True
                elif bool(value):
                    return True
    except Exception:
        return False
    return False


def cuda_runtime_likely_available() -> bool:
    """Best-effort CUDA visibility check used by the runner.

    This is intentionally lightweight. The actual training call is the final
    authority, but this signal is good enough for deciding whether the runner
    should *attempt* GPU-backed nuisance models.
    """

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() not in {"", "-1"}:
        return True
    if os.environ.get("NVIDIA_VISIBLE_DEVICES") not in {None, "", "void"}:
        return True
    return shutil.which("nvidia-smi") is not None


def resolve_xgboost_gpu_mode(mode: str) -> tuple[bool, str]:
    """Resolve the requested runner GPU mode.

    Returns:
    - `use_gpu`: whether the runner should request CUDA-backed XGBoost models
    - `reason`: short explanation for logs / metadata
    """

    mode = str(mode).lower()
    if mode == "off":
        return False, "GPU mode disabled by user."
    if not xgboost_available():
        return False, "xgboost is not installed; falling back to CPU."
    if not xgboost_cuda_build_available():
        return False, "Installed xgboost build does not advertise CUDA support; falling back to CPU."
    if mode == "xgboost":
        return True, "Using XGBoost nuisance models with device='cuda'."
    if cuda_runtime_likely_available():
        return True, "CUDA runtime appears available; enabling XGBoost GPU nuisances."
    return False, "CUDA runtime not detected; falling back to CPU."


def _import_xgboost():
    if not xgboost_available():
        raise ImportError(
            "xgboost is required for GPU-backed nuisance models. "
            "Install it with `python -m pip install xgboost`."
        )
    import xgboost as xgb  # noqa: WPS433

    return xgb


def make_xgb_classifier(
    *,
    random_state: int = 42,
    n_estimators: int = 300,
    min_samples_leaf: int = 20,
    n_jobs: int = 1,
    device: str = "cpu",
):
    """Create an XGBoost classifier for binary treatment modeling."""

    xgb = _import_xgboost()
    kwargs = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": int(n_estimators),
        "max_depth": 6,
        "min_child_weight": max(1, int(min_samples_leaf)),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
        "reg_lambda": 1.0,
        "random_state": int(random_state),
        "tree_method": "hist",
        "n_jobs": int(n_jobs),
    }
    if device:
        kwargs["device"] = str(device)
    return xgb.XGBClassifier(**kwargs)


def make_xgb_regressor(
    *,
    random_state: int = 42,
    n_estimators: int = 300,
    min_samples_leaf: int = 20,
    n_jobs: int = 1,
    device: str = "cpu",
):
    """Create an XGBoost regressor for pseudo-outcome / bridge regression."""

    xgb = _import_xgboost()
    kwargs = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": int(n_estimators),
        "max_depth": 6,
        "min_child_weight": max(1, int(min_samples_leaf)),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
        "reg_lambda": 1.0,
        "random_state": int(random_state),
        "tree_method": "hist",
        "n_jobs": int(n_jobs),
    }
    if device:
        kwargs["device"] = str(device)
    return xgb.XGBRegressor(**kwargs)


__all__ = [
    "cuda_runtime_likely_available",
    "make_xgb_classifier",
    "make_xgb_regressor",
    "resolve_xgboost_gpu_mode",
    "xgboost_available",
    "xgboost_cuda_build_available",
]
