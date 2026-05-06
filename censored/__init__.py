"""Canonical censored models and benchmark runner helpers.

The modules in this package are intentionally self-contained. Imports are kept
lazy so that lightweight runners can load one censored model without importing
oracle-only or R-backend dependencies.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "FinalCensoredModel": ("final_censored_model", "FinalCensoredModel"),
    "FinalModelCensoredSurvivalForest": (
        "final_censored_model",
        "FinalModelCensoredSurvivalForest",
    ),
    "FinalPCISurvOnlyCensoredModel": (
        "final_censored_model",
        "FinalPCISurvOnlyCensoredModel",
    ),
    "FinalPCIXOnlyCensoredModel": (
        "final_censored_model",
        "FinalPCIXOnlyCensoredModel",
    ),
    "FinalModelPCISurvOnlyCensoredSurvivalForest": (
        "final_censored_model",
        "FinalModelPCISurvOnlyCensoredSurvivalForest",
    ),
    "FinalModelPCIXOnlyCensoredSurvivalForest": (
        "final_censored_model",
        "FinalModelPCIXOnlyCensoredSurvivalForest",
    ),
    "ConditionalCensoringFinalCensoredModel": (
        "final_censored_model_conditional",
        "ConditionalCensoringFinalCensoredModel",
    ),
    "ConditionalCensoringFinalPCISurvOnlyCensoredModel": (
        "final_censored_model_conditional",
        "ConditionalCensoringFinalPCISurvOnlyCensoredModel",
    ),
    "ConditionalCensoringFinalPCIXOnlyCensoredModel": (
        "final_censored_model_conditional",
        "ConditionalCensoringFinalPCIXOnlyCensoredModel",
    ),
    "FinalModelConditionalCensoredSurvivalForest": (
        "final_censored_model_conditional",
        "FinalModelConditionalCensoredSurvivalForest",
    ),
    "FinalModelConditionalPCISurvOnlyCensoredSurvivalForest": (
        "final_censored_model_conditional",
        "FinalModelConditionalPCISurvOnlyCensoredSurvivalForest",
    ),
    "FinalModelConditionalPCIXOnlyCensoredSurvivalForest": (
        "final_censored_model_conditional",
        "FinalModelConditionalPCIXOnlyCensoredSurvivalForest",
    ),
    "FinalCensoredOracle": ("final_censored_oracle", "FinalCensoredOracle"),
    "CensoredBaseline": ("censored_baseline", "CensoredBaseline"),
    "CensoredBaselineCensoredSurvivalForest": (
        "censored_baseline",
        "CensoredBaselineCensoredSurvivalForest",
    ),
    "RevisedBaselineCensoredSurvivalForest": (
        "censored_baseline",
        "RevisedBaselineCensoredSurvivalForest",
    ),
    "RevisedCensoredBaseline": ("censored_baseline", "RevisedCensoredBaseline"),
    "NaiveCensoredBaseline": ("naive_censored_baseline", "NaiveCensoredBaseline"),
    "NaiveEconmlXWZCensoredSurvivalForest": (
        "naive_censored_baseline",
        "NaiveEconmlXWZCensoredSurvivalForest",
    ),
    "StrictCensoredBaseline": ("naive_censored_baseline", "StrictCensoredBaseline"),
    "StrictEconmlXWZCensoredSurvivalForest": (
        "naive_censored_baseline",
        "StrictEconmlXWZCensoredSurvivalForest",
    ),
    "GRFCensoredBaseline": ("grf_censored_baseline", "GRFCensoredBaseline"),
    "RGRFCensoredBaseline": ("grf_censored_baseline", "RGRFCensoredBaseline"),
}


__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'censored' has no attribute {name!r}") from exc
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
