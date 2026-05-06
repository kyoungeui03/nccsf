"""Canonical non-censored NC-CSF entrypoints.

Imports are intentionally lazy so lightweight help commands do not eagerly load
EconML, R helpers, or oracle benchmark dependencies.
"""

__all__ = [
    "BaseNoncensoredModel",
    "EconMLNoncensoredBaseline",
    "FinalNoncensoredModel",
    "FinalModelNCCausalForest",
    "FinalNoncensoredOracle",
    "GRFNoncensoredBaseline",
    "NoncensoredBaseline",
]


def __getattr__(name):
    if name in {"FinalNoncensoredModel", "FinalModelNCCausalForest"}:
        from .final_noncensored_model import FinalModelNCCausalForest, FinalNoncensoredModel

        return {
            "FinalNoncensoredModel": FinalNoncensoredModel,
            "FinalModelNCCausalForest": FinalModelNCCausalForest,
        }[name]
    if name in {"BaseNoncensoredModel", "EconMLNoncensoredBaseline", "NoncensoredBaseline"}:
        from .econml_noncensored_baseline import (
            BaseNoncensoredModel,
            EconMLNoncensoredBaseline,
            NoncensoredBaseline,
        )

        return {
            "BaseNoncensoredModel": BaseNoncensoredModel,
            "EconMLNoncensoredBaseline": EconMLNoncensoredBaseline,
            "NoncensoredBaseline": NoncensoredBaseline,
        }[name]
    if name == "FinalNoncensoredOracle":
        from .final_noncensored_oracle import FinalNoncensoredOracle

        return FinalNoncensoredOracle
    if name == "GRFNoncensoredBaseline":
        from .grf_noncensored_baseline import GRFNoncensoredBaseline

        return GRFNoncensoredBaseline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
