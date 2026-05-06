"""Compatibility exports for the conditional-censoring Final Model.

The canonical implementation now lives in ``final_censored_model.py`` and is
selected with ``censoring_mode="conditional"``. This module remains only so
older runners that import ``final_censored_model_conditional`` keep working.
"""

from __future__ import annotations

try:  # pragma: no cover
    from .final_censored_model import (
        ConditionalCensoringFinalCensoredModel,
        ConditionalCensoringFinalPCISurvOnlyCensoredModel,
        ConditionalCensoringFinalPCIXOnlyCensoredModel,
        FinalModelConditionalCensoredSurvivalForest,
        FinalModelConditionalPCISurvOnlyCensoredSurvivalForest,
        FinalModelConditionalPCIXOnlyCensoredSurvivalForest,
    )
except ImportError:  # pragma: no cover
    from censored.final_censored_model import (  # type: ignore
        ConditionalCensoringFinalCensoredModel,
        ConditionalCensoringFinalPCISurvOnlyCensoredModel,
        ConditionalCensoringFinalPCIXOnlyCensoredModel,
        FinalModelConditionalCensoredSurvivalForest,
        FinalModelConditionalPCISurvOnlyCensoredSurvivalForest,
        FinalModelConditionalPCIXOnlyCensoredSurvivalForest,
    )


__all__ = [
    "ConditionalCensoringFinalCensoredModel",
    "FinalModelConditionalCensoredSurvivalForest",
    "ConditionalCensoringFinalPCISurvOnlyCensoredModel",
    "ConditionalCensoringFinalPCIXOnlyCensoredModel",
    "FinalModelConditionalPCISurvOnlyCensoredSurvivalForest",
    "FinalModelConditionalPCIXOnlyCensoredSurvivalForest",
]
