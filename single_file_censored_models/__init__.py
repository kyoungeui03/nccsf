"""Self-contained single-file censored models and benchmark runner helpers.

This package groups together the standalone censored-model reference
implementations we built for readability and reproducibility:

    - FinalCensoredModel / FinalModelCensoredSurvivalForest
    - RevisedBaselineCensoredSurvivalForest
    - StrictEconmlXWZCensoredSurvivalForest
    - ProperNoPCICensoredSurvivalForest
    - GRFCensoredBaseline / RGRFCensoredBaseline

The Python-native modules are intentionally self-contained and rely only on
external libraries such as NumPy, lifelines, scikit-learn, and EconML. The GRF
baseline is a thin Python wrapper around an installed R `grf` package.
"""

from .final_censored_model import FinalCensoredModel, FinalModelCensoredSurvivalForest
from .grf_censored_baseline import GRFCensoredBaseline, RGRFCensoredBaseline
from .proper_censored_baseline import ProperCensoredBaseline, ProperNoPCICensoredSurvivalForest
from .revised_censored_baseline import RevisedBaselineCensoredSurvivalForest, RevisedCensoredBaseline
from .strict_censored_baseline import StrictCensoredBaseline, StrictEconmlXWZCensoredSurvivalForest

__all__ = [
    "FinalCensoredModel",
    "FinalModelCensoredSurvivalForest",
    "RevisedBaselineCensoredSurvivalForest",
    "RevisedCensoredBaseline",
    "GRFCensoredBaseline",
    "RGRFCensoredBaseline",
    "StrictCensoredBaseline",
    "StrictEconmlXWZCensoredSurvivalForest",
    "ProperCensoredBaseline",
    "ProperNoPCICensoredSurvivalForest",
]
