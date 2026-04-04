"""Self-contained single-file censored models and benchmark runner helpers.

This package groups together the three standalone censored-model reference
implementations we built for readability and reproducibility:

    - FinalCensoredModel / FinalModelCensoredSurvivalForest
    - StrictEconmlXWZCensoredSurvivalForest
    - ProperNoPCICensoredSurvivalForest

The modules are intentionally self-contained and rely only on external
libraries such as NumPy, lifelines, scikit-learn, and EconML.
"""

from .final_censored_model import FinalCensoredModel, FinalModelCensoredSurvivalForest
from .proper_censored_baseline import ProperCensoredBaseline, ProperNoPCICensoredSurvivalForest
from .strict_censored_baseline import StrictCensoredBaseline, StrictEconmlXWZCensoredSurvivalForest

__all__ = [
    "FinalCensoredModel",
    "FinalModelCensoredSurvivalForest",
    "StrictCensoredBaseline",
    "StrictEconmlXWZCensoredSurvivalForest",
    "ProperCensoredBaseline",
    "ProperNoPCICensoredSurvivalForest",
]
