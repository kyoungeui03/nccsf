from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backends import NativeRegressionForest, NativeCausalSurvivalForest
from .common import default_mtry, observation_weights, weighted_mean
from .validation import validate_newdata


@dataclass
class CausalSurvivalForest:
    feature_columns: list[str]
    target: str
    horizon: float
    num_trees: int
    num_threads: int
    seed: int
    train_frame: pd.DataFrame
    train_x: np.ndarray
    W_hat: np.ndarray
    Y_hat: np.ndarray
    psi: dict[str, np.ndarray]
    model: NativeCausalSurvivalForest
    oob_predictions: np.ndarray

    @property
    def predictions(self) -> np.ndarray:
        return self.oob_predictions

    def predict(self, newdata: np.ndarray | pd.DataFrame | None = None, estimate_variance: bool = False) -> pd.DataFrame:
        if estimate_variance:
            raise NotImplementedError("Variance estimates are not exposed in this Python+C++ port.")

        if newdata is None:
            return pd.DataFrame({"predictions": self.oob_predictions})

        newdata = validate_newdata(newdata, self.train_x)
        predictions = self.model.predict(newdata, estimate_variance=False)
        return pd.DataFrame({"predictions": predictions})

    def get_scores(self, num_trees_for_weights: int = 500) -> np.ndarray:
        numerator = self.psi["numerator"]
        denominator = self.psi["denominator"]
        cate_hat = self.oob_predictions
        psi_residual = numerator - denominator * cate_hat

        W_orig = self.train_frame["W"].to_numpy(dtype=float)
        if np.all(np.isin(W_orig, [0.0, 1.0])):
            v_hat = self.W_hat * (1.0 - self.W_hat)
        else:
            variance_forest = NativeRegressionForest.fit(
                self.train_x,
                (W_orig - self.W_hat) ** 2,
                num_trees=int(num_trees_for_weights),
                sample_fraction=0.5,
                mtry=default_mtry(self.train_x.shape[1]),
                min_node_size=5,
                honesty=True,
                honesty_fraction=0.5,
                honesty_prune_leaves=True,
                alpha=0.05,
                imbalance_penalty=0.0,
                ci_group_size=1,
                compute_oob_predictions=True,
                num_threads=self.num_threads,
                seed=self.seed,
            )
            try:
                v_hat = variance_forest.predict()
            finally:
                variance_forest.close()

        return cate_hat + psi_residual / v_hat

    def average_treatment_effect(self) -> dict[str, float]:
        dr_scores = self.get_scores()
        weights = observation_weights(len(dr_scores))
        estimate = weighted_mean(dr_scores, weights)
        centered = (dr_scores - estimate) * weights
        sigma2 = float(np.sum(centered**2) * len(dr_scores) / max(len(dr_scores) - 1, 1))
        return {"estimate": estimate, "std.err": float(np.sqrt(sigma2))}

    def cleanup(self) -> None:
        self.model.close()
