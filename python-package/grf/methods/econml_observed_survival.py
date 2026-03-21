from __future__ import annotations

import numpy as np

from .econml_mild_shrink import EconmlMildShrinkNCSurvivalForest, _ensure_2d


class EconmlMildShrinkObservedSurvivalForest(EconmlMildShrinkNCSurvivalForest):
    """
    Survival-precompute baseline with the same final EconML learner as C3, but
    without NC q/h bridges.

    The model stacks X+W+Z into the final-stage heterogeneity feature, then
    passes zero-valued proxy blocks to the nuisance layer so that q, h, and the
    survival nuisance are all estimated from observed features only.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("q_trees", 300)
        kwargs.setdefault("q_min_samples_leaf", 20)
        kwargs.setdefault("q_poly_degree", 2)
        kwargs.setdefault("q_clip", 0.02)
        kwargs.setdefault("h_kind", "rf")
        kwargs.setdefault("h_n_estimators", 300)
        kwargs.setdefault("h_min_samples_leaf", 20)
        super().__init__(*args, **kwargs)

    def fit_components(self, X, A, time, event, Z, W, **kwargs):
        x_full = self.stack_final_features(X, W, Z, mode=self._final_feature_mode)
        w_dummy = np.zeros_like(_ensure_2d(W), dtype=float)
        z_dummy = np.zeros_like(_ensure_2d(Z), dtype=float)
        return self.fit_survival(x_full, A, time, event, z_dummy, w_dummy, **kwargs)

    def effect_from_components(self, X, W, Z):
        x_full = self.stack_final_features(X, W, Z, mode=self._final_feature_mode)
        return self.effect(x_full)
