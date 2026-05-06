"""Oracle comparator for the non-censored Final Model.

This file wraps the existing package oracle path used by the benchmark runners.
It is intentionally a comparator, not a deployable estimator: fitting requires
access to the latent confounder ``U`` and DGP-specific oracle nuisance functions.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.non_censored.benchmarks import oracle_h_from_proxy, oracle_q_from_proxy  # noqa: E402
from grf.non_censored.models import SinglePassBridgeFeatureNCCausalForestDML  # noqa: E402


def _ensure_2d(array):
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


class FinalNoncensoredOracle:
    """Non-censored oracle Final Model using latent ``U``.

    The default hyperparameters match the final non-censored benchmark runner:
    augmented full final features, broad proxy duplication, logistic treatment
    bridge, ExtraTrees outcome bridges, and EconML's CausalForestDML final stage.
    """

    def __init__(
        self,
        *,
        cfg,
        dgp,
        p_x: int | None = None,
        use_true_q: bool = True,
        use_true_h: bool = True,
        n_estimators: int = 200,
        min_samples_leaf: int = 20,
        cv: int = 5,
        random_state: int | None = None,
        h_n_estimators: int = 600,
        h_min_samples_leaf: int = 5,
        n_jobs: int = 1,
    ):
        self.cfg = cfg
        self.dgp = dgp
        self.p_x = p_x
        self.use_true_q = bool(use_true_q)
        self.use_true_h = bool(use_true_h)
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.cv = int(cv)
        self.random_state = int(cfg.seed if random_state is None and hasattr(cfg, "seed") else (42 if random_state is None else random_state))
        self.h_n_estimators = int(h_n_estimators)
        self.h_min_samples_leaf = int(h_min_samples_leaf)
        self.n_jobs = int(n_jobs)
        self._model = None

    def _split_x_u(self, X, U):
        x = _ensure_2d(X)
        u = _ensure_2d(U)
        if self.p_x is None:
            if x.shape[1] <= u.shape[1]:
                raise ValueError("p_x must be supplied when X already contains oracle U.")
            self.p_x = int(x.shape[1] - u.shape[1])
        if x.shape[1] == self.p_x:
            x_base = x
            x_oracle = np.hstack([x_base, u])
        elif x.shape[1] == self.p_x + u.shape[1]:
            x_base = x[:, : self.p_x]
            x_oracle = x
        else:
            raise ValueError(
                f"X has {x.shape[1]} columns, expected p_x={self.p_x} "
                f"or p_x + dim(U)={self.p_x + u.shape[1]}."
            )
        return x_base, x_oracle, u

    def _make_model(self, *, x_core_dim: int):
        return SinglePassBridgeFeatureNCCausalForestDML(
            final_feature_mode="aug_full",
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            cv=self.cv,
            random_state=self.random_state,
            q_kind="logit",
            h_kind="extra",
            h_n_estimators=self.h_n_estimators,
            h_min_samples_leaf=self.h_min_samples_leaf,
            q_clip=0.02,
            y_clip_quantile=0.99,
            y_res_clip_percentiles=(1.0, 99.0),
            n_jobs=self.n_jobs,
            x_core_dim=x_core_dim,
            duplicate_proxies_in_nuisance=True,
            nuisance_feature_mode="broad_dup",
            oracle=True,
            use_true_q=self.use_true_q,
            use_true_h=self.use_true_h,
            q_true_fn=partial(oracle_q_from_proxy, dgp=self.dgp, cfg=self.cfg),
            h_true_fn=partial(oracle_h_from_proxy, cfg=self.cfg, dgp=self.dgp),
        )

    def fit_oracle(self, X, A, Y, U):
        """Fit the oracle model using latent ``U``."""

        x_base, x_oracle, u = self._split_x_u(X, U)
        self._model = self._make_model(x_core_dim=x_base.shape[1])
        self._model._raw_w_for_final = np.zeros((x_oracle.shape[0], 0), dtype=float)
        self._model._raw_z_for_final = np.zeros((x_oracle.shape[0], 0), dtype=float)
        self._model.fit_oracle(x_oracle, A, Y, u)
        return self

    def effect_training(self):
        """Return oracle CATE predictions on the fitted training sample."""

        if self._model is None:
            raise RuntimeError("Model must be fit before calling effect_training.")
        return self._model.effect_on_final_features(self._model.training_x_final()).ravel()


FinalModelNoncensoredOracle = FinalNoncensoredOracle


__all__ = ["FinalNoncensoredOracle", "FinalModelNoncensoredOracle"]
