"""Single-file reference implementation of the current non-censored Final Model.

This file intentionally inlines the helper functions that are otherwise spread
across multiple project files. The goal is readability: someone should be able
to follow the full non-censored finalized pipeline here without jumping between
local modules.

Scope
-----
This file reproduces the *current default* non-censored final model used in the
benchmark pipeline:

    - final feature mode: "aug_full"
    - raw final backbone: [X, W, Z]
    - bridge summaries passed to the final learner:
        [q_pred, h1_pred, h0_pred, m_pred]
    - prediction nuisance mode: "full_refit"
    - nuisance feature mode: "broad_dup"
    - q clip: 0.02
    - bridge target clip: 0.99 quantile
    - residual clip percentiles: (1, 99)
    - final learner: EconML CausalForestDML with a custom final-feature wrapper

The implementation below is deliberately focused on the default path that we
actually benchmark. It does not try to preserve every experimental ablation
branch from the original research files.
"""

from __future__ import annotations

import numpy as np
from econml._ortho_learner import _OrthoLearner
from econml.dml import CausalForestDML
from econml.utilities import filter_none_kwargs
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# ---------------------------------------------------------------------------
# Generic array helpers
# ---------------------------------------------------------------------------


def _ensure_2d(array):
    """Return a 2D float array."""

    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _clip_quantile(values, q):
    """Winsorize values to the symmetric quantile range used by the model."""

    values = np.asarray(values, dtype=float)
    if q is None:
        return values
    lo = float(np.quantile(values, 1.0 - q))
    hi = float(np.quantile(values, q))
    return np.clip(values, lo, hi)


def _pairwise_products(left, right):
    """Compute all pairwise feature interactions between two matrices."""

    left = _ensure_2d(left)
    right = _ensure_2d(right)
    if left.shape[1] == 0 or right.shape[1] == 0:
        return np.empty((left.shape[0], 0), dtype=float)
    return (left[:, :, None] * right[:, None, :]).reshape(left.shape[0], -1)


# ---------------------------------------------------------------------------
# Base ML models used inside the nuisance layer
# ---------------------------------------------------------------------------


def make_q_model(
    kind="logit",
    *,
    random_state=42,
    n_estimators=300,
    min_samples_leaf=20,
):
    """Create the treatment nuisance model q(X, Z)."""

    if kind == "poly2":
        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("logit", LogisticRegression(max_iter=10000)),
            ]
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    return LogisticRegression(max_iter=10000)


def make_h_model(
    kind="extra",
    *,
    random_state=42,
    n_estimators=600,
    min_samples_leaf=5,
    n_jobs=1,
):
    """Create the bridge regression model h(X, W)."""

    if kind == "extra":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def _build_single_pass_nc_features(X, W_raw, Z_raw, bridge):
    """Build the current default final-stage feature block.

    Current finalized non-censored features are exactly:

        [X, W, Z, q_pred, h1_pred, h0_pred, m_pred]
    """

    return np.hstack(
        [
            _ensure_2d(X).astype(float),
            _ensure_2d(W_raw).astype(float),
            _ensure_2d(Z_raw).astype(float),
            np.asarray(bridge["q_pred"], dtype=float).reshape(-1, 1),
            np.asarray(bridge["h1_pred"], dtype=float).reshape(-1, 1),
            np.asarray(bridge["h0_pred"], dtype=float).reshape(-1, 1),
            np.asarray(bridge["m_pred"], dtype=float).reshape(-1, 1),
        ]
    )


# ---------------------------------------------------------------------------
# Nuisance model
# ---------------------------------------------------------------------------


class _ConfigurableNCNuisance:
    """Bridge nuisance logic for the default non-censored finalized model."""

    def __init__(
        self,
        *,
        q_model,
        h_model,
        q_clip,
        y_clip_quantile,
        y_res_clip_percentiles,
        x_core_dim,
        duplicate_proxies_in_nuisance,
        nuisance_feature_mode,
    ):
        self._q_model_template = q_model
        self._h_model_template = h_model
        self._q_clip = float(q_clip)
        self._y_clip_quantile = y_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._x_core_dim = int(x_core_dim)
        self._duplicate_proxies_in_nuisance = bool(duplicate_proxies_in_nuisance)
        self._nuisance_feature_mode = nuisance_feature_mode

        self._q_model = None
        self._h1_model = None
        self._h0_model = None

    def _split_features(self, X, W, Z):
        X = np.asarray(X, dtype=float)
        X_core = X[:, : self._x_core_dim]
        W = _ensure_2d(W)
        Z = _ensure_2d(Z)
        x_base = X if self._duplicate_proxies_in_nuisance else X_core
        base = np.column_stack([x_base, W, Z])

        if self._nuisance_feature_mode == "broad_dup":
            XZ = np.column_stack([base, Z])
            XW = np.column_stack([base, W])
            return XZ, XW
        if self._duplicate_proxies_in_nuisance:
            XZ = np.column_stack([X, Z])
            XW = np.column_stack([X, W])
        else:
            XZ = np.column_stack([X_core, Z])
            XW = np.column_stack([X_core, W])
        if self._nuisance_feature_mode == "interact":
            extra = np.column_stack(
                [
                    _pairwise_products(X_core, W),
                    _pairwise_products(X_core, Z),
                    _pairwise_products(W, Z),
                ]
            )
            XZ = np.column_stack([XZ, extra])
            XW = np.column_stack([XW, extra])
        elif self._nuisance_feature_mode != "dup":
            raise ValueError(f"Unsupported nuisance_feature_mode: {self._nuisance_feature_mode}")
        return XZ, XW

    def _predict_bridge_components(self, *, X=None, W=None, Z=None):
        q_features, h_features = self._split_features(X, W, Z)
        q_pred = self._q_model.predict_proba(q_features)[:, 1]
        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        h1_pred = self._h1_model.predict(h_features)
        h0_pred = self._h0_model.predict(h_features)
        m_pred = q_pred * h1_pred + (1.0 - q_pred) * h0_pred
        return q_pred, h1_pred, h0_pred, m_pred

    def train(
        self,
        is_selecting,
        folds,
        Y,
        T,
        X=None,
        W=None,
        Z=None,
        sample_weight=None,
        groups=None,
    ):
        del is_selecting, folds, groups

        A = np.asarray(T, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        Y_train = _clip_quantile(Y, self._y_clip_quantile)
        q_features, h_features = self._split_features(X, W, Z)

        self._q_model = clone(self._q_model_template)
        self._q_model.fit(q_features, A, **filter_none_kwargs(sample_weight=sample_weight))

        treated_mask = A == 1
        control_mask = A == 0
        if treated_mask.sum() <= 10 or control_mask.sum() <= 10:
            raise ValueError("Each treatment arm must have more than 10 samples per fold for this model.")

        self._h1_model = clone(self._h_model_template)
        self._h0_model = clone(self._h_model_template)
        self._h1_model.fit(
            h_features[treated_mask],
            Y_train[treated_mask],
            **filter_none_kwargs(
                sample_weight=None if sample_weight is None else sample_weight[treated_mask]
            ),
        )
        self._h0_model.fit(
            h_features[control_mask],
            Y_train[control_mask],
            **filter_none_kwargs(
                sample_weight=None if sample_weight is None else sample_weight[control_mask]
            ),
        )
        return self

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        del sample_weight, groups

        A = np.asarray(T, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        q_pred, _, _, m_pred = self._predict_bridge_components(X=X, W=W, Z=Z)

        y_res = Y - m_pred
        lo, hi = np.percentile(y_res, self._y_res_clip_percentiles)
        y_res = np.clip(y_res, lo, hi)
        a_res = (A - q_pred).reshape(-1, 1)
        return y_res, a_res

    def predict_bridge_outputs(self, *, X=None, W=None, Z=None):
        q_pred, h1_pred, h0_pred, m_pred = self._predict_bridge_components(X=X, W=W, Z=Z)
        return {
            "q_pred": q_pred,
            "h1_pred": h1_pred,
            "h0_pred": h0_pred,
            "m_pred": m_pred,
        }


class _BridgeOutputNCNuisance(_ConfigurableNCNuisance):
    """Nuisance variant whose predict() returns everything needed for x_final."""

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        y_res, a_res = super().predict(
            Y,
            T,
            X=X,
            W=W,
            Z=Z,
            sample_weight=sample_weight,
            groups=groups,
        )
        bridge = super().predict_bridge_outputs(X=X, W=W, Z=Z)
        return (
            y_res,
            a_res,
            bridge["q_pred"],
            bridge["h1_pred"],
            bridge["h0_pred"],
            bridge["m_pred"],
        )


# ---------------------------------------------------------------------------
# Custom final-model wrapper for CausalForestDML
# ---------------------------------------------------------------------------


class _BridgeFeatureModelFinal:
    """Translate nuisance outputs into x_final before calling EconML's final stage."""

    def __init__(self, base_model_final, *, raw_proxy_supplier=None):
        self._base_model_final = base_model_final
        self._raw_proxy_supplier = raw_proxy_supplier
        self._train_x_final = None

    def _transform(self, X, W, Z, nuisances):
        y_res, a_res, q_pred, h1_pred, h0_pred, m_pred = nuisances
        bridge = {
            "q_pred": q_pred,
            "h1_pred": h1_pred,
            "h0_pred": h0_pred,
            "m_pred": m_pred,
        }

        w_for_final = W
        z_for_final = Z
        if self._raw_proxy_supplier is not None:
            supplied = self._raw_proxy_supplier(X=X, W=W, Z=Z)
            if supplied is not None:
                w_for_final, z_for_final = supplied

        x_final = _build_single_pass_nc_features(X, w_for_final, z_for_final, bridge)
        return x_final, (y_res, a_res)

    def fit(
        self,
        Y,
        T,
        X=None,
        W=None,
        Z=None,
        nuisances=None,
        sample_weight=None,
        freq_weight=None,
        sample_var=None,
        groups=None,
    ):
        x_final, core_nuisances = self._transform(X, W, Z, nuisances)
        self._train_x_final = np.asarray(x_final, dtype=float).copy()
        self._base_model_final.fit(
            Y,
            T,
            X=x_final,
            W=W,
            Z=Z,
            nuisances=core_nuisances,
            sample_weight=sample_weight,
            freq_weight=freq_weight,
            sample_var=sample_var,
            groups=groups,
        )
        return self

    def score(
        self,
        Y,
        T,
        X=None,
        W=None,
        Z=None,
        nuisances=None,
        sample_weight=None,
        groups=None,
        scoring="mean_squared_error",
    ):
        x_final, core_nuisances = self._transform(X, W, Z, nuisances)
        return self._base_model_final.score(
            Y,
            T,
            X=x_final,
            W=W,
            Z=Z,
            nuisances=core_nuisances,
            sample_weight=sample_weight,
            groups=groups,
            scoring=scoring,
        )

    def predict(self, X=None):
        return self._base_model_final.predict(X)

    def training_x_final(self):
        if self._train_x_final is None:
            return None
        return np.asarray(self._train_x_final, dtype=float)


# ---------------------------------------------------------------------------
# Custom CausalForestDML subclass
# ---------------------------------------------------------------------------


class _SinglePassBridgeFeatureNCCausalForest(CausalForestDML):
    """CausalForestDML whose nuisance and final stages follow our Final Model."""

    def __init__(
        self,
        *,
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=42,
        q_kind="logit",
        q_trees=300,
        q_leaf=20,
        q_clip=0.02,
        h_kind="extra",
        y_clip_quantile=0.99,
        y_res_clip_percentiles=(1.0, 99.0),
        h_n_estimators=600,
        h_min_samples_leaf=5,
        nuisance_feature_mode="broad_dup",
        x_core_dim=1,
    ):
        self._q_model_template = make_q_model(
            q_kind,
            random_state=random_state,
            n_estimators=q_trees,
            min_samples_leaf=q_leaf,
        )
        self._h_model_template = make_h_model(
            h_kind,
            random_state=random_state,
            n_estimators=h_n_estimators,
            min_samples_leaf=h_min_samples_leaf,
            n_jobs=1,
        )
        self._custom_q_clip = float(q_clip)
        self._custom_y_clip_quantile = y_clip_quantile
        self._custom_y_res_clip_percentiles = y_res_clip_percentiles
        self._nuisance_feature_mode = nuisance_feature_mode
        self._x_core_dim = int(x_core_dim)
        self._raw_w_for_final = None
        self._raw_z_for_final = None

        super().__init__(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            cv=cv,
            random_state=random_state,
            discrete_treatment=True,
            criterion="het",
        )

    def _raw_proxy_for_final(self, *, X=None, W=None, Z=None):
        del W, Z
        if X is None:
            return None
        if self._raw_w_for_final is None or self._raw_z_for_final is None:
            return None
        if len(np.asarray(X, dtype=float)) != len(self._raw_w_for_final):
            return None
        return self._raw_w_for_final, self._raw_z_for_final

    def _gen_ortho_learner_model_nuisance(self):
        return _BridgeOutputNCNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            q_clip=self._custom_q_clip,
            y_clip_quantile=self._custom_y_clip_quantile,
            y_res_clip_percentiles=self._custom_y_res_clip_percentiles,
            x_core_dim=self._x_core_dim,
            duplicate_proxies_in_nuisance=True,
            nuisance_feature_mode=self._nuisance_feature_mode,
        )

    def _gen_ortho_learner_model_final(self):
        return _BridgeFeatureModelFinal(
            super()._gen_ortho_learner_model_final(),
            raw_proxy_supplier=self._raw_proxy_for_final,
        )

    def fit_nc(self, X, A, Y, Z, W, **kwargs):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).ravel()
        A = np.asarray(A, dtype=float).ravel()
        Z = _ensure_2d(Z)
        W = _ensure_2d(W)
        return _OrthoLearner.fit(self, Y, A, X=X, W=W, Z=Z, **kwargs)

    def effect_on_final_features(self, x_final):
        return np.asarray(self._ortho_learner_model_final.predict(x_final), dtype=float)

    def training_x_final(self):
        return self._ortho_learner_model_final.training_x_final()


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------


class FinalNoncensoredModel:
    """Single-file version of the current non-censored Final Model.

    Public API mirrors the current project style:

        model.fit_components(X, A, Y, Z, W)
        tau_hat = model.effect_from_components(X, W, Z)
    """

    def __init__(
        self,
        *,
        cv=5,
        random_state=42,
        q_kind="logit",
        q_trees=300,
        q_leaf=20,
        q_clip=0.02,
        h_kind="extra",
        h_n_estimators=600,
        h_min_samples_leaf=5,
        y_clip_quantile=0.99,
        y_res_clip_percentiles=(1.0, 99.0),
        n_estimators=200,
        min_samples_leaf=20,
        nuisance_feature_mode="broad_dup",
        prediction_nuisance_mode="full_refit",
        final_feature_mode="aug_full",
        observed_only=False,
    ):
        if final_feature_mode != "aug_full":
            raise ValueError("This single-file implementation reproduces only final_feature_mode='aug_full'.")
        if prediction_nuisance_mode != "full_refit":
            raise ValueError("This single-file implementation supports prediction_nuisance_mode='full_refit' only.")

        self._cv = int(cv)
        self._random_state = int(random_state)
        self._q_kind = q_kind
        self._q_trees = int(q_trees)
        self._q_leaf = int(q_leaf)
        self._q_clip = float(q_clip)
        self._h_kind = h_kind
        self._h_n_estimators = int(h_n_estimators)
        self._h_min_samples_leaf = int(h_min_samples_leaf)
        self._y_clip_quantile = y_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._nuisance_feature_mode = nuisance_feature_mode
        self._prediction_nuisance_mode = prediction_nuisance_mode
        self._observed_only = bool(observed_only)

        self._q_model_template = make_q_model(
            q_kind,
            random_state=random_state,
            n_estimators=q_trees,
            min_samples_leaf=q_leaf,
        )
        self._h_model_template = make_h_model(
            h_kind,
            random_state=random_state,
            n_estimators=h_n_estimators,
            min_samples_leaf=h_min_samples_leaf,
            n_jobs=1,
        )

        self._dml_model = None
        self._feature_nuisance = None
        self._train_x = None
        self._train_w = None
        self._train_z = None
        self._train_x_final = None

    def _prepare_nuisance_inputs(self, W, Z):
        w = _ensure_2d(W).astype(float)
        z = _ensure_2d(Z).astype(float)
        if self._observed_only:
            w = np.zeros_like(w)
            z = np.zeros_like(z)
        return w, z

    def _make_feature_nuisance(self, x_core_dim):
        return _ConfigurableNCNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            q_clip=self._q_clip,
            y_clip_quantile=self._y_clip_quantile,
            y_res_clip_percentiles=self._y_res_clip_percentiles,
            x_core_dim=x_core_dim,
            duplicate_proxies_in_nuisance=True,
            nuisance_feature_mode=self._nuisance_feature_mode,
        )

    def fit_components(self, X, A, Y, Z, W):
        """Fit the finalized non-censored model."""

        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)

        self._dml_model = _SinglePassBridgeFeatureNCCausalForest(
            n_estimators=self._n_estimators,
            min_samples_leaf=self._min_samples_leaf,
            cv=self._cv,
            random_state=self._random_state,
            q_kind=self._q_kind,
            q_trees=self._q_trees,
            q_leaf=self._q_leaf,
            q_clip=self._q_clip,
            h_kind=self._h_kind,
            y_clip_quantile=self._y_clip_quantile,
            y_res_clip_percentiles=self._y_res_clip_percentiles,
            h_n_estimators=self._h_n_estimators,
            h_min_samples_leaf=self._h_min_samples_leaf,
            nuisance_feature_mode=self._nuisance_feature_mode,
            x_core_dim=x.shape[1],
        )
        self._dml_model._raw_w_for_final = raw_w
        self._dml_model._raw_z_for_final = raw_z
        self._dml_model.fit_nc(x, A, Y, z_nuis, w_nuis)

        self._feature_nuisance = self._make_feature_nuisance(x.shape[1])
        self._feature_nuisance.train(
            False,
            None,
            np.asarray(Y, dtype=float).ravel(),
            np.asarray(A, dtype=float).ravel(),
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )

        self._train_x = np.asarray(x, dtype=float).copy()
        self._train_w = np.asarray(raw_w, dtype=float).copy()
        self._train_z = np.asarray(raw_z, dtype=float).copy()
        self._train_x_final = self._dml_model.training_x_final()
        return self

    def effect_from_components(self, X, W, Z):
        """Predict CATE using full-data nuisance reconstruction."""

        if self._dml_model is None or self._feature_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")

        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)

        bridge = self._feature_nuisance.predict_bridge_outputs(X=x, W=w_nuis, Z=z_nuis)
        x_final = _build_single_pass_nc_features(x, raw_w, raw_z, bridge)
        return self._dml_model.effect_on_final_features(x_final)

    def training_x_final(self):
        if self._train_x_final is None:
            return None
        return np.asarray(self._train_x_final, dtype=float)


# Familiar alias so callers can use the existing project-style model name.
FinalModelNCCausalForest = FinalNoncensoredModel


__all__ = ["FinalNoncensoredModel", "FinalModelNCCausalForest"]
