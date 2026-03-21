from __future__ import annotations

from typing import Callable

import numpy as np
from econml._ortho_learner import _OrthoLearner
from econml.dml import CausalForestDML
from econml.grf import CausalForest
from econml.utilities import filter_none_kwargs
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


ArrayLike = np.ndarray


def _ensure_2d(array: ArrayLike | None) -> ArrayLike | None:
    if array is None:
        return None
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _clip_quantile(values: ArrayLike, q: float | None) -> ArrayLike:
    values = np.asarray(values, dtype=float)
    if q is None:
        return values
    lo = float(np.quantile(values, 1.0 - q))
    hi = float(np.quantile(values, q))
    return np.clip(values, lo, hi)


def _pairwise_products(left: ArrayLike, right: ArrayLike) -> ArrayLike:
    left = _ensure_2d(left)
    right = _ensure_2d(right)
    if left.shape[1] == 0 or right.shape[1] == 0:
        return np.empty((left.shape[0], 0), dtype=float)
    return (left[:, :, None] * right[:, None, :]).reshape(left.shape[0], -1)


def _select_curve_knots(curve: ArrayLike, n_knots: int = 3) -> ArrayLike:
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2:
        raise ValueError("curve must have shape (n_samples, n_grid).")
    n_grid = curve.shape[1]
    if n_grid == 0:
        return np.zeros((curve.shape[0], n_knots), dtype=float)
    knot_idx = np.linspace(0, n_grid - 1, num=min(n_knots, n_grid)).round().astype(int)
    knot_idx = np.unique(knot_idx)
    selected = curve[:, knot_idx]
    if selected.shape[1] < n_knots:
        pad = np.repeat(selected[:, [-1]], n_knots - selected.shape[1], axis=1)
        selected = np.hstack([selected, pad])
    return selected


def _build_nc_summary_features(
    X: ArrayLike,
    bridge: dict[str, ArrayLike],
    *,
    mode: str = "basic",
    n_curve_knots: int = 3,
) -> ArrayLike:
    x = _ensure_2d(X).astype(float)
    q_pred = np.asarray(bridge["q_pred"], dtype=float).reshape(-1, 1)
    h1_pred = np.asarray(bridge["h1_pred"], dtype=float).reshape(-1, 1)
    h0_pred = np.asarray(bridge["h0_pred"], dtype=float).reshape(-1, 1)
    m_pred = np.asarray(bridge["m_pred"], dtype=float).reshape(-1, 1)
    h_diff = h1_pred - h0_pred
    q_margin = np.abs(q_pred - 0.5)
    balance = 4.0 * q_pred * (1.0 - q_pred)

    parts = [x, q_pred, h1_pred, h0_pred, h_diff, m_pred]
    if mode == "basic":
        return np.hstack(parts)

    bridge_curve = np.hstack([h0_pred, m_pred, h1_pred])
    curve_knots = _select_curve_knots(bridge_curve, n_knots=n_curve_knots)
    curve_diff = np.diff(curve_knots, axis=1)
    curve_mean = (
        0.5 * (curve_knots[:, 1:] + curve_knots[:, :-1])
        if curve_knots.shape[1] >= 2
        else curve_knots
    )
    curve_span = curve_knots[:, [-1]] - curve_knots[:, [0]]

    if mode == "curve":
        return np.hstack(parts + [q_margin, balance, curve_knots, curve_mean, curve_diff, curve_span])
    if mode == "curve_x_interact":
        curve_block = np.hstack([curve_knots, curve_mean, curve_diff, curve_span])
        interact_base = np.hstack([q_pred, h_diff, m_pred, q_margin, balance, curve_diff, curve_span])
        return np.hstack(parts + [q_margin, balance, curve_block, _pairwise_products(x, interact_base)])
    raise ValueError(f"Unsupported non-censored summary feature mode: {mode}")


def make_q_model(
    kind: str = "logit",
    *,
    random_state: int = 42,
    n_estimators: int = 300,
    min_samples_leaf: int = 20,
):
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
    kind: str = "rf",
    *,
    random_state: int = 42,
    n_estimators: int = 300,
    min_samples_leaf: int = 20,
    n_jobs: int = 1,
):
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


class _ConfigurableNCNuisance:
    def __init__(
        self,
        *,
        q_model,
        h_model,
        q_clip: float,
        y_clip_quantile: float | None,
        y_res_clip_percentiles: tuple[float, float],
        x_core_dim: int,
        duplicate_proxies_in_nuisance: bool,
        nuisance_feature_mode: str,
        oracle: bool,
        use_true_q: bool,
        use_true_h: bool,
        q_true_fn: Callable[[ArrayLike, ArrayLike], ArrayLike] | None,
        h_true_fn: Callable[[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]] | None,
    ):
        self._q_model_template = q_model
        self._h_model_template = h_model
        self._q_clip = q_clip
        self._y_clip_quantile = y_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._x_core_dim = int(x_core_dim)
        self._duplicate_proxies_in_nuisance = bool(duplicate_proxies_in_nuisance)
        self._nuisance_feature_mode = nuisance_feature_mode
        self._oracle = bool(oracle)
        self._use_true_q = bool(use_true_q)
        self._use_true_h = bool(use_true_h)
        self._q_true_fn = q_true_fn
        self._h_true_fn = h_true_fn

        self._q_model = None
        self._h1_model = None
        self._h0_model = None

    def _split_features(self, X, W, Z):
        X = np.asarray(X, dtype=float)
        X_core = X[:, : self._x_core_dim]
        if self._oracle:
            U = _ensure_2d(W)
            if self._duplicate_proxies_in_nuisance:
                X_proxy = np.column_stack([X, U])
            else:
                X_proxy = np.column_stack([X_core, U])
            return X_core, U, X_proxy, X_proxy

        W = _ensure_2d(W)
        Z = _ensure_2d(Z)
        base = np.column_stack([X, W, Z]) if self._duplicate_proxies_in_nuisance else np.column_stack([X_core, W, Z])
        if self._duplicate_proxies_in_nuisance:
            XZ = np.column_stack([X, Z])
            XW = np.column_stack([X, W])
        else:
            XZ = np.column_stack([X_core, Z])
            XW = np.column_stack([X_core, W])
        if self._nuisance_feature_mode == "interact":
            extra = np.column_stack([_pairwise_products(X_core, W), _pairwise_products(X_core, Z), _pairwise_products(W, Z)])
            XZ = np.column_stack([XZ, extra])
            XW = np.column_stack([XW, extra])
        elif self._nuisance_feature_mode != "dup":
            raise ValueError(f"Unsupported nuisance feature mode: {self._nuisance_feature_mode}")
        return X_core, (W, Z), XZ, XW

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
        A = np.asarray(T).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        Y_train = _clip_quantile(Y, self._y_clip_quantile)

        X_core, proxy, q_features, h_features = self._split_features(X, W, Z)

        if not self._use_true_q:
            self._q_model = clone(self._q_model_template)
            self._q_model.fit(q_features, A, **filter_none_kwargs(sample_weight=sample_weight))

        if not self._use_true_h:
            treated_mask = A == 1
            control_mask = A == 0

            self._h1_model = clone(self._h_model_template)
            self._h0_model = clone(self._h_model_template)

            if treated_mask.sum() > 10:
                self._h1_model.fit(
                    h_features[treated_mask],
                    Y_train[treated_mask],
                    **filter_none_kwargs(
                        sample_weight=None if sample_weight is None else sample_weight[treated_mask]
                    ),
                )
            if control_mask.sum() > 10:
                self._h0_model.fit(
                    h_features[control_mask],
                    Y_train[control_mask],
                    **filter_none_kwargs(
                        sample_weight=None if sample_weight is None else sample_weight[control_mask]
                    ),
                )
        return self

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        A = np.asarray(T).ravel()
        Y = np.asarray(Y, dtype=float).ravel()

        X_core, proxy, q_features, h_features = self._split_features(X, W, Z)

        if self._use_true_q:
            if self._q_true_fn is None:
                raise ValueError("True q requested but q_true_fn is missing.")
            q_pred = np.asarray(self._q_true_fn(X_core, proxy), dtype=float).ravel()
        else:
            q_pred = self._q_model.predict_proba(q_features)[:, 1]

        if self._use_true_h:
            if self._h_true_fn is None:
                raise ValueError("True h requested but h_true_fn is missing.")
            h0_pred, h1_pred = self._h_true_fn(X_core, proxy)
            h0_pred = np.asarray(h0_pred, dtype=float).ravel()
            h1_pred = np.asarray(h1_pred, dtype=float).ravel()
        else:
            h1_pred = self._h1_model.predict(h_features)
            h0_pred = self._h0_model.predict(h_features)

        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        m_pred = q_pred * h1_pred + (1.0 - q_pred) * h0_pred

        y_res = Y - m_pred
        lo, hi = np.percentile(y_res, self._y_res_clip_percentiles)
        y_res = np.clip(y_res, lo, hi)
        a_res = (A - q_pred).reshape(-1, 1)
        return y_res, a_res

    def predict_bridge_outputs(self, *, X=None, W=None, Z=None):
        X_core, proxy, q_features, h_features = self._split_features(X, W, Z)

        if self._use_true_q:
            if self._q_true_fn is None:
                raise ValueError("True q requested but q_true_fn is missing.")
            q_pred = np.asarray(self._q_true_fn(X_core, proxy), dtype=float).ravel()
        else:
            q_pred = self._q_model.predict_proba(q_features)[:, 1]

        if self._use_true_h:
            if self._h_true_fn is None:
                raise ValueError("True h requested but h_true_fn is missing.")
            h0_pred, h1_pred = self._h_true_fn(X_core, proxy)
            h0_pred = np.asarray(h0_pred, dtype=float).ravel()
            h1_pred = np.asarray(h1_pred, dtype=float).ravel()
        else:
            h1_pred = self._h1_model.predict(h_features)
            h0_pred = self._h0_model.predict(h_features)

        q_pred = np.clip(q_pred, self._q_clip, 1.0 - self._q_clip)
        m_pred = q_pred * h1_pred + (1.0 - q_pred) * h0_pred
        return {
            "q_pred": q_pred,
            "h1_pred": h1_pred,
            "h0_pred": h0_pred,
            "m_pred": m_pred,
        }


class BaselineCausalForestDML(CausalForestDML):
    def __init__(self, *, n_estimators=200, min_samples_leaf=20, random_state=42, **kwargs):
        kwargs.setdefault(
            "model_y",
            RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=20,
                random_state=random_state,
            ),
        )
        kwargs.setdefault(
            "model_t",
            RandomForestClassifier(
                n_estimators=100,
                min_samples_leaf=20,
                random_state=random_state,
            ),
        )
        kwargs.setdefault("discrete_treatment", True)
        kwargs.setdefault("criterion", "het")
        super().__init__(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs,
        )

    def fit_baseline(self, X, A, Y, **kwargs):
        return self.fit(Y=np.asarray(Y, dtype=float), T=np.asarray(A).ravel(), X=np.asarray(X, dtype=float), **kwargs)


class MildShrinkNCCausalForestDML(CausalForestDML):
    def __init__(
        self,
        *,
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=42,
        criterion="het",
        q_kind="logit",
        q_trees=300,
        q_leaf=20,
        q_clip=0.02,
        h_kind="rf",
        y_clip_quantile=0.99,
        y_res_clip_percentiles=(1.0, 99.0),
        h_n_estimators=300,
        h_min_samples_leaf=20,
        n_jobs=1,
        x_core_dim: int,
        duplicate_proxies_in_nuisance=True,
        nuisance_feature_mode="dup",
        oracle=False,
        use_true_q=False,
        use_true_h=False,
        q_true_fn=None,
        h_true_fn=None,
        **kwargs,
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
            n_jobs=n_jobs,
        )
        self._custom_q_clip = q_clip
        self._custom_y_clip_quantile = y_clip_quantile
        self._custom_y_res_clip_percentiles = y_res_clip_percentiles
        self._x_core_dim = int(x_core_dim)
        self._duplicate_proxies_in_nuisance = bool(duplicate_proxies_in_nuisance)
        self._nuisance_feature_mode = nuisance_feature_mode
        self._oracle = bool(oracle)
        self._use_true_q = bool(use_true_q)
        self._use_true_h = bool(use_true_h)
        self._q_true_fn = q_true_fn
        self._h_true_fn = h_true_fn

        kwargs["discrete_treatment"] = True
        kwargs["criterion"] = criterion
        kwargs.pop("model_y", None)
        kwargs.pop("model_t", None)

        super().__init__(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            cv=cv,
            random_state=random_state,
            **kwargs,
        )

    def _gen_ortho_learner_model_nuisance(self):
        return _ConfigurableNCNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            q_clip=self._custom_q_clip,
            y_clip_quantile=self._custom_y_clip_quantile,
            y_res_clip_percentiles=self._custom_y_res_clip_percentiles,
            x_core_dim=self._x_core_dim,
            duplicate_proxies_in_nuisance=self._duplicate_proxies_in_nuisance,
            nuisance_feature_mode=self._nuisance_feature_mode,
            oracle=self._oracle,
            use_true_q=self._use_true_q,
            use_true_h=self._use_true_h,
            q_true_fn=self._q_true_fn,
            h_true_fn=self._h_true_fn,
        )

    @staticmethod
    def stack_final_features(*arrays):
        parts = [_ensure_2d(np.asarray(arr, dtype=float)) for arr in arrays]
        return np.hstack(parts)

    def fit_nc(self, X, A, Y, Z, W, **kwargs):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).ravel()
        A = np.asarray(A).ravel()
        Z = _ensure_2d(Z)
        W = _ensure_2d(W)
        return _OrthoLearner.fit(self, Y, A, X=X, W=W, Z=Z, **kwargs)

    def fit_oracle(self, X, A, Y, U, **kwargs):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).ravel()
        A = np.asarray(A).ravel()
        U = _ensure_2d(U)
        return _OrthoLearner.fit(self, Y, A, X=X, W=U, Z=None, **kwargs)


def _crossfit_summary_arrays_nc(owner, X, A, Y, W, Z):
    x = _ensure_2d(X).astype(float)
    y = np.asarray(Y, dtype=float).ravel()
    a = np.asarray(A, dtype=float).ravel()
    w_nuis, z_nuis = owner._prepare_nuisance_inputs(W, Z)

    splitter = KFold(n_splits=owner._cv, shuffle=True, random_state=owner._random_state)
    x_final = None
    y_res = np.empty_like(y, dtype=float)
    a_res = np.empty_like(y, dtype=float)

    for train_idx, test_idx in splitter.split(x):
        nuisance = owner._make_nuisance()
        nuisance.train(
            False,
            None,
            y[train_idx],
            a[train_idx],
            X=x[train_idx],
            W=w_nuis[train_idx],
            Z=z_nuis[train_idx],
        )
        y_res_fold, a_res_fold = nuisance.predict(
            y[test_idx],
            a[test_idx],
            X=x[test_idx],
            W=w_nuis[test_idx],
            Z=z_nuis[test_idx],
        )
        bridge = nuisance.predict_bridge_outputs(
            X=x[test_idx],
            W=w_nuis[test_idx],
            Z=z_nuis[test_idx],
        )
        x_final_fold = _build_nc_summary_features(
            x[test_idx],
            bridge,
            mode=owner._summary_feature_mode,
            n_curve_knots=owner._summary_curve_knots,
        )
        if x_final is None:
            x_final = np.empty((len(y), x_final_fold.shape[1]), dtype=float)
        x_final[test_idx] = x_final_fold
        y_res[test_idx] = np.asarray(y_res_fold, dtype=float).ravel()
        a_res[test_idx] = np.asarray(a_res_fold, dtype=float).ravel()

    return x, a, w_nuis, z_nuis, x_final, y_res, a_res


class _BaseTwoStageSummaryNCCausalForest:
    def __init__(
        self,
        *,
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=42,
        observed_only=False,
        nuisance_feature_mode="dup",
        q_kind="logit",
        q_trees=300,
        q_leaf=20,
        q_clip=0.02,
        h_kind="extra",
        y_clip_quantile=0.99,
        y_res_clip_percentiles=(1.0, 99.0),
        h_n_estimators=300,
        h_min_samples_leaf=20,
        n_jobs=1,
        summary_feature_mode="basic",
        summary_curve_knots=3,
        forest_max_depth=None,
        forest_honest=True,
        forest_inference=True,
        forest_fit_intercept=True,
        oracle=False,
        use_true_q=False,
        use_true_h=False,
        q_true_fn=None,
        h_true_fn=None,
    ):
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._cv = int(cv)
        self._random_state = int(random_state)
        self._observed_only = bool(observed_only)
        self._nuisance_feature_mode = nuisance_feature_mode
        self._summary_feature_mode = summary_feature_mode
        self._summary_curve_knots = int(summary_curve_knots)
        self._forest_max_depth = forest_max_depth
        self._forest_honest = bool(forest_honest)
        self._forest_inference = bool(forest_inference)
        self._forest_fit_intercept = bool(forest_fit_intercept)
        self._oracle = bool(oracle)
        self._use_true_q = bool(use_true_q)
        self._use_true_h = bool(use_true_h)
        self._q_true_fn = q_true_fn
        self._h_true_fn = h_true_fn
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
            n_jobs=n_jobs,
        )
        self._q_clip = q_clip
        self._y_clip_quantile = y_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._forest = None
        self._full_nuisance = None
        self._x_core_dim = None

    def _make_nuisance(self):
        return _ConfigurableNCNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            q_clip=self._q_clip,
            y_clip_quantile=self._y_clip_quantile,
            y_res_clip_percentiles=self._y_res_clip_percentiles,
            x_core_dim=self._x_core_dim,
            duplicate_proxies_in_nuisance=True,
            nuisance_feature_mode=self._nuisance_feature_mode,
            oracle=self._oracle,
            use_true_q=self._use_true_q,
            use_true_h=self._use_true_h,
            q_true_fn=self._q_true_fn,
            h_true_fn=self._h_true_fn,
        )

    def _prepare_nuisance_inputs(self, W, Z):
        w = _ensure_2d(W).astype(float)
        if Z is None:
            z = np.zeros((w.shape[0], 1), dtype=float)
        else:
            z = _ensure_2d(Z).astype(float)
        if self._observed_only:
            w = np.zeros_like(w)
            z = np.zeros_like(z)
        return w, z

    def fit_components(self, X, A, Y, Z, W):
        x = _ensure_2d(X).astype(float)
        self._x_core_dim = int(x.shape[1])
        x, a, w_nuis, z_nuis, x_final, y_res, a_res = _crossfit_summary_arrays_nc(self, x, A, Y, W, Z)
        self._forest = CausalForest(
            n_estimators=self._n_estimators,
            criterion="het",
            min_samples_leaf=self._min_samples_leaf,
            max_depth=self._forest_max_depth,
            honest=self._forest_honest,
            inference=self._forest_inference,
            fit_intercept=self._forest_fit_intercept,
            n_jobs=1,
            random_state=self._random_state,
        )
        self._forest.fit(x_final, a_res, y_res)

        self._full_nuisance = self._make_nuisance()
        self._full_nuisance.train(
            False,
            None,
            np.asarray(Y, dtype=float).ravel(),
            np.asarray(A, dtype=float).ravel(),
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def fit_oracle(self, X, A, Y, U):
        x = _ensure_2d(X).astype(float)
        u = _ensure_2d(U).astype(float)
        z_dummy = np.zeros((x.shape[0], 1), dtype=float)
        return self.fit_components(x, A, Y, z_dummy, u)

    def effect_from_components(self, X, W, Z):
        if self._forest is None or self._full_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._full_nuisance.predict_bridge_outputs(X=x, W=w_nuis, Z=z_nuis)
        x_final = _build_nc_summary_features(
            x,
            bridge,
            mode=self._summary_feature_mode,
            n_curve_knots=self._summary_curve_knots,
        )
        return self._forest.predict(x_final)

    def effect_oracle(self, X, U):
        x = _ensure_2d(X).astype(float)
        u = _ensure_2d(U).astype(float)
        z_dummy = np.zeros((x.shape[0], 1), dtype=float)
        return self.effect_from_components(x, u, z_dummy)


class TwoStageBridgeSummaryNCCausalForest(_BaseTwoStageSummaryNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class TwoStageObservedSummaryNCCausalForest(_BaseTwoStageSummaryNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("observed_only", True)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 800)
        kwargs.setdefault("h_min_samples_leaf", 5)
        super().__init__(*args, **kwargs)


class BestCurveLocalNCCausalForest(TwoStageBridgeSummaryNCCausalForest):
    """
    Low-complexity non-censored PCI model aligned with the censored BestCurveLocal design.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 800)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("summary_feature_mode", "curve_x_interact")
        kwargs.setdefault("nuisance_feature_mode", "interact")
        kwargs.setdefault("n_estimators", 400)
        kwargs.setdefault("min_samples_leaf", 30)
        kwargs.setdefault("summary_curve_knots", 3)
        super().__init__(*args, **kwargs)


class BestCurveLocalObservedNCCausalForest(TwoStageObservedSummaryNCCausalForest):
    """
    Matched no-PCI baseline for BestCurveLocalNCCausalForest.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 800)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("summary_feature_mode", "curve_x_interact")
        kwargs.setdefault("nuisance_feature_mode", "interact")
        kwargs.setdefault("n_estimators", 400)
        kwargs.setdefault("min_samples_leaf", 30)
        kwargs.setdefault("summary_curve_knots", 3)
        super().__init__(*args, **kwargs)
