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


def _proxy_summary_block(proxy: ArrayLike) -> ArrayLike:
    proxy = _ensure_2d(proxy).astype(float)
    if proxy.shape[1] == 0:
        return np.zeros((proxy.shape[0], 6), dtype=float)
    mean = np.mean(proxy, axis=1, keepdims=True)
    std = np.std(proxy, axis=1, keepdims=True)
    mean_abs = np.mean(np.abs(proxy), axis=1, keepdims=True)
    l2 = np.sqrt(np.mean(np.square(proxy), axis=1, keepdims=True))
    max_abs = np.max(np.abs(proxy), axis=1, keepdims=True)
    disp = (
        np.mean(np.abs(np.diff(proxy, axis=1)), axis=1, keepdims=True)
        if proxy.shape[1] >= 2
        else np.zeros_like(mean)
    )
    return np.hstack([mean, std, mean_abs, l2, max_abs, disp])


def _joint_proxy_summary_block(w_proxy: ArrayLike, z_proxy: ArrayLike) -> ArrayLike:
    w_proxy = _ensure_2d(w_proxy).astype(float)
    z_proxy = _ensure_2d(z_proxy).astype(float)
    cols = min(w_proxy.shape[1], z_proxy.shape[1])
    if cols <= 0:
        return np.zeros((w_proxy.shape[0], 4), dtype=float)
    w_sub = w_proxy[:, :cols]
    z_sub = z_proxy[:, :cols]
    aligned = np.mean(w_sub * z_sub, axis=1, keepdims=True)
    centered = np.mean(
        (w_sub - np.mean(w_sub, axis=1, keepdims=True))
        * (z_sub - np.mean(z_sub, axis=1, keepdims=True)),
        axis=1,
        keepdims=True,
    )
    mean_gap = np.mean(w_sub, axis=1, keepdims=True) - np.mean(z_sub, axis=1, keepdims=True)
    abs_gap = np.mean(np.abs(w_sub - z_sub), axis=1, keepdims=True)
    return np.hstack([aligned, centered, mean_gap, abs_gap])


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
    if mode == "curve_proxy_x_interact":
        curve_block = np.hstack([curve_knots, curve_mean, curve_diff, curve_span])
        w_proxy = np.asarray(bridge["w_raw"], dtype=float)
        z_proxy = np.asarray(bridge["z_raw"], dtype=float)
        proxy_block = np.hstack(
            [
                _proxy_summary_block(w_proxy),
                _proxy_summary_block(z_proxy),
                _joint_proxy_summary_block(w_proxy, z_proxy),
            ]
        )
        proxy_core = proxy_block[:, [0, 2, 6, 8, 12, 15]]
        interact_base = np.hstack(
            [q_pred, h_diff, m_pred, q_margin, balance, curve_diff, curve_span, proxy_core]
        )
        return np.hstack(
            parts + [q_margin, balance, curve_block, proxy_block, _pairwise_products(x, interact_base)]
        )
    raise ValueError(f"Unsupported non-censored summary feature mode: {mode}")


def _build_oldc3_ablation_features(
    X: ArrayLike,
    W_raw: ArrayLike,
    Z_raw: ArrayLike,
    bridge: dict[str, ArrayLike],
    *,
    include_raw_proxy: bool,
    include_bridge_stats: bool = False,
    extra_bridge_features: tuple[str, ...] = (),
) -> ArrayLike:
    x = _ensure_2d(X).astype(float)
    parts = [x]
    if include_raw_proxy:
        parts.extend([_ensure_2d(W_raw).astype(float), _ensure_2d(Z_raw).astype(float)])
    q_pred = np.asarray(bridge["q_pred"], dtype=float).reshape(-1, 1)
    h1_pred = np.asarray(bridge["h1_pred"], dtype=float).reshape(-1, 1)
    h0_pred = np.asarray(bridge["h0_pred"], dtype=float).reshape(-1, 1)
    m_pred = np.asarray(bridge["m_pred"], dtype=float).reshape(-1, 1)
    parts.extend([q_pred, h1_pred, h0_pred, m_pred])
    h_diff = h1_pred - h0_pred
    q_margin = np.abs(q_pred - 0.5)
    balance = 4.0 * q_pred * (1.0 - q_pred)
    q_centered = q_pred - 0.5
    abs_h_diff = np.abs(h_diff)
    q_logit = np.log(q_pred / (1.0 - q_pred))
    agreement = (2.0 * q_pred - 1.0) * h_diff
    feature_map = {
        "h_diff": h_diff,
        "abs_h_diff": abs_h_diff,
        "q_margin": q_margin,
        "balance": balance,
        "q_centered": q_centered,
        "q_logit": q_logit,
        "agreement": agreement,
    }
    ordered_features: list[str] = []
    if include_bridge_stats:
        ordered_features.extend(["h_diff", "q_margin", "balance"])
    ordered_features.extend(extra_bridge_features)
    seen: set[str] = set()
    for name in ordered_features:
        if name in seen:
            continue
        if name not in feature_map:
            raise ValueError(f"Unsupported bridge feature: {name}")
        parts.append(feature_map[name])
        seen.add(name)
    return np.hstack(parts)


def _build_single_pass_nc_features(
    X: ArrayLike,
    W_raw: ArrayLike,
    Z_raw: ArrayLike,
    bridge: dict[str, ArrayLike],
    *,
    mode: str,
) -> ArrayLike:
    x = _ensure_2d(X).astype(float)
    w_raw = _ensure_2d(W_raw).astype(float)
    z_raw = _ensure_2d(Z_raw).astype(float)
    q_pred = np.asarray(bridge["q_pred"], dtype=float).reshape(-1, 1)
    h1_pred = np.asarray(bridge["h1_pred"], dtype=float).reshape(-1, 1)
    h0_pred = np.asarray(bridge["h0_pred"], dtype=float).reshape(-1, 1)
    m_pred = np.asarray(bridge["m_pred"], dtype=float).reshape(-1, 1)
    h_diff = h1_pred - h0_pred
    q_margin = np.abs(q_pred - 0.5)
    balance = 4.0 * q_pred * (1.0 - q_pred)
    q_centered = q_pred - 0.5
    agreement = (2.0 * q_pred - 1.0) * h_diff

    if mode == "aug_full":
        return np.hstack([x, w_raw, z_raw, q_pred, h1_pred, h0_pred, m_pred])
    if mode == "aug_compact":
        return np.hstack([x, w_raw, z_raw, q_pred, m_pred, h_diff])
    if mode == "aug_compact_stats":
        return np.hstack([x, w_raw, z_raw, q_pred, m_pred, h_diff, q_margin, balance])
    if mode == "aug_compact_qcenter_agreement":
        return np.hstack([x, w_raw, z_raw, q_pred, m_pred, h_diff, q_centered, agreement])
    if mode == "aug_noq":
        return np.hstack([x, w_raw, z_raw, m_pred, h_diff])
    if mode == "summary_compact":
        return np.hstack([x, q_pred, m_pred, h_diff])
    if mode == "summary_compact_stats":
        return np.hstack([x, q_pred, m_pred, h_diff, q_margin, balance])
    if mode == "summary_compact_qcenter_agreement":
        return np.hstack([x, q_pred, m_pred, h_diff, q_centered, agreement])
    raise ValueError(f"Unsupported single-pass non-censored feature mode: {mode}")


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
        x_base = X if self._duplicate_proxies_in_nuisance else X_core
        base = np.column_stack([x_base, W, Z])
        if self._nuisance_feature_mode == "broad_dup":
            XZ = np.column_stack([base, Z])
            XW = np.column_stack([base, W])
            return X_core, (W, Z), XZ, XW
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

    def _predict_bridge_components(self, *, X=None, W=None, Z=None):
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
    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        A = np.asarray(T).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        q_pred, h1_pred, h0_pred, m_pred = self._predict_bridge_components(X=X, W=W, Z=Z)
        y_res = Y - m_pred
        lo, hi = np.percentile(y_res, self._y_res_clip_percentiles)
        y_res = np.clip(y_res, lo, hi)
        a_res = (A - q_pred).reshape(-1, 1)
        return y_res, a_res, q_pred, h1_pred, h0_pred, m_pred


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


class _BridgeFeatureModelFinal:
    def __init__(self, base_model_final, *, feature_mode: str, raw_proxy_supplier=None):
        self._base_model_final = base_model_final
        self._feature_mode = feature_mode
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
        x_final = _build_single_pass_nc_features(
            X,
            w_for_final,
            z_for_final,
            bridge,
            mode=self._feature_mode,
        )
        return x_final, (y_res, a_res)

    def fit(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, freq_weight=None, sample_var=None, groups=None):
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

    def score(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, groups=None, scoring="mean_squared_error"):
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


class SinglePassBridgeFeatureNCCausalForestDML(MildShrinkNCCausalForestDML):
    def __init__(self, *, final_feature_mode: str, **kwargs):
        self._final_feature_mode = final_feature_mode
        self._raw_w_for_final = None
        self._raw_z_for_final = None
        super().__init__(**kwargs)

    def _raw_proxy_for_final(self, *, X=None, W=None, Z=None):
        if X is None:
            return W, Z
        if self._raw_w_for_final is None or self._raw_z_for_final is None:
            return W, Z
        if len(np.asarray(X)) != len(self._raw_w_for_final):
            return W, Z
        return self._raw_w_for_final, self._raw_z_for_final

    def _gen_ortho_learner_model_nuisance(self):
        return _BridgeOutputNCNuisance(
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

    def _gen_ortho_learner_model_final(self):
        return _BridgeFeatureModelFinal(
            super()._gen_ortho_learner_model_final(),
            feature_mode=self._final_feature_mode,
            raw_proxy_supplier=self._raw_proxy_for_final,
        )

    def effect_on_final_features(self, X_final):
        return np.asarray(self._ortho_learner_model_final.predict(X_final), dtype=float)

    def training_x_final(self):
        return self._ortho_learner_model_final.training_x_final()


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
        bridge["w_raw"] = w_nuis[test_idx]
        bridge["z_raw"] = z_nuis[test_idx]
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


def _crossfit_oldc3_ablation_arrays_nc(owner, X, A, Y, W, Z):
    x = _ensure_2d(X).astype(float)
    y = np.asarray(Y, dtype=float).ravel()
    a = np.asarray(A, dtype=float).ravel()
    raw_w = _ensure_2d(W).astype(float)
    raw_z = _ensure_2d(Z).astype(float)
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
        x_final_fold = _build_oldc3_ablation_features(
            x[test_idx],
            raw_w[test_idx],
            raw_z[test_idx],
            bridge,
            include_raw_proxy=owner._include_raw_proxy,
            include_bridge_stats=getattr(owner, "_include_bridge_stats", False),
            extra_bridge_features=getattr(owner, "_extra_bridge_features", ()),
        )
        if x_final is None:
            x_final = np.empty((len(y), x_final_fold.shape[1]), dtype=float)
        x_final[test_idx] = x_final_fold
        y_res[test_idx] = np.asarray(y_res_fold, dtype=float).ravel()
        a_res[test_idx] = np.asarray(a_res_fold, dtype=float).ravel()

    return x, raw_w, raw_z, w_nuis, z_nuis, x_final, y_res, a_res


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
        bridge["w_raw"] = w_nuis
        bridge["z_raw"] = z_nuis
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


class _BaseOldC3FeatureGRFNCCausalForest:
    def __init__(
        self,
        *,
        include_raw_proxy: bool,
        observed_only: bool = False,
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=42,
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
        nuisance_feature_mode="dup",
        forest_max_depth=None,
        forest_honest=True,
        forest_inference=True,
        forest_fit_intercept=True,
    ):
        self._include_raw_proxy = bool(include_raw_proxy)
        self._observed_only = bool(observed_only)
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._cv = int(cv)
        self._random_state = int(random_state)
        self._nuisance_feature_mode = nuisance_feature_mode
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
        self._forest_max_depth = forest_max_depth
        self._forest_honest = bool(forest_honest)
        self._forest_inference = bool(forest_inference)
        self._forest_fit_intercept = bool(forest_fit_intercept)
        self._forest = None
        self._full_nuisance = None
        self._x_core_dim = None

    def _prepare_nuisance_inputs(self, W, Z):
        w = _ensure_2d(W).astype(float)
        z = _ensure_2d(Z).astype(float)
        if self._observed_only:
            w = np.zeros_like(w)
            z = np.zeros_like(z)
        return w, z

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
            oracle=False,
            use_true_q=False,
            use_true_h=False,
            q_true_fn=None,
            h_true_fn=None,
        )

    def fit_components(self, X, A, Y, Z, W):
        x = _ensure_2d(X).astype(float)
        self._x_core_dim = int(x.shape[1])
        x, raw_w, raw_z, w_nuis, z_nuis, x_final, y_res, a_res = _crossfit_oldc3_ablation_arrays_nc(
            self,
            x,
            A,
            Y,
            W,
            Z,
        )
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

    def effect_from_components(self, X, W, Z):
        if self._forest is None or self._full_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._full_nuisance.predict_bridge_outputs(X=x, W=w_nuis, Z=z_nuis)
        x_final = _build_oldc3_ablation_features(
            x,
            raw_w,
            raw_z,
            bridge,
            include_raw_proxy=self._include_raw_proxy,
        )
        return self._forest.predict(x_final)


class OldC3SummaryGRFNCCausalForest(_BaseOldC3FeatureGRFNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", False)
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class OldC3SummaryObservedGRFNCCausalForest(_BaseOldC3FeatureGRFNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", False)
        kwargs.setdefault("observed_only", True)
        super().__init__(*args, **kwargs)


class OldC3AugmentedGRFNCCausalForest(_BaseOldC3FeatureGRFNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class OldC3AugmentedObservedGRFNCCausalForest(_BaseOldC3FeatureGRFNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("observed_only", True)
        super().__init__(*args, **kwargs)


class _BaseOldC3FeatureDMLNCausalForest:
    def __init__(
        self,
        *,
        include_raw_proxy: bool,
        include_bridge_stats: bool = False,
        observed_only: bool = False,
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=42,
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
        nuisance_feature_mode="dup",
    ):
        self._include_raw_proxy = bool(include_raw_proxy)
        self._include_bridge_stats = bool(include_bridge_stats)
        self._observed_only = bool(observed_only)
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._cv = int(cv)
        self._random_state = int(random_state)
        self._q_kind = q_kind
        self._q_trees = int(q_trees)
        self._q_leaf = int(q_leaf)
        self._q_clip = q_clip
        self._h_kind = h_kind
        self._y_clip_quantile = y_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._h_n_estimators = int(h_n_estimators)
        self._h_min_samples_leaf = int(h_min_samples_leaf)
        self._n_jobs = int(n_jobs)
        self._nuisance_feature_mode = nuisance_feature_mode
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
        self._feature_nuisance = None
        self._dml_model = None
        self._x_core_dim = None

    def _prepare_nuisance_inputs(self, W, Z):
        w = _ensure_2d(W).astype(float)
        z = _ensure_2d(Z).astype(float)
        if self._observed_only:
            w = np.zeros_like(w)
            z = np.zeros_like(z)
        return w, z

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
            oracle=False,
            use_true_q=False,
            use_true_h=False,
            q_true_fn=None,
            h_true_fn=None,
        )

    def fit_components(self, X, A, Y, Z, W):
        x = _ensure_2d(X).astype(float)
        self._x_core_dim = int(x.shape[1])
        _, raw_w, raw_z, w_nuis, z_nuis, x_final, _, _ = _crossfit_oldc3_ablation_arrays_nc(
            self,
            x,
            A,
            Y,
            W,
            Z,
        )
        self._dml_model = MildShrinkNCCausalForestDML(
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
            n_jobs=self._n_jobs,
            x_core_dim=self._x_core_dim,
            duplicate_proxies_in_nuisance=False,
            nuisance_feature_mode=self._nuisance_feature_mode,
        )
        self._dml_model.fit_nc(x_final, A, Y, z_nuis, w_nuis)

        self._feature_nuisance = self._make_nuisance()
        self._feature_nuisance.train(
            False,
            None,
            np.asarray(Y, dtype=float).ravel(),
            np.asarray(A, dtype=float).ravel(),
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._dml_model is None or self._feature_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._feature_nuisance.predict_bridge_outputs(X=x, W=w_nuis, Z=z_nuis)
        x_final = _build_oldc3_ablation_features(
            x,
            raw_w,
            raw_z,
            bridge,
            include_raw_proxy=self._include_raw_proxy,
            include_bridge_stats=self._include_bridge_stats,
        )
        return self._dml_model.effect(x_final)


class _BaseSinglePassBridgeFeatureNCCausalForest:
    def __init__(
        self,
        *,
        final_feature_mode: str,
        prediction_nuisance_mode="full_refit",
        observed_only=False,
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
        h_n_estimators=800,
        h_min_samples_leaf=5,
        n_jobs=1,
        nuisance_feature_mode="dup",
    ):
        self._final_feature_mode = final_feature_mode
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._cv = int(cv)
        self._random_state = int(random_state)
        self._prediction_nuisance_mode = prediction_nuisance_mode
        self._observed_only = bool(observed_only)
        self._q_kind = q_kind
        self._q_trees = int(q_trees)
        self._q_leaf = int(q_leaf)
        self._q_clip = q_clip
        self._h_kind = h_kind
        self._y_clip_quantile = y_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._h_n_estimators = int(h_n_estimators)
        self._h_min_samples_leaf = int(h_min_samples_leaf)
        self._n_jobs = int(n_jobs)
        self._nuisance_feature_mode = nuisance_feature_mode
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
        self._feature_nuisance = None
        self._dml_model = None
        self._x_core_dim = None
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
            oracle=False,
            use_true_q=False,
            use_true_h=False,
            q_true_fn=None,
            h_true_fn=None,
        )

    def fit_components(self, X, A, Y, Z, W):
        x = _ensure_2d(X).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        self._x_core_dim = int(x.shape[1])

        self._dml_model = SinglePassBridgeFeatureNCCausalForestDML(
            final_feature_mode=self._final_feature_mode,
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
            n_jobs=self._n_jobs,
            x_core_dim=self._x_core_dim,
            duplicate_proxies_in_nuisance=True,
            nuisance_feature_mode=self._nuisance_feature_mode,
        )
        self._dml_model._raw_w_for_final = raw_w
        self._dml_model._raw_z_for_final = raw_z
        self._dml_model.fit_nc(x, A, Y, z_nuis, w_nuis)
        self._train_x = np.asarray(x, dtype=float).copy()
        self._train_w = np.asarray(raw_w, dtype=float).copy()
        self._train_z = np.asarray(raw_z, dtype=float).copy()
        self._train_x_final = self._dml_model.training_x_final()

        if self._prediction_nuisance_mode == "full_refit":
            self._feature_nuisance = self._make_nuisance()
            self._feature_nuisance.train(
                False,
                None,
                np.asarray(Y, dtype=float).ravel(),
                np.asarray(A, dtype=float).ravel(),
                X=x,
                W=w_nuis,
                Z=z_nuis,
            )
        elif self._prediction_nuisance_mode not in {"fold_ensemble", "cached_oof"}:
            raise ValueError(f"Unsupported prediction nuisance mode: {self._prediction_nuisance_mode}")
        return self

    def _predict_bridge_from_fold_ensemble(self, X, W, Z):
        model_groups = getattr(self._dml_model, "_models_nuisance", None)
        if not model_groups:
            raise RuntimeError("Fold nuisance models are unavailable for ensemble prediction.")
        fold_models = []
        for group in model_groups:
            if isinstance(group, list):
                fold_models.extend(group)
            else:
                fold_models.append(group)
        if not fold_models:
            raise RuntimeError("No fitted fold nuisance models were found.")
        preds = [model.predict_bridge_outputs(X=X, W=W, Z=Z) for model in fold_models]
        keys = preds[0].keys()
        return {key: np.mean(np.column_stack([np.asarray(pred[key], dtype=float).ravel() for pred in preds]), axis=1) for key in keys}

    def _matches_training_data(self, x, w, z):
        if self._train_x is None or self._train_w is None or self._train_z is None:
            return False
        return (
            x.shape == self._train_x.shape
            and w.shape == self._train_w.shape
            and z.shape == self._train_z.shape
            and np.array_equal(x, self._train_x)
            and np.array_equal(w, self._train_w)
            and np.array_equal(z, self._train_z)
        )

    def effect_from_components(self, X, W, Z):
        if self._dml_model is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        if self._prediction_nuisance_mode == "full_refit":
            if self._feature_nuisance is None:
                raise RuntimeError("Full nuisance retrain is unavailable.")
            bridge = self._feature_nuisance.predict_bridge_outputs(X=x, W=w_nuis, Z=z_nuis)
            x_final = _build_single_pass_nc_features(
                x,
                raw_w,
                raw_z,
                bridge,
                mode=self._final_feature_mode,
            )
        elif self._prediction_nuisance_mode == "fold_ensemble":
            bridge = self._predict_bridge_from_fold_ensemble(x, w_nuis, z_nuis)
            x_final = _build_single_pass_nc_features(
                x,
                raw_w,
                raw_z,
                bridge,
                mode=self._final_feature_mode,
            )
        elif self._prediction_nuisance_mode == "cached_oof":
            if self._matches_training_data(x, raw_w, raw_z):
                if self._train_x_final is None:
                    raise RuntimeError("Cached training final features are unavailable.")
                x_final = np.asarray(self._train_x_final, dtype=float)
            else:
                bridge = self._predict_bridge_from_fold_ensemble(x, w_nuis, z_nuis)
                x_final = _build_single_pass_nc_features(
                    x,
                    raw_w,
                    raw_z,
                    bridge,
                    mode=self._final_feature_mode,
                )
        else:
            raise ValueError(f"Unsupported prediction nuisance mode: {self._prediction_nuisance_mode}")
        return self._dml_model.effect_on_final_features(x_final)


class SinglePassAugmentedFullNCCausalForest(_BaseSinglePassBridgeFeatureNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "aug_full")
        super().__init__(*args, **kwargs)


class SinglePassAugmentedCompactNCCausalForest(_BaseSinglePassBridgeFeatureNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "aug_compact")
        super().__init__(*args, **kwargs)


class SinglePassAugmentedCompactStatsNCCausalForest(_BaseSinglePassBridgeFeatureNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "aug_compact_stats")
        super().__init__(*args, **kwargs)


class SinglePassAugmentedNoQNCCausalForest(_BaseSinglePassBridgeFeatureNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "aug_noq")
        super().__init__(*args, **kwargs)


class SinglePassSummaryCompactNCCausalForest(_BaseSinglePassBridgeFeatureNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "summary_compact")
        super().__init__(*args, **kwargs)


class SinglePassSummaryCompactStatsNCCausalForest(_BaseSinglePassBridgeFeatureNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "summary_compact_stats")
        super().__init__(*args, **kwargs)


class OldC3SummaryDMLNCCausalForest(_BaseOldC3FeatureDMLNCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", False)
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class OldC3SummaryObservedDMLNCCausalForest(_BaseOldC3FeatureDMLNCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", False)
        kwargs.setdefault("observed_only", True)
        super().__init__(*args, **kwargs)


class OldC3NCCausalForest:
    def __init__(
        self,
        *,
        observed_only: bool = False,
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
        h_n_estimators=1200,
        h_min_samples_leaf=3,
        n_jobs=1,
        nuisance_feature_mode="dup",
    ):
        self._observed_only = bool(observed_only)
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._cv = int(cv)
        self._random_state = int(random_state)
        self._q_kind = q_kind
        self._q_trees = int(q_trees)
        self._q_leaf = int(q_leaf)
        self._q_clip = q_clip
        self._h_kind = h_kind
        self._y_clip_quantile = y_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._h_n_estimators = int(h_n_estimators)
        self._h_min_samples_leaf = int(h_min_samples_leaf)
        self._n_jobs = int(n_jobs)
        self._nuisance_feature_mode = nuisance_feature_mode
        self._model = None
        self._x_core_dim = None

    def _prepare_nuisance_inputs(self, W, Z):
        w = _ensure_2d(W).astype(float)
        z = _ensure_2d(Z).astype(float)
        if self._observed_only:
            w = np.zeros_like(w)
            z = np.zeros_like(z)
        return w, z

    def fit_components(self, X, A, Y, Z, W):
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        x_full = MildShrinkNCCausalForestDML.stack_final_features(x, raw_w, raw_z)
        self._x_core_dim = int(x.shape[1])
        self._model = MildShrinkNCCausalForestDML(
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
            n_jobs=self._n_jobs,
            x_core_dim=self._x_core_dim,
            duplicate_proxies_in_nuisance=True,
            nuisance_feature_mode=self._nuisance_feature_mode,
        )
        self._model.fit_nc(x_full, A, Y, z_nuis, w_nuis)
        return self

    def effect_from_components(self, X, W, Z):
        if self._model is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_full = MildShrinkNCCausalForestDML.stack_final_features(x, raw_w, raw_z)
        return self._model.effect(x_full)


class OldC3ObservedNCCausalForest(OldC3NCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("observed_only", True)
        super().__init__(*args, **kwargs)


class PureB2NCCausalForest:
    def __init__(
        self,
        *,
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=42,
    ):
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._cv = int(cv)
        self._random_state = int(random_state)
        self._baseline = None

    def fit_components(self, X, A, Y, Z, W):
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_full = MildShrinkNCCausalForestDML.stack_final_features(x, raw_w, raw_z)
        self._baseline = BaselineCausalForestDML(
            n_estimators=self._n_estimators,
            min_samples_leaf=self._min_samples_leaf,
            cv=self._cv,
            random_state=self._random_state,
        )
        self._baseline.fit_baseline(x_full, A, Y)
        return self

    def effect_from_components(self, X, W, Z):
        if self._baseline is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_full = MildShrinkNCCausalForestDML.stack_final_features(x, raw_w, raw_z)
        return self._baseline.effect(x_full)


class StrictEconmlXWZNCCausalForest:
    def __init__(
        self,
        *,
        model_y="auto",
        model_t="auto",
        cv=2,
        discrete_treatment=True,
        criterion="het",
        random_state=42,
        **kwargs,
    ):
        self._model_y = model_y
        self._model_t = model_t
        self._cv = cv
        self._discrete_treatment = bool(discrete_treatment)
        self._criterion = criterion
        self._random_state = random_state
        self._extra_kwargs = dict(kwargs)
        self._model = None

    def fit_components(self, X, A, Y, Z, W):
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_full = MildShrinkNCCausalForestDML.stack_final_features(x, raw_w, raw_z)
        self._model = CausalForestDML(
            model_y=self._model_y,
            model_t=self._model_t,
            cv=self._cv,
            discrete_treatment=self._discrete_treatment,
            criterion=self._criterion,
            random_state=self._random_state,
            **self._extra_kwargs,
        )
        self._model.fit(
            Y=np.asarray(Y, dtype=float).ravel(),
            T=np.asarray(A).ravel(),
            X=x_full,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._model is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_full = MildShrinkNCCausalForestDML.stack_final_features(x, raw_w, raw_z)
        return self._model.effect(x_full)


class _BaseB2PCINCCausalForest:
    def __init__(
        self,
        *,
        include_raw_proxy: bool = True,
        include_bridge_stats: bool = False,
        extra_bridge_features: tuple[str, ...] = (),
        observed_only: bool = False,
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
        h_n_estimators=1200,
        h_min_samples_leaf=3,
        n_jobs=1,
        nuisance_feature_mode="dup",
    ):
        self._include_raw_proxy = bool(include_raw_proxy)
        self._include_bridge_stats = bool(include_bridge_stats)
        self._extra_bridge_features = tuple(extra_bridge_features)
        self._observed_only = bool(observed_only)
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._cv = int(cv)
        self._random_state = int(random_state)
        self._q_kind = q_kind
        self._q_trees = int(q_trees)
        self._q_leaf = int(q_leaf)
        self._q_clip = q_clip
        self._h_kind = h_kind
        self._y_clip_quantile = y_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._h_n_estimators = int(h_n_estimators)
        self._h_min_samples_leaf = int(h_min_samples_leaf)
        self._n_jobs = int(n_jobs)
        self._nuisance_feature_mode = nuisance_feature_mode
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
        self._feature_nuisance = None
        self._baseline = None
        self._x_core_dim = None

    def _prepare_nuisance_inputs(self, W, Z):
        w = _ensure_2d(W).astype(float)
        z = _ensure_2d(Z).astype(float)
        if self._observed_only:
            w = np.zeros_like(w)
            z = np.zeros_like(z)
        return w, z

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
            oracle=False,
            use_true_q=False,
            use_true_h=False,
            q_true_fn=None,
            h_true_fn=None,
        )

    def fit_components(self, X, A, Y, Z, W):
        x = _ensure_2d(X).astype(float)
        self._x_core_dim = int(x.shape[1])
        _, raw_w, raw_z, w_nuis, z_nuis, x_final, _, _ = _crossfit_oldc3_ablation_arrays_nc(
            self,
            x,
            A,
            Y,
            W,
            Z,
        )
        self._baseline = BaselineCausalForestDML(
            n_estimators=self._n_estimators,
            min_samples_leaf=self._min_samples_leaf,
            cv=self._cv,
            random_state=self._random_state,
        )
        self._baseline.fit_baseline(x_final, A, Y)

        self._feature_nuisance = self._make_nuisance()
        self._feature_nuisance.train(
            False,
            None,
            np.asarray(Y, dtype=float).ravel(),
            np.asarray(A, dtype=float).ravel(),
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._baseline is None or self._feature_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._feature_nuisance.predict_bridge_outputs(X=x, W=w_nuis, Z=z_nuis)
        x_final = _build_oldc3_ablation_features(
            x,
            raw_w,
            raw_z,
            bridge,
            include_raw_proxy=self._include_raw_proxy,
            include_bridge_stats=self._include_bridge_stats,
            extra_bridge_features=self._extra_bridge_features,
        )
        return self._baseline.effect(x_final)


class B2PCINCCausalForest(_BaseB2PCINCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", False)
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "dup")
        super().__init__(*args, **kwargs)


class B2ObservedPCINCCausalForest(_BaseB2PCINCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", False)
        kwargs.setdefault("observed_only", True)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "dup")
        super().__init__(*args, **kwargs)


class B2SummaryNCCausalForest(B2PCINCCausalForest):
    pass


class B2SummaryObservedNCCausalForest(B2ObservedPCINCCausalForest):
    pass


class B2SummaryBroadDupNCCausalForest(_BaseB2PCINCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", False)
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class B2SummaryRichNCCausalForest(_BaseB2PCINCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", True)
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "dup")
        super().__init__(*args, **kwargs)


class B2SummaryBroadDupRichNCCausalForest(_BaseB2PCINCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", True)
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class B2SummaryFeatureComboNCCausalForest(_BaseB2PCINCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", False)
        kwargs.setdefault("extra_bridge_features", ())
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class B2SummaryProxyDupGRFNCCausalForest(_BaseOldC3FeatureGRFNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class UnifiedB2SumBaselineNCCausalForest(B2SummaryBroadDupNCCausalForest):
    """Matched non-censored B2Sum baseline family."""


class UnifiedB2SumMildShrinkNCCausalForest(_BaseOldC3FeatureDMLNCausalForest):
    """Matched non-censored B2Sum mild-shrink family."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", False)
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class UnifiedB2SumSinglePassBaselineNCCausalForest(_BaseSinglePassBridgeFeatureNCCausalForest):
    """Single-pass baseline-style B2Sum with structured broad proxy duplication."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "aug_full")
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "rf")
        kwargs.setdefault("q_trees", 100)
        kwargs.setdefault("q_leaf", 20)
        kwargs.setdefault("h_kind", "rf")
        kwargs.setdefault("h_n_estimators", 100)
        kwargs.setdefault("h_min_samples_leaf", 20)
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class FinalModelNCCausalForest(_BaseSinglePassBridgeFeatureNCCausalForest):
    """Finalized non-censored single-pass B2Sum model."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "aug_full")
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 600)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("q_clip", 0.02)
        kwargs.setdefault("y_clip_quantile", 0.99)
        kwargs.setdefault("y_res_clip_percentiles", (1.0, 99.0))
        kwargs.setdefault("n_estimators", 200)
        kwargs.setdefault("min_samples_leaf", 20)
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        kwargs.setdefault("n_jobs", 1)
        super().__init__(*args, **kwargs)


class OldC3AugmentedDMLNCCausalForest(_BaseOldC3FeatureDMLNCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", False)
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class OldC3AugmentedObservedDMLNCCausalForest(_BaseOldC3FeatureDMLNCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", False)
        kwargs.setdefault("observed_only", True)
        super().__init__(*args, **kwargs)


class OldC3AugmentedDMLBridgeStatsNCCausalForest(_BaseOldC3FeatureDMLNCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", True)
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class OldC3AugmentedDMLPoly2QNCCausalForest(_BaseOldC3FeatureDMLNCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", False)
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "poly2")
        super().__init__(*args, **kwargs)


class OldC3AugmentedDMLExtraHNCCausalForest(_BaseOldC3FeatureDMLNCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", False)
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 800)
        kwargs.setdefault("h_min_samples_leaf", 5)
        super().__init__(*args, **kwargs)


class OldC3AugmentedDMLExtraH1200NCCausalForest(_BaseOldC3FeatureDMLNCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", False)
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        super().__init__(*args, **kwargs)


class AugDMLNCCausalForest(_BaseOldC3FeatureDMLNCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", False)
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "dup")
        super().__init__(*args, **kwargs)


class AugDMLObservedNCCausalForest(_BaseOldC3FeatureDMLNCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", False)
        kwargs.setdefault("observed_only", True)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "dup")
        super().__init__(*args, **kwargs)


class BroadAugDMLNCCausalForest(_BaseOldC3FeatureDMLNCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", False)
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class BroadAugDMLObservedNCCausalForest(_BaseOldC3FeatureDMLNCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", False)
        kwargs.setdefault("observed_only", True)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class AugSPNCCausalForest(_BaseSinglePassBridgeFeatureNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "aug_full")
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "dup")
        super().__init__(*args, **kwargs)


class AugSPObservedNCCausalForest(_BaseSinglePassBridgeFeatureNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "aug_full")
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("observed_only", True)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "dup")
        super().__init__(*args, **kwargs)


class BroadAugSPNCCausalForest(_BaseSinglePassBridgeFeatureNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "aug_full")
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class BroadAugSPObservedNCCausalForest(_BaseSinglePassBridgeFeatureNCCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("final_feature_mode", "aug_full")
        kwargs.setdefault("prediction_nuisance_mode", "full_refit")
        kwargs.setdefault("observed_only", True)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 1200)
        kwargs.setdefault("h_min_samples_leaf", 3)
        kwargs.setdefault("nuisance_feature_mode", "broad_dup")
        super().__init__(*args, **kwargs)


class UnifiedBroadAugSPBaselineNCCausalForest(UnifiedB2SumBaselineNCCausalForest):
    """
    Matched baseline-family BroadAugSP alias.

    In the non-censored setting, BroadAugSP's aug_full representation collapses to
    the same final input as broad-dup B2Sum baseline:
      [X, W, Z, q, h1, h0, m].
    """


class UnifiedBroadAugSPMildShrinkNCCausalForest(BroadAugSPNCCausalForest):
    """Matched mild-shrink BroadAugSP family."""


class OldC3RawNCCausalForest(OldC3NCCausalForest):
    pass


class OldC3RawObservedNCCausalForest(OldC3ObservedNCCausalForest):
    pass


class OldC3AugmentedDMLBridgeStatsPoly2QExtraHNCCausalForest(_BaseOldC3FeatureDMLNCausalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("include_raw_proxy", True)
        kwargs.setdefault("include_bridge_stats", True)
        kwargs.setdefault("observed_only", False)
        kwargs.setdefault("q_kind", "poly2")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 800)
        kwargs.setdefault("h_min_samples_leaf", 5)
        super().__init__(*args, **kwargs)


class HybridProxyNCCausalForest(TwoStageBridgeSummaryNCCausalForest):
    """
    Bridge-summary forest with compact retained proxy summaries for high-dimensional regimes.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 800)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("summary_feature_mode", "curve_proxy_x_interact")
        kwargs.setdefault("nuisance_feature_mode", "interact")
        kwargs.setdefault("n_estimators", 400)
        kwargs.setdefault("min_samples_leaf", 40)
        kwargs.setdefault("summary_curve_knots", 5)
        super().__init__(*args, **kwargs)
