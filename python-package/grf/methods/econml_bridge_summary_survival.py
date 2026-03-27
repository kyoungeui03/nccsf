from __future__ import annotations

import numpy as np
from econml.grf import CausalForest
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from .econml_mild_shrink import (
    _MildShrinkNCSurvivalNuisance,
    _ensure_2d,
    _pairwise_products,
    make_h_model,
    make_q_model,
)


def _select_curve_knots(curve, n_knots=5, drop_last=False):
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2:
        raise ValueError("curve must have shape (n_samples, n_grid).")
    n_grid = curve.shape[1]
    if n_grid == 0:
        return np.zeros((curve.shape[0], n_knots), dtype=float)
    max_idx = max(n_grid - 2, 0) if drop_last and n_grid >= 2 else n_grid - 1
    knot_idx = np.linspace(0, max_idx, num=min(n_knots, max_idx + 1)).round().astype(int)
    knot_idx = np.unique(knot_idx)
    selected = curve[:, knot_idx]
    if selected.shape[1] < n_knots:
        pad = np.repeat(selected[:, [-1]], n_knots - selected.shape[1], axis=1)
        selected = np.hstack([selected, pad])
    return selected


def _curve_mean_and_diff(curve, *, drop_last=True, n_knots=5):
    selected = _select_curve_knots(curve, drop_last=drop_last, n_knots=n_knots)
    diffs = np.diff(selected, axis=1)
    mean = 0.5 * (selected[:, 1:] + selected[:, :-1]) if selected.shape[1] >= 2 else selected
    return selected, mean, diffs


def _block_summary_stats(block):
    block = np.asarray(block, dtype=float)
    if block.ndim != 2:
        raise ValueError("block must have shape (n_samples, n_features).")
    if block.shape[1] == 0:
        return np.zeros((block.shape[0], 6), dtype=float)
    first = block[:, [0]]
    last = block[:, [-1]]
    mean = np.mean(block, axis=1, keepdims=True)
    std = np.std(block, axis=1, keepdims=True)
    span = last - first
    if block.shape[1] >= 2:
        slope = np.mean(np.diff(block, axis=1), axis=1, keepdims=True)
    else:
        slope = np.zeros_like(mean)
    energy = np.mean(np.abs(block), axis=1, keepdims=True)
    return np.hstack([mean, std, span, slope, energy, last])


def _proxy_summary_block(proxy):
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


def _joint_proxy_summary_block(w_proxy, z_proxy):
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


def _curve_rmst_basis_stats(block, *, n_segments=3):
    block = np.asarray(block, dtype=float)
    if block.ndim != 2:
        raise ValueError("block must have shape (n_samples, n_features).")
    n, k = block.shape
    if k == 0:
        return np.zeros((n, 10), dtype=float)

    x_grid = np.linspace(0.0, 1.0, num=k)

    def _integral(values, grid):
        if values.shape[1] <= 1:
            return values.mean(axis=1, keepdims=True)
        delta = np.diff(grid).reshape(1, -1)
        mids = 0.5 * (values[:, 1:] + values[:, :-1])
        return np.sum(mids * delta, axis=1, keepdims=True)

    overall_auc = _integral(block, x_grid)
    head = block[:, [0]]
    mid = block[:, [k // 2]]
    tail = block[:, [-1]]
    span = tail - head

    boundaries = np.linspace(0, k, n_segments + 1).round().astype(int)
    seg_aucs = []
    seg_slopes = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start <= 0:
            seg_aucs.append(np.zeros((n, 1), dtype=float))
            seg_slopes.append(np.zeros((n, 1), dtype=float))
            continue
        seg = block[:, start:end]
        seg_grid = x_grid[start:end]
        if seg.shape[1] == 1:
            seg_aucs.append(seg.copy())
            seg_slopes.append(np.zeros((n, 1), dtype=float))
        else:
            seg_aucs.append(_integral(seg, seg_grid))
            seg_slopes.append(np.mean(np.diff(seg, axis=1), axis=1, keepdims=True))

    if k >= 3:
        curvature = np.mean(np.abs(np.diff(block, n=2, axis=1)), axis=1, keepdims=True)
    else:
        curvature = np.zeros((n, 1), dtype=float)

    return np.hstack([overall_auc, head, mid, tail, span, *seg_aucs, *seg_slopes, curvature])


def _make_final_regressor(
    kind="extra",
    *,
    random_state=42,
    n_estimators=1200,
    min_samples_leaf=5,
    n_jobs=1,
    max_depth=None,
    max_features=1.0,
    learning_rate=0.05,
):
    kind = str(kind)
    if kind == "extra":
        return make_h_model(
            "extra",
            random_state=random_state,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
        ).set_params(max_depth=max_depth, max_features=max_features)
    if kind == "rf":
        return make_h_model(
            "rf",
            random_state=random_state,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
        ).set_params(max_depth=max_depth, max_features=max_features)
    if kind == "hgb":
        return make_h_model(
            "hgb",
            random_state=random_state,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
        ).set_params(max_depth=max_depth)
    if kind == "hgb_abs":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
    if kind == "gb_huber":
        return GradientBoostingRegressor(
            loss="huber",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth or 3,
            random_state=random_state,
        )
    if kind == "ridge":
        return Ridge(alpha=1.0)
    raise ValueError(f"Unsupported final regressor kind: {kind}")


def _build_regime_descriptors(bridge, *, n_curve_knots=5):
    q_pred = np.asarray(bridge["q_pred"], dtype=float).reshape(-1, 1)
    h1_pred = np.asarray(bridge["h1_pred"], dtype=float).reshape(-1, 1)
    h0_pred = np.asarray(bridge["h0_pred"], dtype=float).reshape(-1, 1)
    h_diff = h1_pred - h0_pred
    surv_diff = np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1)

    q_curve_1 = _select_curve_knots(bridge["qhat1_curve"], drop_last=True, n_knots=n_curve_knots)
    q_curve_0 = _select_curve_knots(bridge["qhat0_curve"], drop_last=True, n_knots=n_curve_knots)
    q_curve_diff = q_curve_1 - q_curve_0
    q_curve_mean = 0.5 * (q_curve_1 + q_curve_0)

    s_curve_1 = _select_curve_knots(bridge["s1_curve"], drop_last=True, n_knots=n_curve_knots)
    s_curve_0 = _select_curve_knots(bridge["s0_curve"], drop_last=True, n_knots=n_curve_knots)
    s_curve_diff = s_curve_1 - s_curve_0
    s_curve_mean = 0.5 * (s_curve_1 + s_curve_0)

    _, c_curve_mean, c_curve_diff = _curve_mean_and_diff(bridge["c_curve"], drop_last=True, n_knots=n_curve_knots)

    h_abs = np.abs(h_diff)
    surv_abs = np.abs(surv_diff)
    q_margin = np.abs(q_pred - 0.5)
    balance = 4.0 * q_pred * (1.0 - q_pred)

    q_curve_energy = np.mean(np.abs(q_curve_diff), axis=1, keepdims=True)
    q_curve_dispersion = np.std(q_curve_diff, axis=1, keepdims=True)
    q_curve_level = np.mean(np.abs(q_curve_mean), axis=1, keepdims=True)

    s_curve_energy = np.mean(np.abs(s_curve_diff), axis=1, keepdims=True)
    s_curve_dispersion = np.std(s_curve_diff, axis=1, keepdims=True)
    s_curve_level = np.mean(np.abs(s_curve_mean), axis=1, keepdims=True)
    if s_curve_mean.shape[1] >= 2:
        s_curve_slope = np.mean(np.abs(np.diff(s_curve_mean, axis=1)), axis=1, keepdims=True)
    else:
        s_curve_slope = s_curve_energy

    if c_curve_mean.size:
        censor_level = np.mean(c_curve_mean, axis=1, keepdims=True)
    else:
        censor_level = np.zeros_like(q_pred)
    if c_curve_diff.size:
        censor_slope = np.mean(np.abs(c_curve_diff), axis=1, keepdims=True)
        censor_dispersion = np.std(c_curve_diff, axis=1, keepdims=True)
    else:
        censor_slope = np.zeros_like(q_pred)
        censor_dispersion = np.zeros_like(q_pred)

    proxy_signal = h_abs + q_curve_energy + s_curve_energy
    weak_proxy_raw = 1.0 / (0.05 + proxy_signal)
    survival_flat_raw = 1.0 / (0.05 + s_curve_slope + s_curve_dispersion)
    censor_flat_raw = 1.0 / (0.05 + censor_slope + censor_dispersion)
    near_null_raw = 1.0 / (0.05 + h_abs + surv_abs + 0.5 * (q_curve_energy + s_curve_energy))
    tail_raw = near_null_raw * weak_proxy_raw * survival_flat_raw * np.maximum(balance, 0.05)

    proxy_strength = np.log1p(proxy_signal)
    weak_proxy_risk = np.log1p(weak_proxy_raw)
    survival_flatness = np.log1p(survival_flat_raw)
    censoring_flatness = np.log1p(censor_flat_raw)
    near_null_risk = np.log1p(near_null_raw)
    tail_risk = np.log1p(tail_raw)
    nonlinear_risk = np.log1p(q_curve_dispersion + s_curve_dispersion + np.abs(h_abs - surv_abs))
    confounding_load = np.log1p(q_margin + q_curve_level + s_curve_level)

    return np.hstack(
        [
            q_margin,
            balance,
            h_abs,
            surv_abs,
            q_curve_energy,
            q_curve_dispersion,
            s_curve_energy,
            s_curve_dispersion,
            s_curve_slope,
            censor_level,
            censor_slope,
            proxy_strength,
            weak_proxy_risk,
            survival_flatness,
            censoring_flatness,
            near_null_risk,
            tail_risk,
            nonlinear_risk,
            confounding_load,
        ]
    )


def _regime_priority_score(regime_meta):
    regime_meta = np.asarray(regime_meta, dtype=float)
    if regime_meta.ndim != 2 or regime_meta.shape[1] < 19:
        return np.ones(regime_meta.shape[0], dtype=float)
    weak_proxy = regime_meta[:, 12]
    survival_flat = regime_meta[:, 13]
    near_null = regime_meta[:, 15]
    tail_risk = regime_meta[:, 16]
    nonlinear_risk = regime_meta[:, 17]
    score = tail_risk + 0.35 * near_null + 0.25 * weak_proxy + 0.15 * survival_flat + 0.10 * nonlinear_risk
    score = np.asarray(score, dtype=float).reshape(-1)
    q95 = float(np.percentile(score, 95)) if score.size else 1.0
    return np.clip(score / max(q95, 1e-6), 0.25, 4.0)


def _build_summary_features(X, bridge, mode="basic", *, n_curve_knots=5):
    x = _ensure_2d(X)
    q_pred = np.asarray(bridge["q_pred"], dtype=float).reshape(-1, 1)
    h1_pred = np.asarray(bridge["h1_pred"], dtype=float).reshape(-1, 1)
    h0_pred = np.asarray(bridge["h0_pred"], dtype=float).reshape(-1, 1)
    m_pred = np.asarray(bridge["m_pred"], dtype=float).reshape(-1, 1)
    h_diff = h1_pred - h0_pred
    parts = [x, q_pred, h1_pred, h0_pred, h_diff, m_pred]

    if mode in {"surv", "surv_x_interact"}:
        surv1_pred = np.asarray(bridge["surv1_pred"], dtype=float).reshape(-1, 1)
        surv0_pred = np.asarray(bridge["surv0_pred"], dtype=float).reshape(-1, 1)
        surv_diff = np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1)
        parts.extend([surv1_pred, surv0_pred, surv_diff])

    if mode == "surv_x_interact":
        interact_base = np.hstack(
            [
                q_pred,
                h_diff,
                m_pred,
                np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1),
            ]
        )
        parts.append(_pairwise_products(x, interact_base))
    elif mode == "curve":
        q_curve_1 = _select_curve_knots(bridge["qhat1_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_0 = _select_curve_knots(bridge["qhat0_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_diff = q_curve_1 - q_curve_0
        q_curve_mean = 0.5 * (q_curve_1 + q_curve_0)

        s_curve_1 = _select_curve_knots(bridge["s1_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_0 = _select_curve_knots(bridge["s0_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_diff = s_curve_1 - s_curve_0
        s_curve_mean = 0.5 * (s_curve_1 + s_curve_0)

        parts.extend([q_curve_mean, q_curve_diff, s_curve_mean, s_curve_diff])
    elif mode == "curve_x_interact":
        q_curve_1 = _select_curve_knots(bridge["qhat1_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_0 = _select_curve_knots(bridge["qhat0_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_diff = q_curve_1 - q_curve_0
        q_curve_mean = 0.5 * (q_curve_1 + q_curve_0)

        s_curve_1 = _select_curve_knots(bridge["s1_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_0 = _select_curve_knots(bridge["s0_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_diff = s_curve_1 - s_curve_0
        s_curve_mean = 0.5 * (s_curve_1 + s_curve_0)

        curve_block = np.hstack([q_curve_mean, q_curve_diff, s_curve_mean, s_curve_diff])
        parts.append(curve_block)
        parts.append(_pairwise_products(x, np.hstack([q_curve_diff, s_curve_diff])))
    elif mode == "curve_proxy_compact_x_interact":
        surv1_pred = np.asarray(bridge["surv1_pred"], dtype=float).reshape(-1, 1)
        surv0_pred = np.asarray(bridge["surv0_pred"], dtype=float).reshape(-1, 1)
        surv_diff = np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1)
        q_margin = np.abs(q_pred - 0.5)
        balance = 4.0 * q_pred * (1.0 - q_pred)

        q_curve_1 = _select_curve_knots(bridge["qhat1_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_0 = _select_curve_knots(bridge["qhat0_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_diff = q_curve_1 - q_curve_0
        q_curve_mean = 0.5 * (q_curve_1 + q_curve_0)

        s_curve_1 = _select_curve_knots(bridge["s1_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_0 = _select_curve_knots(bridge["s0_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_diff = s_curve_1 - s_curve_0
        s_curve_mean = 0.5 * (s_curve_1 + s_curve_0)

        w_proxy = np.asarray(bridge["w_raw"], dtype=float)
        z_proxy = np.asarray(bridge["z_raw"], dtype=float)
        proxy_block = np.hstack(
            [
                _proxy_summary_block(w_proxy),
                _proxy_summary_block(z_proxy),
                _joint_proxy_summary_block(w_proxy, z_proxy),
            ]
        )
        regime = _build_regime_descriptors(bridge, n_curve_knots=n_curve_knots)
        regime_core = regime[:, [11, 12, 13, 15, 16, 18]]
        compact_curve = np.hstack(
            [
                _block_summary_stats(q_curve_mean),
                _block_summary_stats(q_curve_diff),
                _block_summary_stats(s_curve_mean),
                _block_summary_stats(s_curve_diff),
            ]
        )
        interact_base = np.hstack(
            [
                q_pred,
                h_diff,
                m_pred,
                surv_diff,
                q_margin,
                balance,
                proxy_block[:, [0, 2, 6, 8, 12, 15]],
                regime_core[:, [1, 3, 4]],
            ]
        )
        parts.extend(
            [
                surv1_pred,
                surv0_pred,
                surv_diff,
                q_margin,
                balance,
                compact_curve,
                proxy_block,
                regime_core,
                _pairwise_products(x, interact_base),
            ]
        )
    elif mode == "curve_compact_x_interact":
        q_curve_1 = _select_curve_knots(bridge["qhat1_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_0 = _select_curve_knots(bridge["qhat0_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_diff = q_curve_1 - q_curve_0
        q_curve_mean = 0.5 * (q_curve_1 + q_curve_0)

        s_curve_1 = _select_curve_knots(bridge["s1_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_0 = _select_curve_knots(bridge["s0_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_diff = s_curve_1 - s_curve_0
        s_curve_mean = 0.5 * (s_curve_1 + s_curve_0)

        compact_block = np.hstack(
            [
                _block_summary_stats(q_curve_mean),
                _block_summary_stats(q_curve_diff),
                _block_summary_stats(s_curve_mean),
                _block_summary_stats(s_curve_diff),
            ]
        )
        interact_base = np.hstack(
            [
                _block_summary_stats(q_curve_diff)[:, :4],
                _block_summary_stats(s_curve_diff)[:, :4],
            ]
        )
        parts.append(compact_block)
        parts.append(_pairwise_products(x, interact_base))
    elif mode == "curve_only":
        q_curve_1 = _select_curve_knots(bridge["qhat1_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_0 = _select_curve_knots(bridge["qhat0_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_diff = q_curve_1 - q_curve_0
        s_curve_1 = _select_curve_knots(bridge["s1_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_0 = _select_curve_knots(bridge["s0_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_diff = s_curve_1 - s_curve_0
        parts = [x, q_pred, h_diff, m_pred, q_curve_diff, s_curve_diff]
    elif mode == "multi_x_interact":
        surv1_pred = np.asarray(bridge["surv1_pred"], dtype=float).reshape(-1, 1)
        surv0_pred = np.asarray(bridge["surv0_pred"], dtype=float).reshape(-1, 1)
        surv_diff = np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1)

        q_curve_1 = _select_curve_knots(bridge["qhat1_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_0 = _select_curve_knots(bridge["qhat0_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_diff = q_curve_1 - q_curve_0
        q_curve_mean = 0.5 * (q_curve_1 + q_curve_0)

        s_curve_1 = _select_curve_knots(bridge["s1_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_0 = _select_curve_knots(bridge["s0_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_diff = s_curve_1 - s_curve_0
        s_curve_mean = 0.5 * (s_curve_1 + s_curve_0)

        curve_block = np.hstack([q_curve_mean, q_curve_diff, s_curve_mean, s_curve_diff])
        interact_base = np.hstack([q_pred, h_diff, m_pred, surv_diff, q_curve_diff, s_curve_diff])
        parts.extend([surv1_pred, surv0_pred, surv_diff, curve_block, _pairwise_products(x, interact_base)])
    elif mode == "multi_censor_x_interact":
        surv1_pred = np.asarray(bridge["surv1_pred"], dtype=float).reshape(-1, 1)
        surv0_pred = np.asarray(bridge["surv0_pred"], dtype=float).reshape(-1, 1)
        surv_diff = np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1)

        q_curve_1 = _select_curve_knots(bridge["qhat1_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_0 = _select_curve_knots(bridge["qhat0_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_diff = q_curve_1 - q_curve_0
        q_curve_mean = 0.5 * (q_curve_1 + q_curve_0)

        s_curve_1 = _select_curve_knots(bridge["s1_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_0 = _select_curve_knots(bridge["s0_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_diff = s_curve_1 - s_curve_0
        s_curve_mean = 0.5 * (s_curve_1 + s_curve_0)

        _, c_curve_mean, c_curve_diff = _curve_mean_and_diff(
            bridge["c_curve"], drop_last=True, n_knots=n_curve_knots
        )
        censor_level = c_curve_mean.mean(axis=1, keepdims=True)
        censor_slope = c_curve_diff.mean(axis=1, keepdims=True) if c_curve_diff.size else np.zeros_like(censor_level)

        curve_block = np.hstack([q_curve_mean, q_curve_diff, s_curve_mean, s_curve_diff, c_curve_mean, c_curve_diff])
        interact_base = np.hstack(
            [q_pred, h_diff, m_pred, surv_diff, q_curve_diff, s_curve_diff, censor_level, censor_slope]
        )
        parts.extend(
            [
                surv1_pred,
                surv0_pred,
                surv_diff,
                censor_level,
                censor_slope,
                curve_block,
                _pairwise_products(x, interact_base),
            ]
        )
    elif mode == "multi_compact":
        surv1_pred = np.asarray(bridge["surv1_pred"], dtype=float).reshape(-1, 1)
        surv0_pred = np.asarray(bridge["surv0_pred"], dtype=float).reshape(-1, 1)
        surv_diff = np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1)

        q_curve_1 = _select_curve_knots(bridge["qhat1_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_0 = _select_curve_knots(bridge["qhat0_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_diff = q_curve_1 - q_curve_0
        q_curve_mean = 0.5 * (q_curve_1 + q_curve_0)

        s_curve_1 = _select_curve_knots(bridge["s1_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_0 = _select_curve_knots(bridge["s0_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_diff = s_curve_1 - s_curve_0
        s_curve_mean = 0.5 * (s_curve_1 + s_curve_0)

        _, c_curve_mean, c_curve_diff = _curve_mean_and_diff(
            bridge["c_curve"], drop_last=True, n_knots=n_curve_knots
        )
        censor_level = c_curve_mean.mean(axis=1, keepdims=True) if c_curve_mean.size else np.zeros_like(q_pred)
        censor_slope = c_curve_diff.mean(axis=1, keepdims=True) if c_curve_diff.size else np.zeros_like(q_pred)

        regime = _build_regime_descriptors(bridge, n_curve_knots=n_curve_knots)
        regime_core = regime[:, [11, 12, 13, 14, 15, 16, 18]]
        compact_curve = np.hstack(
            [
                _block_summary_stats(q_curve_mean),
                _block_summary_stats(q_curve_diff),
                _block_summary_stats(s_curve_mean),
                _block_summary_stats(s_curve_diff),
                _block_summary_stats(c_curve_mean),
                _block_summary_stats(c_curve_diff),
            ]
        )
        parts.extend(
            [
                surv1_pred,
                surv0_pred,
                surv_diff,
                censor_level,
                censor_slope,
                compact_curve,
                regime_core,
            ]
        )
    elif mode == "multi_compact_x_interact":
        surv1_pred = np.asarray(bridge["surv1_pred"], dtype=float).reshape(-1, 1)
        surv0_pred = np.asarray(bridge["surv0_pred"], dtype=float).reshape(-1, 1)
        surv_diff = np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1)

        q_curve_1 = _select_curve_knots(bridge["qhat1_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_0 = _select_curve_knots(bridge["qhat0_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_diff = q_curve_1 - q_curve_0
        q_curve_mean = 0.5 * (q_curve_1 + q_curve_0)

        s_curve_1 = _select_curve_knots(bridge["s1_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_0 = _select_curve_knots(bridge["s0_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_diff = s_curve_1 - s_curve_0
        s_curve_mean = 0.5 * (s_curve_1 + s_curve_0)

        _, c_curve_mean, c_curve_diff = _curve_mean_and_diff(
            bridge["c_curve"], drop_last=True, n_knots=n_curve_knots
        )
        censor_level = c_curve_mean.mean(axis=1, keepdims=True) if c_curve_mean.size else np.zeros_like(q_pred)
        censor_slope = c_curve_diff.mean(axis=1, keepdims=True) if c_curve_diff.size else np.zeros_like(q_pred)

        regime = _build_regime_descriptors(bridge, n_curve_knots=n_curve_knots)
        regime_core = regime[:, [11, 12, 13, 14, 15, 16, 18]]
        compact_curve = np.hstack(
            [
                _block_summary_stats(q_curve_mean),
                _block_summary_stats(q_curve_diff),
                _block_summary_stats(s_curve_mean),
                _block_summary_stats(s_curve_diff),
                _block_summary_stats(c_curve_mean),
                _block_summary_stats(c_curve_diff),
            ]
        )
        interact_base = np.hstack(
            [
                q_pred,
                h_diff,
                m_pred,
                surv_diff,
                censor_level,
                censor_slope,
                regime_core[:, [1, 2, 4, 5]],
            ]
        )
        parts.extend(
            [
                surv1_pred,
                surv0_pred,
                surv_diff,
                censor_level,
                censor_slope,
                compact_curve,
                regime_core,
                _pairwise_products(x, interact_base),
            ]
        )
    elif mode == "multi_regime_compact":
        surv1_pred = np.asarray(bridge["surv1_pred"], dtype=float).reshape(-1, 1)
        surv0_pred = np.asarray(bridge["surv0_pred"], dtype=float).reshape(-1, 1)
        surv_diff = np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1)

        q_curve_1 = _select_curve_knots(bridge["qhat1_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_0 = _select_curve_knots(bridge["qhat0_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_diff = q_curve_1 - q_curve_0
        q_curve_mean = 0.5 * (q_curve_1 + q_curve_0)

        s_curve_1 = _select_curve_knots(bridge["s1_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_0 = _select_curve_knots(bridge["s0_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_diff = s_curve_1 - s_curve_0
        s_curve_mean = 0.5 * (s_curve_1 + s_curve_0)

        regime = _build_regime_descriptors(bridge, n_curve_knots=n_curve_knots)
        regime_core = regime[:, [12, 13, 15, 16, 18]]
        curve_block = np.hstack([q_curve_mean, q_curve_diff, s_curve_mean, s_curve_diff])
        parts.extend(
            [
                surv1_pred,
                surv0_pred,
                surv_diff,
                curve_block,
                regime,
                _pairwise_products(x, regime_core),
            ]
        )
    elif mode == "multi_regime_x_interact":
        surv1_pred = np.asarray(bridge["surv1_pred"], dtype=float).reshape(-1, 1)
        surv0_pred = np.asarray(bridge["surv0_pred"], dtype=float).reshape(-1, 1)
        surv_diff = np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1)

        q_curve_1 = _select_curve_knots(bridge["qhat1_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_0 = _select_curve_knots(bridge["qhat0_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_diff = q_curve_1 - q_curve_0
        q_curve_mean = 0.5 * (q_curve_1 + q_curve_0)

        s_curve_1 = _select_curve_knots(bridge["s1_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_0 = _select_curve_knots(bridge["s0_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_diff = s_curve_1 - s_curve_0
        s_curve_mean = 0.5 * (s_curve_1 + s_curve_0)

        _, c_curve_mean, c_curve_diff = _curve_mean_and_diff(
            bridge["c_curve"], drop_last=True, n_knots=n_curve_knots
        )
        censor_level = c_curve_mean.mean(axis=1, keepdims=True)
        censor_slope = c_curve_diff.mean(axis=1, keepdims=True) if c_curve_diff.size else np.zeros_like(censor_level)
        regime = _build_regime_descriptors(bridge, n_curve_knots=n_curve_knots)
        regime_core = regime[:, [12, 13, 15, 16, 18]]

        curve_block = np.hstack([q_curve_mean, q_curve_diff, s_curve_mean, s_curve_diff, c_curve_mean, c_curve_diff])
        interact_base = np.hstack(
            [q_pred, h_diff, m_pred, surv_diff, q_curve_diff, s_curve_diff, censor_level, censor_slope, regime_core]
        )
        parts.extend(
            [
                surv1_pred,
                surv0_pred,
                surv_diff,
                censor_level,
                censor_slope,
                curve_block,
                regime,
                _pairwise_products(x, interact_base),
                _pairwise_products(x, regime_core),
            ]
        )
    elif mode in {"rmst_basis", "rmst_basis_x_interact", "rmst_basis_regime"}:
        surv1_pred = np.asarray(bridge["surv1_pred"], dtype=float).reshape(-1, 1)
        surv0_pred = np.asarray(bridge["surv0_pred"], dtype=float).reshape(-1, 1)
        surv_diff = np.asarray(bridge["surv_diff_pred"], dtype=float).reshape(-1, 1)

        q_curve_1 = _select_curve_knots(bridge["qhat1_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_0 = _select_curve_knots(bridge["qhat0_curve"], drop_last=True, n_knots=n_curve_knots)
        q_curve_diff = q_curve_1 - q_curve_0
        q_curve_mean = 0.5 * (q_curve_1 + q_curve_0)

        s_curve_1 = _select_curve_knots(bridge["s1_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_0 = _select_curve_knots(bridge["s0_curve"], drop_last=True, n_knots=n_curve_knots)
        s_curve_diff = s_curve_1 - s_curve_0
        s_curve_mean = 0.5 * (s_curve_1 + s_curve_0)

        _, c_curve_mean, c_curve_diff = _curve_mean_and_diff(
            bridge["c_curve"], drop_last=True, n_knots=n_curve_knots
        )
        censor_level = c_curve_mean.mean(axis=1, keepdims=True) if c_curve_mean.size else np.zeros_like(q_pred)
        censor_slope = c_curve_diff.mean(axis=1, keepdims=True) if c_curve_diff.size else np.zeros_like(q_pred)

        q_basis = np.hstack([_curve_rmst_basis_stats(q_curve_mean), _curve_rmst_basis_stats(q_curve_diff)])
        s_basis = np.hstack([_curve_rmst_basis_stats(s_curve_mean), _curve_rmst_basis_stats(s_curve_diff)])
        c_basis = np.hstack([_curve_rmst_basis_stats(c_curve_mean), _curve_rmst_basis_stats(c_curve_diff)])
        basis = np.hstack([q_basis, s_basis, c_basis])
        parts.extend([surv1_pred, surv0_pred, surv_diff, censor_level, censor_slope, basis])

        if mode == "rmst_basis_x_interact":
            interact_base = np.hstack(
                [
                    q_pred,
                    h_diff,
                    m_pred,
                    surv_diff,
                    censor_level,
                    censor_slope,
                    q_basis[:, :4],
                    s_basis[:, :4],
                ]
            )
            parts.append(_pairwise_products(x, interact_base))
        elif mode == "rmst_basis_regime":
            regime = _build_regime_descriptors(bridge, n_curve_knots=n_curve_knots)
            regime_core = regime[:, [11, 12, 13, 14, 15, 16, 18]]
            parts.extend([regime_core, _pairwise_products(x, regime_core[:, [1, 2, 4, 5]])])
    elif mode not in {"basic", "surv"}:
        raise ValueError(f"Unsupported summary_feature_mode: {mode}")

    return np.hstack(parts)


def _crossfit_summary_arrays(owner, X, A, y_packed, W, Z, *, return_regime_meta=False):
    x = _ensure_2d(X).astype(float)
    a = np.asarray(A, dtype=float).ravel()
    w_nuis, z_nuis = owner._prepare_nuisance_inputs(W, Z)

    n = len(a)
    y_res = np.zeros(n, dtype=float)
    a_res = np.zeros(n, dtype=float)
    x_final = None
    regime_meta = None

    splitter = KFold(n_splits=owner._cv, shuffle=True, random_state=owner._random_state)
    for train_idx, test_idx in splitter.split(x):
        nuisance = owner._make_nuisance()
        nuisance.train(
            False,
            None,
            y_packed[train_idx],
            a[train_idx],
            X=x[train_idx],
            W=w_nuis[train_idx],
            Z=z_nuis[train_idx],
        )
        y_fold, a_fold = nuisance.predict(
            y_packed[test_idx],
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
        y_res[test_idx] = y_fold
        a_res[test_idx] = a_fold.ravel()
        x_final_fold = _build_summary_features(
            x[test_idx],
            bridge,
            mode=owner._summary_feature_mode,
            n_curve_knots=owner._summary_curve_knots,
        )
        if x_final is None:
            x_final = np.zeros((n, x_final_fold.shape[1]), dtype=float)
        x_final[test_idx] = x_final_fold
        if return_regime_meta:
            regime_fold = _build_regime_descriptors(bridge, n_curve_knots=owner._summary_curve_knots)
            if regime_meta is None:
                regime_meta = np.zeros((n, regime_fold.shape[1]), dtype=float)
            regime_meta[test_idx] = regime_fold

    if return_regime_meta:
        return x, a, w_nuis, z_nuis, x_final, y_res, a_res, regime_meta
    return x, a, w_nuis, z_nuis, x_final, y_res, a_res


class _BaseTwoStageSummarySurvivalForest:
    def __init__(
        self,
        *,
        target="RMST",
        horizon=None,
        n_estimators=200,
        min_samples_leaf=20,
        cv=3,
        random_state=42,
        observed_only=False,
        nuisance_feature_mode="dup",
        q_kind="logit",
        q_trees=300,
        q_min_samples_leaf=20,
        q_poly_degree=2,
        q_clip=0.02,
        y_tilde_clip_quantile=0.99,
        y_res_clip_percentiles=(1.0, 99.0),
        h_kind="extra",
        h_n_estimators=800,
        h_min_samples_leaf=5,
        censoring_estimator="kaplan-meier",
        n_jobs=1,
        summary_feature_mode="basic",
        forest_max_depth=None,
        summary_curve_knots=5,
        forest_honest=True,
        forest_inference=True,
        forest_fit_intercept=True,
    ):
        self._target = target
        self._horizon = horizon
        self._n_estimators = n_estimators
        self._min_samples_leaf = min_samples_leaf
        self._cv = cv
        self._random_state = random_state
        self._observed_only = observed_only
        self._nuisance_feature_mode = nuisance_feature_mode
        self._summary_feature_mode = summary_feature_mode
        self._q_model_template = make_q_model(
            q_kind,
            random_state=random_state,
            n_estimators=q_trees,
            min_samples_leaf=q_min_samples_leaf,
            poly_degree=q_poly_degree,
        )
        self._h_model_template = make_h_model(
            h_kind,
            random_state=random_state,
            n_estimators=h_n_estimators,
            min_samples_leaf=h_min_samples_leaf,
            n_jobs=n_jobs,
        )
        self._q_clip = q_clip
        self._y_tilde_clip_quantile = y_tilde_clip_quantile
        self._y_res_clip_percentiles = y_res_clip_percentiles
        self._censoring_estimator = censoring_estimator
        self._forest_max_depth = forest_max_depth
        self._summary_curve_knots = int(summary_curve_knots)
        self._forest_honest = bool(forest_honest)
        self._forest_inference = bool(forest_inference)
        self._forest_fit_intercept = bool(forest_fit_intercept)

        self._forest = None
        self._full_nuisance = None

    def _make_nuisance(self):
        return _MildShrinkNCSurvivalNuisance(
            q_model=self._q_model_template,
            h_model=self._h_model_template,
            target=self._target,
            horizon=self._horizon,
            final_feature_mode="x_only",
            nuisance_feature_mode=self._nuisance_feature_mode,
            censoring_estimator=self._censoring_estimator,
            q_clip=self._q_clip,
            y_tilde_clip_quantile=self._y_tilde_clip_quantile,
            y_res_clip_percentiles=self._y_res_clip_percentiles,
        )

    def _prepare_nuisance_inputs(self, W, Z):
        w = _ensure_2d(W)
        z = _ensure_2d(Z)
        if self._observed_only:
            w = np.zeros_like(w, dtype=float)
            z = np.zeros_like(z, dtype=float)
        return w, z

    def fit_components(self, X, A, time, event, Z, W):
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        x, a, w_nuis, z_nuis, x_final, y_res, a_res = _crossfit_summary_arrays(self, X, A, y_packed, W, Z)
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
            y_packed,
            a,
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._forest is None or self._full_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._full_nuisance.predict_bridge_outputs(
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        bridge["w_raw"] = w_nuis
        bridge["z_raw"] = z_nuis
        x_final = _build_summary_features(
            x,
            bridge,
            mode=self._summary_feature_mode,
            n_curve_knots=self._summary_curve_knots,
        )
        return self._forest.predict(x_final)


class TwoStageBridgeSummarySurvivalForest(_BaseTwoStageSummarySurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("observed_only", False)
        super().__init__(*args, **kwargs)


class TwoStageObservedSummarySurvivalForest(_BaseTwoStageSummarySurvivalForest):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("observed_only", True)
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("q_trees", 300)
        kwargs.setdefault("q_min_samples_leaf", 20)
        kwargs.setdefault("q_poly_degree", 2)
        kwargs.setdefault("q_clip", 0.02)
        kwargs.setdefault("h_kind", "rf")
        kwargs.setdefault("h_n_estimators", 300)
        kwargs.setdefault("h_min_samples_leaf", 20)
        super().__init__(*args, **kwargs)


class BlendedTwoStageBridgeSummarySurvivalForest:
    def __init__(
        self,
        *,
        alpha=0.55,
        model_a_params=None,
        model_b_params=None,
    ):
        self._alpha = float(alpha)
        self._model_a = TwoStageBridgeSummarySurvivalForest(**(model_a_params or {}))
        self._model_b = TwoStageBridgeSummarySurvivalForest(**(model_b_params or {}))

    def fit_components(self, X, A, time, event, Z, W):
        self._model_a.fit_components(X, A, time, event, Z, W)
        self._model_b.fit_components(X, A, time, event, Z, W)
        return self

    def effect_from_components(self, X, W, Z):
        pred_a = self._model_a.effect_from_components(X, W, Z)
        pred_b = self._model_b.effect_from_components(X, W, Z)
        return self._alpha * pred_a + (1.0 - self._alpha) * pred_b


class TwoStageBridgeSummaryPseudoOutcomeRegressor(_BaseTwoStageSummarySurvivalForest):
    def __init__(
        self,
        *args,
        final_model_kind="hgb",
        final_model_trees=600,
        final_model_min_samples_leaf=10,
        pseudo_clip=0.05,
        tau_clip_percentiles=(1.0, 99.0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._final_model_template = make_h_model(
            final_model_kind,
            random_state=self._random_state,
            n_estimators=final_model_trees,
            min_samples_leaf=final_model_min_samples_leaf,
            n_jobs=1,
        )
        self._pseudo_clip = float(pseudo_clip)
        self._tau_clip_percentiles = tau_clip_percentiles
        self._regressor = None

    def fit_components(self, X, A, time, event, Z, W):
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        x, _, w_nuis, z_nuis, x_final, y_res, a_res = _crossfit_summary_arrays(self, X, A, y_packed, W, Z)

        denom = np.where(np.abs(a_res) >= self._pseudo_clip, a_res, np.sign(a_res) * self._pseudo_clip)
        denom[np.abs(denom) < self._pseudo_clip] = self._pseudo_clip
        tau_tilde = y_res / denom
        lo, hi = np.percentile(tau_tilde, self._tau_clip_percentiles)
        tau_tilde = np.clip(tau_tilde, lo, hi)
        weights = np.square(a_res)

        self._regressor = clone(self._final_model_template)
        self._regressor.fit(x_final, tau_tilde, sample_weight=weights)

        self._full_nuisance = self._make_nuisance()
        self._full_nuisance.train(
            False,
            None,
            y_packed,
            np.asarray(A, dtype=float).ravel(),
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._regressor is None or self._full_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._full_nuisance.predict_bridge_outputs(
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        x_final = _build_summary_features(
            x,
            bridge,
            mode=self._summary_feature_mode,
            n_curve_knots=self._summary_curve_knots,
        )
        return np.asarray(self._regressor.predict(x_final), dtype=float)


class CalibratedTwoStageBridgeSummarySurvivalForest(_BaseTwoStageSummarySurvivalForest):
    def __init__(
        self,
        *args,
        calibration_degree=1,
        calibration_clip=0.05,
        calibration_target_clip_percentiles=(5.0, 95.0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._calibration_degree = int(calibration_degree)
        self._calibration_clip = float(calibration_clip)
        self._calibration_target_clip_percentiles = calibration_target_clip_percentiles
        self._calibrator = None

    def fit_components(self, X, A, time, event, Z, W):
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        x, a, w_nuis, z_nuis, x_final, y_res, a_res = _crossfit_summary_arrays(self, X, A, y_packed, W, Z)
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

        base_pred = np.asarray(self._forest.predict(x_final), dtype=float).reshape(-1, 1)
        denom = np.where(np.abs(a_res) >= self._calibration_clip, a_res, np.sign(a_res) * self._calibration_clip)
        denom[np.abs(denom) < self._calibration_clip] = self._calibration_clip
        tau_tilde = y_res / denom
        lo, hi = np.percentile(tau_tilde, self._calibration_target_clip_percentiles)
        tau_tilde = np.clip(tau_tilde, lo, hi)
        weights = np.square(a_res)

        if self._calibration_degree <= 1:
            self._calibrator = LinearRegression()
            self._calibrator.fit(base_pred, tau_tilde, sample_weight=weights)
        else:
            self._calibrator = Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=self._calibration_degree, include_bias=False)),
                    ("linear", LinearRegression()),
                ]
            )
            self._calibrator.fit(base_pred, tau_tilde, linear__sample_weight=weights)

        self._full_nuisance = self._make_nuisance()
        self._full_nuisance.train(
            False,
            None,
            y_packed,
            a,
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._forest is None or self._full_nuisance is None or self._calibrator is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._full_nuisance.predict_bridge_outputs(
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        x_final = _build_summary_features(
            x,
            bridge,
            mode=self._summary_feature_mode,
            n_curve_knots=self._summary_curve_knots,
        )
        base_pred = np.asarray(self._forest.predict(x_final), dtype=float).reshape(-1, 1)
        return np.asarray(self._calibrator.predict(base_pred), dtype=float)


class RegimeShrunkTwoStageBridgeSummarySurvivalForest(_BaseTwoStageSummarySurvivalForest):
    def __init__(
        self,
        *args,
        shrink_anchor="mean",
        shrink_alpha_grid=(0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
        shrink_score_power=1.0,
        shrink_lambda_floor=0.15,
        shrink_lambda_ceiling=1.0,
        shrink_pseudo_clip=0.05,
        shrink_target_clip_percentiles=(2.0, 98.0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._shrink_anchor = str(shrink_anchor)
        self._shrink_alpha_grid = tuple(float(v) for v in shrink_alpha_grid)
        self._shrink_score_power = float(shrink_score_power)
        self._shrink_lambda_floor = float(shrink_lambda_floor)
        self._shrink_lambda_ceiling = float(shrink_lambda_ceiling)
        self._shrink_pseudo_clip = float(shrink_pseudo_clip)
        self._shrink_target_clip_percentiles = shrink_target_clip_percentiles
        self._best_shrink_alpha = 0.0
        self._shrink_anchor_value = 0.0

    def _scaled_regime_score(self, regime_meta):
        score = _regime_priority_score(regime_meta)
        score = np.power(np.maximum(score, 1e-6), self._shrink_score_power)
        median = float(np.median(score)) if score.size else 1.0
        return score / max(median, 1e-6)

    def _lambda_from_score(self, score, alpha):
        lam = 1.0 / (1.0 + alpha * score)
        return np.clip(lam, self._shrink_lambda_floor, self._shrink_lambda_ceiling)

    def _anchor_value(self, pseudo_target, weights):
        if self._shrink_anchor == "zero":
            return 0.0
        weights = np.asarray(weights, dtype=float)
        denom = float(np.sum(weights))
        if denom <= 1e-8:
            return float(np.mean(pseudo_target))
        return float(np.sum(weights * pseudo_target) / denom)

    def fit_components(self, X, A, time, event, Z, W):
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        x, a, w_nuis, z_nuis, x_final, y_res, a_res, regime_meta = _crossfit_summary_arrays(
            self,
            X,
            A,
            y_packed,
            W,
            Z,
            return_regime_meta=True,
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

        base_pred = np.asarray(self._forest.predict(x_final), dtype=float).reshape(-1)
        denom = np.where(np.abs(a_res) >= self._shrink_pseudo_clip, a_res, np.sign(a_res) * self._shrink_pseudo_clip)
        denom[np.abs(denom) < self._shrink_pseudo_clip] = self._shrink_pseudo_clip
        pseudo_target = y_res / denom
        lo, hi = np.percentile(pseudo_target, self._shrink_target_clip_percentiles)
        pseudo_target = np.clip(pseudo_target, lo, hi)
        weights = np.square(a_res)

        regime_score = self._scaled_regime_score(regime_meta)
        self._shrink_anchor_value = self._anchor_value(pseudo_target, weights)
        best_loss = np.inf
        best_alpha = 0.0
        for alpha in self._shrink_alpha_grid:
            lam = self._lambda_from_score(regime_score, alpha)
            pred = self._shrink_anchor_value + lam * (base_pred - self._shrink_anchor_value)
            loss = float(np.average(np.square(pred - pseudo_target), weights=np.clip(weights, 1e-6, None)))
            if loss < best_loss:
                best_loss = loss
                best_alpha = alpha
        self._best_shrink_alpha = float(best_alpha)

        self._full_nuisance = self._make_nuisance()
        self._full_nuisance.train(
            False,
            None,
            y_packed,
            a,
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._forest is None or self._full_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._full_nuisance.predict_bridge_outputs(
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        x_final = _build_summary_features(
            x,
            bridge,
            mode=self._summary_feature_mode,
            n_curve_knots=self._summary_curve_knots,
        )
        regime_meta = _build_regime_descriptors(bridge, n_curve_knots=self._summary_curve_knots)
        base_pred = np.asarray(self._forest.predict(x_final), dtype=float).reshape(-1)
        regime_score = self._scaled_regime_score(regime_meta)
        lam = self._lambda_from_score(regime_score, self._best_shrink_alpha)
        return self._shrink_anchor_value + lam * (base_pred - self._shrink_anchor_value)


class DistilledTwoStageBridgeSummaryRegressor(_BaseTwoStageSummarySurvivalForest):
    def __init__(
        self,
        *args,
        student_model_kind="extra",
        student_model_trees=1200,
        student_model_min_samples_leaf=5,
        teacher_mode="blend",
        teacher_alpha=0.55,
        teacher_weight=1.0,
        pseudo_clip=0.05,
        pseudo_target_clip_percentiles=(1.0, 99.0),
        teacher_curve_params=None,
        teacher_surv_params=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._teacher_mode = str(teacher_mode)
        self._teacher_alpha = float(teacher_alpha)
        self._teacher_weight = float(teacher_weight)
        self._pseudo_clip = float(pseudo_clip)
        self._pseudo_target_clip_percentiles = pseudo_target_clip_percentiles
        self._teacher_curve_params = teacher_curve_params or {}
        self._teacher_surv_params = teacher_surv_params or {}
        self._student_model_template = make_h_model(
            student_model_kind,
            random_state=self._random_state,
            n_estimators=student_model_trees,
            min_samples_leaf=student_model_min_samples_leaf,
            n_jobs=1,
        )
        self._student = None

    def _make_teacher_models(self):
        shared = dict(
            target=self._target,
            horizon=self._horizon,
            cv=self._cv,
            random_state=self._random_state,
            censoring_estimator=self._censoring_estimator,
            observed_only=False,
        )
        curve_params = dict(
            shared,
            n_estimators=200,
            min_samples_leaf=20,
            q_kind="logit",
            h_kind="extra",
            h_n_estimators=800,
            h_min_samples_leaf=5,
            summary_feature_mode="curve_x_interact",
            nuisance_feature_mode="interact",
        )
        curve_params.update(self._teacher_curve_params)
        surv_params = dict(
            shared,
            n_estimators=200,
            min_samples_leaf=20,
            q_kind="logit",
            h_kind="extra",
            h_n_estimators=1200,
            h_min_samples_leaf=5,
            summary_feature_mode="surv_x_interact",
            nuisance_feature_mode="dup",
        )
        surv_params.update(self._teacher_surv_params)
        return TwoStageBridgeSummarySurvivalForest(**curve_params), TwoStageBridgeSummarySurvivalForest(**surv_params)

    def fit_components(self, X, A, time, event, Z, W):
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        x, _, w_nuis, z_nuis, x_final, y_res, a_res = _crossfit_summary_arrays(self, X, A, y_packed, W, Z)

        splitter = KFold(n_splits=self._cv, shuffle=True, random_state=self._random_state)
        teacher_target = np.zeros(x.shape[0], dtype=float)
        for train_idx, test_idx in splitter.split(x):
            curve_teacher, surv_teacher = self._make_teacher_models()
            mode = self._teacher_mode.lower()
            if mode == "curve":
                curve_teacher.fit_components(
                    x[train_idx],
                    np.asarray(A)[train_idx],
                    np.asarray(time)[train_idx],
                    np.asarray(event)[train_idx],
                    z_nuis[train_idx],
                    w_nuis[train_idx],
                )
                teacher_target[test_idx] = curve_teacher.effect_from_components(
                    x[test_idx], w_nuis[test_idx], z_nuis[test_idx]
                ).ravel()
            elif mode == "surv":
                surv_teacher.fit_components(
                    x[train_idx],
                    np.asarray(A)[train_idx],
                    np.asarray(time)[train_idx],
                    np.asarray(event)[train_idx],
                    z_nuis[train_idx],
                    w_nuis[train_idx],
                )
                teacher_target[test_idx] = surv_teacher.effect_from_components(
                    x[test_idx], w_nuis[test_idx], z_nuis[test_idx]
                ).ravel()
            elif mode == "blend":
                curve_teacher.fit_components(
                    x[train_idx],
                    np.asarray(A)[train_idx],
                    np.asarray(time)[train_idx],
                    np.asarray(event)[train_idx],
                    z_nuis[train_idx],
                    w_nuis[train_idx],
                )
                surv_teacher.fit_components(
                    x[train_idx],
                    np.asarray(A)[train_idx],
                    np.asarray(time)[train_idx],
                    np.asarray(event)[train_idx],
                    z_nuis[train_idx],
                    w_nuis[train_idx],
                )
                pred_curve = curve_teacher.effect_from_components(x[test_idx], w_nuis[test_idx], z_nuis[test_idx]).ravel()
                pred_surv = surv_teacher.effect_from_components(x[test_idx], w_nuis[test_idx], z_nuis[test_idx]).ravel()
                teacher_target[test_idx] = self._teacher_alpha * pred_curve + (1.0 - self._teacher_alpha) * pred_surv
            else:
                raise ValueError(f"Unsupported teacher_mode: {self._teacher_mode}")

        denom = np.where(np.abs(a_res) >= self._pseudo_clip, a_res, np.sign(a_res) * self._pseudo_clip)
        denom[np.abs(denom) < self._pseudo_clip] = self._pseudo_clip
        pseudo_target = y_res / denom
        lo, hi = np.percentile(pseudo_target, self._pseudo_target_clip_percentiles)
        pseudo_target = np.clip(pseudo_target, lo, hi)
        student_target = self._teacher_weight * teacher_target + (1.0 - self._teacher_weight) * pseudo_target
        weights = np.square(a_res)

        self._student = clone(self._student_model_template)
        self._student.fit(x_final, student_target, sample_weight=weights)

        self._full_nuisance = self._make_nuisance()
        self._full_nuisance.train(
            False,
            None,
            y_packed,
            np.asarray(A, dtype=float).ravel(),
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._student is None or self._full_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._full_nuisance.predict_bridge_outputs(
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        x_final = _build_summary_features(
            x,
            bridge,
            mode=self._summary_feature_mode,
            n_curve_knots=self._summary_curve_knots,
        )
        return np.asarray(self._student.predict(x_final), dtype=float)


def _crossfit_single_teacher_target(owner, teacher_factory, X, A, time, event, W, Z):
    x = _ensure_2d(X).astype(float)
    a = np.asarray(A, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=float).ravel()
    w = _ensure_2d(W).astype(float)
    z = _ensure_2d(Z).astype(float)
    teacher_target = np.zeros(x.shape[0], dtype=float)
    splitter = KFold(n_splits=owner._cv, shuffle=True, random_state=owner._random_state)
    for train_idx, test_idx in splitter.split(x):
        teacher = teacher_factory()
        teacher.fit_components(
            x[train_idx],
            a[train_idx],
            time[train_idx],
            event[train_idx],
            z[train_idx],
            w[train_idx],
        )
        teacher_target[test_idx] = teacher.effect_from_components(x[test_idx], w[test_idx], z[test_idx]).ravel()
    return teacher_target


class SingleTeacherDistilledTwoStageBridgeSummaryRegressor(_BaseTwoStageSummarySurvivalForest):
    def __init__(
        self,
        *args,
        student_model_kind="extra",
        student_model_trees=1200,
        student_model_min_samples_leaf=5,
        teacher_weight=1.0,
        adaptive_teacher_mix=False,
        teacher_weight_min=0.0,
        teacher_regime_strength=1.0,
        pseudo_clip=0.05,
        pseudo_target_clip_percentiles=(1.0, 99.0),
        teacher_params=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._teacher_weight = float(teacher_weight)
        self._adaptive_teacher_mix = bool(adaptive_teacher_mix)
        self._teacher_weight_min = float(teacher_weight_min)
        self._teacher_regime_strength = float(teacher_regime_strength)
        self._pseudo_clip = float(pseudo_clip)
        self._pseudo_target_clip_percentiles = pseudo_target_clip_percentiles
        self._teacher_params = teacher_params or {}
        self._student_model_template = make_h_model(
            student_model_kind,
            random_state=self._random_state,
            n_estimators=student_model_trees,
            min_samples_leaf=student_model_min_samples_leaf,
            n_jobs=1,
        )
        self._student = None

    def _make_teacher(self):
        teacher_params = dict(
            target=self._target,
            horizon=self._horizon,
            cv=self._cv,
            random_state=self._random_state,
            censoring_estimator=self._censoring_estimator,
            observed_only=False,
            n_estimators=self._n_estimators,
            min_samples_leaf=self._min_samples_leaf,
            q_kind="logit",
            h_kind="extra",
            h_n_estimators=800,
            h_min_samples_leaf=5,
            summary_feature_mode=self._summary_feature_mode,
            nuisance_feature_mode=self._nuisance_feature_mode,
            summary_curve_knots=self._summary_curve_knots,
            forest_honest=self._forest_honest,
            forest_inference=self._forest_inference,
            forest_fit_intercept=self._forest_fit_intercept,
            forest_max_depth=self._forest_max_depth,
        )
        teacher_params.update(self._teacher_params)
        return TwoStageBridgeSummarySurvivalForest(**teacher_params)

    def fit_components(self, X, A, time, event, Z, W):
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        x, _, w_nuis, z_nuis, x_final, y_res, a_res, regime_meta = _crossfit_summary_arrays(
            self,
            X,
            A,
            y_packed,
            W,
            Z,
            return_regime_meta=True,
        )
        teacher_target = _crossfit_single_teacher_target(self, self._make_teacher, x, A, time, event, w_nuis, z_nuis)

        denom = np.where(np.abs(a_res) >= self._pseudo_clip, a_res, np.sign(a_res) * self._pseudo_clip)
        denom[np.abs(denom) < self._pseudo_clip] = self._pseudo_clip
        pseudo_target = y_res / denom
        lo, hi = np.percentile(pseudo_target, self._pseudo_target_clip_percentiles)
        pseudo_target = np.clip(pseudo_target, lo, hi)
        if self._adaptive_teacher_mix:
            regime_score = _regime_priority_score(regime_meta)
            teacher_weight = self._teacher_weight * np.exp(-self._teacher_regime_strength * regime_score)
            teacher_weight = np.clip(teacher_weight, self._teacher_weight_min, self._teacher_weight)
            student_target = teacher_weight * teacher_target + (1.0 - teacher_weight) * pseudo_target
        else:
            student_target = self._teacher_weight * teacher_target + (1.0 - self._teacher_weight) * pseudo_target
        weights = np.square(a_res)

        self._student = clone(self._student_model_template)
        self._student.fit(x_final, student_target, sample_weight=weights)

        self._full_nuisance = self._make_nuisance()
        self._full_nuisance.train(
            False,
            None,
            y_packed,
            np.asarray(A, dtype=float).ravel(),
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._student is None or self._full_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._full_nuisance.predict_bridge_outputs(
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        x_final = _build_summary_features(
            x,
            bridge,
            mode=self._summary_feature_mode,
            n_curve_knots=self._summary_curve_knots,
        )
        return np.asarray(self._student.predict(x_final), dtype=float)


class ErrorBiasedSingleTeacherDistilledRegressor(_BaseTwoStageSummarySurvivalForest):
    def __init__(
        self,
        *args,
        student_model_kind="extra",
        student_model_trees=1600,
        student_model_min_samples_leaf=5,
        student_model_max_depth=None,
        student_model_max_features=0.7,
        student_learning_rate=0.05,
        teacher_weight=0.9,
        adaptive_teacher_mix=False,
        teacher_weight_min=0.0,
        teacher_regime_strength=1.0,
        pseudo_clip=0.08,
        pseudo_target_clip_percentiles=(2.0, 98.0),
        sample_weight_power=1.5,
        sample_weight_cap_percentile=99.0,
        regime_dampen=0.15,
        target_mean_shrink=0.0,
        calibration_mode="linear",
        calibration_target="pseudo",
        calibration_clip_percentiles=(2.0, 98.0),
        teacher_params=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._teacher_weight = float(teacher_weight)
        self._adaptive_teacher_mix = bool(adaptive_teacher_mix)
        self._teacher_weight_min = float(teacher_weight_min)
        self._teacher_regime_strength = float(teacher_regime_strength)
        self._pseudo_clip = float(pseudo_clip)
        self._pseudo_target_clip_percentiles = pseudo_target_clip_percentiles
        self._sample_weight_power = float(sample_weight_power)
        self._sample_weight_cap_percentile = float(sample_weight_cap_percentile)
        self._regime_dampen = float(regime_dampen)
        self._target_mean_shrink = float(target_mean_shrink)
        self._calibration_mode = str(calibration_mode)
        self._calibration_target = str(calibration_target)
        self._calibration_clip_percentiles = calibration_clip_percentiles
        self._teacher_params = teacher_params or {}
        self._student_model_template = _make_final_regressor(
            student_model_kind,
            random_state=self._random_state,
            n_estimators=student_model_trees,
            min_samples_leaf=student_model_min_samples_leaf,
            n_jobs=1,
            max_depth=student_model_max_depth,
            max_features=student_model_max_features,
            learning_rate=student_learning_rate,
        )
        self._student = None
        self._calibrator = None

    def _make_teacher(self):
        teacher_params = dict(
            target=self._target,
            horizon=self._horizon,
            cv=self._cv,
            random_state=self._random_state,
            censoring_estimator=self._censoring_estimator,
            observed_only=False,
            n_estimators=self._n_estimators,
            min_samples_leaf=self._min_samples_leaf,
            q_kind="logit",
            h_kind="extra",
            h_n_estimators=800,
            h_min_samples_leaf=5,
            summary_feature_mode=self._summary_feature_mode,
            nuisance_feature_mode=self._nuisance_feature_mode,
            summary_curve_knots=self._summary_curve_knots,
            forest_honest=self._forest_honest,
            forest_inference=self._forest_inference,
            forest_fit_intercept=self._forest_fit_intercept,
            forest_max_depth=self._forest_max_depth,
        )
        teacher_params.update(self._teacher_params)
        return TwoStageBridgeSummarySurvivalForest(**teacher_params)

    def _build_student_target(self, teacher_target, pseudo_target, regime_meta):
        if self._adaptive_teacher_mix:
            regime_score = _regime_priority_score(regime_meta)
            teacher_weight = self._teacher_weight * np.exp(-self._teacher_regime_strength * regime_score)
            teacher_weight = np.clip(teacher_weight, self._teacher_weight_min, self._teacher_weight)
            student_target = teacher_weight * teacher_target + (1.0 - teacher_weight) * pseudo_target
        else:
            student_target = self._teacher_weight * teacher_target + (1.0 - self._teacher_weight) * pseudo_target
        if self._target_mean_shrink > 0.0:
            target_mean = float(np.mean(student_target))
            student_target = (1.0 - self._target_mean_shrink) * student_target + self._target_mean_shrink * target_mean
        return np.asarray(student_target, dtype=float)

    def _build_sample_weights(self, a_res, regime_meta):
        weights = np.power(np.abs(np.asarray(a_res, dtype=float)), self._sample_weight_power)
        weights = np.clip(weights, 1e-4, None)
        if self._regime_dampen > 0.0:
            regime_score = _regime_priority_score(regime_meta)
            weights = weights / (1.0 + self._regime_dampen * regime_score)
        if self._sample_weight_cap_percentile < 100.0:
            cap = float(np.percentile(weights, self._sample_weight_cap_percentile))
            weights = np.clip(weights, None, max(cap, 1e-4))
        return weights

    def _calibration_targets(self, teacher_target, pseudo_target, student_target):
        target_map = {
            "teacher": teacher_target,
            "pseudo": pseudo_target,
            "student": student_target,
            "blend": 0.5 * teacher_target + 0.5 * pseudo_target,
        }
        base = np.asarray(target_map.get(self._calibration_target, pseudo_target), dtype=float)
        lo, hi = np.percentile(base, self._calibration_clip_percentiles)
        return np.clip(base, lo, hi)

    def _fit_calibrator(self, base_pred, cal_target, weights):
        mode = self._calibration_mode.lower()
        if mode in {"none", "off"}:
            self._calibrator = None
            return
        base_pred = np.asarray(base_pred, dtype=float).reshape(-1)
        cal_target = np.asarray(cal_target, dtype=float).reshape(-1)
        if mode == "linear":
            self._calibrator = LinearRegression()
            self._calibrator.fit(base_pred.reshape(-1, 1), cal_target, sample_weight=weights)
            return
        if mode == "ridge":
            self._calibrator = Ridge(alpha=1.0)
            self._calibrator.fit(base_pred.reshape(-1, 1), cal_target, sample_weight=weights)
            return
        if mode == "isotonic":
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(base_pred, cal_target, sample_weight=weights)
            return
        raise ValueError(f"Unsupported calibration_mode: {self._calibration_mode}")

    def _apply_calibrator(self, base_pred):
        if self._calibrator is None:
            return np.asarray(base_pred, dtype=float)
        if isinstance(self._calibrator, IsotonicRegression):
            return np.asarray(self._calibrator.predict(np.asarray(base_pred, dtype=float).reshape(-1)), dtype=float)
        return np.asarray(self._calibrator.predict(np.asarray(base_pred, dtype=float).reshape(-1, 1)), dtype=float)

    def fit_components(self, X, A, time, event, Z, W):
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        x, _, w_nuis, z_nuis, x_final, y_res, a_res, regime_meta = _crossfit_summary_arrays(
            self,
            X,
            A,
            y_packed,
            W,
            Z,
            return_regime_meta=True,
        )
        teacher_target = _crossfit_single_teacher_target(self, self._make_teacher, x, A, time, event, w_nuis, z_nuis)

        denom = np.where(np.abs(a_res) >= self._pseudo_clip, a_res, np.sign(a_res) * self._pseudo_clip)
        denom[np.abs(denom) < self._pseudo_clip] = self._pseudo_clip
        pseudo_target = y_res / denom
        lo, hi = np.percentile(pseudo_target, self._pseudo_target_clip_percentiles)
        pseudo_target = np.clip(pseudo_target, lo, hi)

        student_target = self._build_student_target(teacher_target, pseudo_target, regime_meta)
        weights = self._build_sample_weights(a_res, regime_meta)

        self._student = clone(self._student_model_template)
        self._student.fit(x_final, student_target, sample_weight=weights)

        base_pred = np.asarray(self._student.predict(x_final), dtype=float)
        cal_target = self._calibration_targets(teacher_target, pseudo_target, student_target)
        self._fit_calibrator(base_pred, cal_target, weights)

        self._full_nuisance = self._make_nuisance()
        self._full_nuisance.train(
            False,
            None,
            y_packed,
            np.asarray(A, dtype=float).ravel(),
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._student is None or self._full_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        x = _ensure_2d(X).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._full_nuisance.predict_bridge_outputs(
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        x_final = _build_summary_features(
            x,
            bridge,
            mode=self._summary_feature_mode,
            n_curve_knots=self._summary_curve_knots,
        )
        base_pred = np.asarray(self._student.predict(x_final), dtype=float)
        return self._apply_calibrator(base_pred)


class PairwiseRankingSingleTeacherBridgeSummaryRegressor(_BaseTwoStageSummarySurvivalForest):
    def __init__(
        self,
        *args,
        hidden_dims=(192, 96, 48),
        dropout=0.05,
        learning_rate=2e-3,
        weight_decay=1e-4,
        max_epochs=450,
        patience=45,
        val_fraction=0.15,
        point_weight=1.0,
        corr_weight=0.35,
        rank_weight=0.85,
        pair_samples=4096,
        pair_weight_mode="inverse",
        near_null_boost=1.25,
        teacher_weight=1.0,
        adaptive_teacher_mix=False,
        teacher_weight_min=0.0,
        teacher_regime_strength=1.0,
        pseudo_clip=0.05,
        pseudo_target_clip_percentiles=(1.0, 99.0),
        teacher_params=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._hidden_dims = tuple(int(v) for v in hidden_dims)
        self._dropout = float(dropout)
        self._learning_rate = float(learning_rate)
        self._weight_decay = float(weight_decay)
        self._max_epochs = int(max_epochs)
        self._patience = int(patience)
        self._val_fraction = float(val_fraction)
        self._point_weight = float(point_weight)
        self._corr_weight = float(corr_weight)
        self._rank_weight = float(rank_weight)
        self._pair_samples = int(pair_samples)
        self._pair_weight_mode = str(pair_weight_mode)
        self._near_null_boost = float(near_null_boost)
        self._teacher_weight = float(teacher_weight)
        self._adaptive_teacher_mix = bool(adaptive_teacher_mix)
        self._teacher_weight_min = float(teacher_weight_min)
        self._teacher_regime_strength = float(teacher_regime_strength)
        self._pseudo_clip = float(pseudo_clip)
        self._pseudo_target_clip_percentiles = pseudo_target_clip_percentiles
        self._teacher_params = teacher_params or {}

        self._student = None
        self._feature_mean = None
        self._feature_scale = None
        self._target_mean = None
        self._target_scale = None

    def _make_teacher(self):
        teacher_params = dict(
            target=self._target,
            horizon=self._horizon,
            cv=self._cv,
            random_state=self._random_state,
            censoring_estimator=self._censoring_estimator,
            observed_only=False,
            n_estimators=self._n_estimators,
            min_samples_leaf=self._min_samples_leaf,
            q_kind="logit",
            h_kind="extra",
            h_n_estimators=800,
            h_min_samples_leaf=5,
            summary_feature_mode=self._summary_feature_mode,
            nuisance_feature_mode=self._nuisance_feature_mode,
            summary_curve_knots=self._summary_curve_knots,
            forest_honest=self._forest_honest,
            forest_inference=self._forest_inference,
            forest_fit_intercept=self._forest_fit_intercept,
            forest_max_depth=self._forest_max_depth,
        )
        teacher_params.update(self._teacher_params)
        return TwoStageBridgeSummarySurvivalForest(**teacher_params)

    def _fit_torch_student(self, x_final, student_target, sample_weight, regime_meta):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        rng = np.random.default_rng(self._random_state)
        indices = np.arange(x_final.shape[0])
        rng.shuffle(indices)
        n_val = max(1, int(round(self._val_fraction * len(indices))))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        if train_idx.size == 0:
            train_idx = val_idx
            val_idx = indices[:1]

        x_train = np.asarray(x_final[train_idx], dtype=float)
        x_val = np.asarray(x_final[val_idx], dtype=float)
        y_train = np.asarray(student_target[train_idx], dtype=float)
        y_val = np.asarray(student_target[val_idx], dtype=float)
        w_train = np.asarray(sample_weight[train_idx], dtype=float)
        w_val = np.asarray(sample_weight[val_idx], dtype=float)
        regime_train = _regime_priority_score(regime_meta[train_idx])
        regime_val = _regime_priority_score(regime_meta[val_idx])

        self._feature_mean = x_train.mean(axis=0, keepdims=True)
        self._feature_scale = x_train.std(axis=0, keepdims=True) + 1e-6
        self._target_mean = float(y_train.mean())
        self._target_scale = float(y_train.std() + 1e-6)

        x_train = (x_train - self._feature_mean) / self._feature_scale
        x_val = (x_val - self._feature_mean) / self._feature_scale
        y_train = (y_train - self._target_mean) / self._target_scale
        y_val = (y_val - self._target_mean) / self._target_scale

        device = torch.device("cpu")
        torch.manual_seed(self._random_state)

        layers = []
        in_dim = x_train.shape[1]
        for hidden_dim in self._hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self._dropout > 0.0:
                layers.append(nn.Dropout(self._dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        model = nn.Sequential(*layers).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)

        x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        w_train_t = torch.tensor(w_train / max(np.mean(w_train), 1e-6), dtype=torch.float32, device=device)
        regime_train_t = torch.tensor(regime_train, dtype=torch.float32, device=device)

        x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
        w_val_t = torch.tensor(w_val / max(np.mean(w_val), 1e-6), dtype=torch.float32, device=device)
        regime_val_t = torch.tensor(regime_val, dtype=torch.float32, device=device)

        def _corr_loss(pred, target):
            pred_center = pred - pred.mean()
            target_center = target - target.mean()
            denom = torch.sqrt(torch.mean(pred_center.square()) * torch.mean(target_center.square()) + 1e-8)
            corr = torch.mean(pred_center * target_center) / denom
            return 1.0 - corr

        def _pairwise_loss(pred, target, regime_score):
            n = pred.shape[0]
            if n <= 1 or self._pair_samples <= 0:
                return pred.new_tensor(0.0)
            idx_i = torch.randint(0, n, (self._pair_samples,), device=pred.device)
            idx_j = torch.randint(0, n, (self._pair_samples,), device=pred.device)
            keep = idx_i != idx_j
            idx_i = idx_i[keep]
            idx_j = idx_j[keep]
            if idx_i.numel() == 0:
                return pred.new_tensor(0.0)

            target_diff = target[idx_i] - target[idx_j]
            pred_diff = pred[idx_i] - pred[idx_j]
            sign = torch.sign(target_diff)
            keep = sign != 0
            if keep.sum() == 0:
                return pred.new_tensor(0.0)
            target_diff = target_diff[keep]
            pred_diff = pred_diff[keep]
            sign = sign[keep]
            idx_i = idx_i[keep]
            idx_j = idx_j[keep]

            if self._pair_weight_mode == "uniform":
                base = torch.ones_like(target_diff)
            elif self._pair_weight_mode == "sqrt_inverse":
                base = torch.rsqrt(torch.abs(target_diff) + 0.05)
            else:
                base = 1.0 / (torch.abs(target_diff) + 0.05)
            base = torch.clamp(base, max=12.0)
            regime_boost = 1.0 + self._near_null_boost * 0.5 * (regime_score[idx_i] + regime_score[idx_j])
            pair_weight = base * regime_boost
            return torch.sum(pair_weight * F.softplus(-sign * pred_diff)) / torch.sum(pair_weight)

        best_val = float("inf")
        best_state = None
        wait = 0
        for _ in range(self._max_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_train_t).reshape(-1)
            point_loss = torch.sum(w_train_t * (pred - y_train_t).square()) / torch.sum(w_train_t)
            corr_loss = _corr_loss(pred, y_train_t)
            rank_loss = _pairwise_loss(pred, y_train_t, regime_train_t)
            loss = self._point_weight * point_loss + self._corr_weight * corr_loss + self._rank_weight * rank_loss
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_val = model(x_val_t).reshape(-1)
                val_point = torch.sum(w_val_t * (pred_val - y_val_t).square()) / torch.sum(w_val_t)
                val_corr = _corr_loss(pred_val, y_val_t)
                val_rank = _pairwise_loss(pred_val, y_val_t, regime_val_t)
                val_loss = (
                    self._point_weight * val_point
                    + self._corr_weight * val_corr
                    + self._rank_weight * val_rank
                ).item()
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self._patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._student = model

    def fit_components(self, X, A, time, event, Z, W):
        y_packed = np.column_stack([np.asarray(time, dtype=float).ravel(), np.asarray(event, dtype=float).ravel()])
        x, _, w_nuis, z_nuis, x_final, y_res, a_res, regime_meta = _crossfit_summary_arrays(
            self,
            X,
            A,
            y_packed,
            W,
            Z,
            return_regime_meta=True,
        )
        teacher_target = _crossfit_single_teacher_target(self, self._make_teacher, x, A, time, event, w_nuis, z_nuis)

        denom = np.where(np.abs(a_res) >= self._pseudo_clip, a_res, np.sign(a_res) * self._pseudo_clip)
        denom[np.abs(denom) < self._pseudo_clip] = self._pseudo_clip
        pseudo_target = y_res / denom
        lo, hi = np.percentile(pseudo_target, self._pseudo_target_clip_percentiles)
        pseudo_target = np.clip(pseudo_target, lo, hi)
        if self._adaptive_teacher_mix:
            regime_score = _regime_priority_score(regime_meta)
            teacher_weight = self._teacher_weight * np.exp(-self._teacher_regime_strength * regime_score)
            teacher_weight = np.clip(teacher_weight, self._teacher_weight_min, self._teacher_weight)
            student_target = teacher_weight * teacher_target + (1.0 - teacher_weight) * pseudo_target
        else:
            student_target = self._teacher_weight * teacher_target + (1.0 - self._teacher_weight) * pseudo_target
        weights = np.square(a_res)

        self._fit_torch_student(x_final, student_target, weights, regime_meta)

        self._full_nuisance = self._make_nuisance()
        self._full_nuisance.train(
            False,
            None,
            y_packed,
            np.asarray(A, dtype=float).ravel(),
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._student is None or self._full_nuisance is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        import torch

        x = _ensure_2d(X).astype(float)
        w_nuis, z_nuis = self._prepare_nuisance_inputs(W, Z)
        bridge = self._full_nuisance.predict_bridge_outputs(
            X=x,
            W=w_nuis,
            Z=z_nuis,
        )
        x_final = _build_summary_features(
            x,
            bridge,
            mode=self._summary_feature_mode,
            n_curve_knots=self._summary_curve_knots,
        )
        x_final = (x_final - self._feature_mean) / self._feature_scale
        with torch.no_grad():
            pred = self._student(torch.tensor(x_final, dtype=torch.float32)).cpu().numpy().reshape(-1)
        return pred * self._target_scale + self._target_mean


class BestCensoredPCISurvivalForest(TwoStageBridgeSummarySurvivalForest):
    """
    Best-performing censored PCI configuration from the full idea suite:
      - two-stage bridge-summary final forest
      - covariate-dependent censoring survival via Cox
      - q bridge: logistic regression
      - h bridge: ExtraTreesRegressor(800 trees, min leaf 5)
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("censoring_estimator", "cox")
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("q_clip", 0.02)
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 800)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("summary_feature_mode", "basic")
        super().__init__(*args, **kwargs)


class BestCensoredPCISummaryBlendSurvivalForest(BlendedTwoStageBridgeSummarySurvivalForest):
    """
    Best-performing censored PCI configuration from the extended summary-cox search:
      - model A: curve_x_interact summary, interact nuisance, ExtraTrees(800, leaf 5)
      - model B: surv_x_interact summary, dup nuisance, ExtraTrees(1200, leaf 5)
      - final prediction: 0.55 * model A + 0.45 * model B
    """

    def __init__(self, *args, **kwargs):
        alpha = kwargs.pop("alpha", 0.55)
        shared = dict(
            target=kwargs.pop("target", "RMST"),
            horizon=kwargs.pop("horizon", None),
            cv=kwargs.pop("cv", 3),
            random_state=kwargs.pop("random_state", 42),
            censoring_estimator=kwargs.pop("censoring_estimator", "cox"),
        )
        model_a_params = dict(
            shared,
            n_estimators=200,
            min_samples_leaf=20,
            q_kind="logit",
            h_kind="extra",
            h_n_estimators=800,
            h_min_samples_leaf=5,
            summary_feature_mode="curve_x_interact",
            nuisance_feature_mode="interact",
        )
        model_b_params = dict(
            shared,
            n_estimators=200,
            min_samples_leaf=20,
            q_kind="logit",
            h_kind="extra",
            h_n_estimators=1200,
            h_min_samples_leaf=5,
            summary_feature_mode="surv_x_interact",
            nuisance_feature_mode="dup",
        )
        super().__init__(alpha=alpha, model_a_params=model_a_params, model_b_params=model_b_params)


class BestPureDistilledCensoredPCISurvivalRegressor(DistilledTwoStageBridgeSummaryRegressor):
    """
    Best pure single-student censored PCI model found so far.

    It distills the strong summary-cox teacher into a single ExtraTrees regressor
    on multi-view summary features. This configuration beats the old censored C3
    and E2 baselines on both RMSE and Pearson, although it does not hit the more
    aggressive Pearson > 0.6 and RMSE < 0.12 target simultaneously.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("target", "RMST")
        kwargs.setdefault("horizon", None)
        kwargs.setdefault("cv", 3)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("censoring_estimator", "cox")
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 800)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("summary_feature_mode", "multi_x_interact")
        kwargs.setdefault("nuisance_feature_mode", "interact")
        kwargs.setdefault("summary_curve_knots", 8)
        kwargs.setdefault("student_model_kind", "extra")
        kwargs.setdefault("student_model_trees", 1200)
        kwargs.setdefault("student_model_min_samples_leaf", 5)
        kwargs.setdefault("teacher_alpha", 0.55)
        kwargs.setdefault("teacher_weight", 1.0)
        super().__init__(*args, **kwargs)


class BestCurveLocalCensoredPCISurvivalForest(TwoStageBridgeSummarySurvivalForest):
    """
    Best low-complexity curve-family censored PCI configuration found by the
    narrow local search around curve_f400_l20_k8.

      - summary feature mode: curve_x_interact
      - curve knots: 8
      - final forest trees: 400
      - min leaf size: 30
      - censoring survival: Cox
      - bridge nuisances: q=logit, h=ExtraTrees(800, leaf 5)
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("target", "RMST")
        kwargs.setdefault("horizon", None)
        kwargs.setdefault("cv", 3)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("censoring_estimator", "cox")
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 800)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("summary_feature_mode", "curve_x_interact")
        kwargs.setdefault("nuisance_feature_mode", "interact")
        kwargs.setdefault("n_estimators", 400)
        kwargs.setdefault("min_samples_leaf", 30)
        kwargs.setdefault("summary_curve_knots", 8)
        super().__init__(*args, **kwargs)


class BestCurveLocalObservedCensoredSurvivalForest(TwoStageObservedSummarySurvivalForest):
    """
    Matched no-PCI baseline for BestCurveLocalCensoredPCISurvivalForest.

    It keeps the same two-stage curve-summary forest architecture and the same
    censoring/final-forest settings, but removes the PCI bridge by forcing the
    nuisance layer to use observed features only.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("target", "RMST")
        kwargs.setdefault("horizon", None)
        kwargs.setdefault("cv", 3)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("censoring_estimator", "cox")
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 800)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("summary_feature_mode", "curve_x_interact")
        kwargs.setdefault("nuisance_feature_mode", "interact")
        kwargs.setdefault("n_estimators", 400)
        kwargs.setdefault("min_samples_leaf", 30)
        kwargs.setdefault("summary_curve_knots", 8)
        super().__init__(*args, **kwargs)


class HybridProxyCensoredPCISurvivalForest(TwoStageBridgeSummarySurvivalForest):
    """
    BestCurveLocal-style censored PCI model with compact retained proxy summaries.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("target", "RMST")
        kwargs.setdefault("horizon", None)
        kwargs.setdefault("cv", 3)
        kwargs.setdefault("random_state", 42)
        kwargs.setdefault("censoring_estimator", "cox")
        kwargs.setdefault("q_kind", "logit")
        kwargs.setdefault("h_kind", "extra")
        kwargs.setdefault("h_n_estimators", 800)
        kwargs.setdefault("h_min_samples_leaf", 5)
        kwargs.setdefault("summary_feature_mode", "curve_proxy_compact_x_interact")
        kwargs.setdefault("nuisance_feature_mode", "interact")
        kwargs.setdefault("n_estimators", 400)
        kwargs.setdefault("min_samples_leaf", 40)
        kwargs.setdefault("summary_curve_knots", 8)
        super().__init__(*args, **kwargs)
