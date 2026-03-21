from __future__ import annotations

import json
import math
import os
import textwrap
import time
from functools import lru_cache, partial
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("XDG_CACHE_HOME", str(_PROJECT_ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .data_generation import (
    SynthConfig,
    add_ground_truth_cate,
    calibrate_intercept_for_prevalence,
    generate_synthetic_nc_cox,
    sigmoid,
    weibull_ph_time_paper,
)
from .models import BaselineCausalForestDML, MildShrinkNCCausalForestDML


FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
TITLE_SUFFIX = "n=2000, p=5, seed=42, censoring rate=0%"

TABLE_COLUMNS = ["Variant", "Pred CATE", "True CATE", "Bias", "RMSE", "PEHE", "MAE", "Pearson", "Time"]
TABLE_KEYS = [
    "name",
    "mean_pred",
    "mean_true_cate",
    "bias",
    "rmse",
    "pehe",
    "mae",
    "pearson",
    "time_str",
]

SUMMARY_COLUMNS = [
    "Rank",
    "Variant",
    "Avg Pred CATE",
    "Avg True CATE",
    "Avg Acc",
    "Avg RMSE",
    "Avg PEHE",
    "Avg MAE",
    "Avg Pearson",
    "Avg Bias",
]
SUMMARY_KEYS = [
    "rank",
    "name",
    "avg_pred_cate",
    "avg_true_cate",
    "avg_acc",
    "avg_rmse",
    "avg_pehe",
    "avg_mae",
    "avg_pearson",
    "avg_bias",
]

TOP5_COLUMNS = ["Rank", "Variant", "Avg RMSE", "Avg PEHE", "Avg MAE", "Avg Pearson", "Avg Bias"]
TOP5_KEYS = ["rank", "name", "avg_rmse", "avg_pehe", "avg_mae", "avg_pearson", "avg_bias"]

BASE_CONFIG = dict(
    n=2000,
    p_x=5,
    seed=42,
    a_prevalence=0.5,
    gamma_u_in_a=0.9,
    k_t=1.5,
    lam_t=0.4,
    tau_log_hr=-0.7,
    beta_u_in_t=1.1,
    k_c=1.2,
    lam_c=1e6,
    beta_u_in_c=0.3,
    target_censor_rate=0.0,
    max_censor_calib_iter=60,
    censor_lam_lo=1e-8,
    censor_lam_hi=1e6,
    admin_censor_time=None,
    aZ=1.5,
    sigma_z=1.125,
    aW=1.5,
    sigma_w=1.53,
)

CASE_SPECS = [
    {
        "case_id": 1,
        "slug": "linear_linear_informative_strong_beneficial_large",
        "title": f"Linear treatment / linear outcome DGP, informative proxies, strong confounding, large beneficial treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.9, beta_u_in_t=1.1, tau_log_hr=-0.7, sigma_z=1.125, sigma_w=1.53, linear_treatment=True, linear_outcome=True),
    },
    {
        "case_id": 2,
        "slug": "linear_linear_informative_weak_harmful_small",
        "title": f"Linear treatment / linear outcome DGP, informative proxies, weak confounding, small harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.2, beta_u_in_t=0.2, tau_log_hr=0.25, sigma_z=1.125, sigma_w=1.53, linear_treatment=True, linear_outcome=True),
    },
    {
        "case_id": 3,
        "slug": "linear_linear_weakproxy_strong_harmful_moderate",
        "title": f"Linear treatment / linear outcome DGP, weak proxies, strong confounding, moderate harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.9, beta_u_in_t=1.1, tau_log_hr=0.5, sigma_z=7.35, sigma_w=4.77, linear_treatment=True, linear_outcome=True),
    },
    {
        "case_id": 4,
        "slug": "linear_linear_weakproxy_weak_nearnull_harmful",
        "title": f"Linear treatment / linear outcome DGP, weak proxies, weak confounding, near-null harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.25, beta_u_in_t=0.2, tau_log_hr=0.12, sigma_z=7.35, sigma_w=4.77, linear_treatment=True, linear_outcome=True),
    },
    {
        "case_id": 5,
        "slug": "linear_nonlinear_informative_strong_beneficial_large",
        "title": f"Linear treatment / nonlinear outcome DGP, informative proxies, strong confounding, large beneficial treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.9, beta_u_in_t=1.1, tau_log_hr=-0.7, sigma_z=1.125, sigma_w=1.53, linear_treatment=True, linear_outcome=False),
    },
    {
        "case_id": 6,
        "slug": "linear_nonlinear_informative_weak_harmful_small",
        "title": f"Linear treatment / nonlinear outcome DGP, informative proxies, weak confounding, small harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.2, beta_u_in_t=0.2, tau_log_hr=0.3, sigma_z=1.125, sigma_w=1.53, linear_treatment=True, linear_outcome=False),
    },
    {
        "case_id": 7,
        "slug": "linear_nonlinear_weakproxy_strong_harmful_moderate",
        "title": f"Linear treatment / nonlinear outcome DGP, weak proxies, strong confounding, moderate harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.9, beta_u_in_t=1.1, tau_log_hr=0.5, sigma_z=7.35, sigma_w=4.77, linear_treatment=True, linear_outcome=False),
    },
    {
        "case_id": 8,
        "slug": "linear_nonlinear_weakproxy_weak_nearnull_harmful",
        "title": f"Linear treatment / nonlinear outcome DGP, weak proxies, weak confounding, near-null harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.25, beta_u_in_t=0.2, tau_log_hr=0.12, sigma_z=7.35, sigma_w=4.77, linear_treatment=True, linear_outcome=False),
    },
    {
        "case_id": 9,
        "slug": "nonlinear_nonlinear_informative_strong_beneficial_large",
        "title": f"Nonlinear treatment / nonlinear outcome DGP, informative proxies, strong confounding, large beneficial treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.9, beta_u_in_t=1.1, tau_log_hr=-0.7, sigma_z=1.125, sigma_w=1.53, linear_treatment=False, linear_outcome=False),
    },
    {
        "case_id": 10,
        "slug": "nonlinear_nonlinear_informative_weak_harmful_small",
        "title": f"Nonlinear treatment / nonlinear outcome DGP, informative proxies, weak confounding, small harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.2, beta_u_in_t=0.2, tau_log_hr=0.3, sigma_z=1.125, sigma_w=1.53, linear_treatment=False, linear_outcome=False),
    },
    {
        "case_id": 11,
        "slug": "nonlinear_nonlinear_weakproxy_strong_harmful_moderate",
        "title": f"Nonlinear treatment / nonlinear outcome DGP, weak proxies, strong confounding, moderate harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.9, beta_u_in_t=1.1, tau_log_hr=0.5, sigma_z=7.35, sigma_w=4.77, linear_treatment=False, linear_outcome=False),
    },
    {
        "case_id": 12,
        "slug": "nonlinear_nonlinear_weakproxy_weak_nearnull_harmful",
        "title": f"Nonlinear treatment / nonlinear outcome DGP, weak proxies, weak confounding, near-null harmful treatment effect, {TITLE_SUFFIX}",
        "cfg": dict(gamma_u_in_a=0.25, beta_u_in_t=0.2, tau_log_hr=0.12, sigma_z=7.35, sigma_w=4.77, linear_treatment=False, linear_outcome=False),
    },
]

VARIANT_SPECS = [
    {"name": "A1  Oracle (all true q/h)", "kind": "oracle", "use_true_q": True, "use_true_h": True},
    {"name": "A2  Oracle (true q, est h)", "kind": "oracle", "use_true_q": True, "use_true_h": False},
    {"name": "A3  Oracle (all estimated q/h)", "kind": "oracle", "use_true_q": False, "use_true_h": False},
    {"name": "B1  EconML baseline (X only)", "kind": "baseline_x"},
    {"name": "B2  EconML baseline (X+W+Z)", "kind": "baseline_xwz"},
    {"name": "C1  NC-CSF (all true q/h)", "kind": "nc", "use_true_q": True, "use_true_h": True},
    {"name": "C2  NC-CSF (true q, est h)", "kind": "nc", "use_true_q": True, "use_true_h": False},
    {"name": "C3  NC-CSF (all estimated q/h)", "kind": "nc", "use_true_q": False, "use_true_h": False},
]


def _build_treatment_nonlinear_features(X):
    p = X.shape[1]
    return np.column_stack(
        [sigmoid(X[:, i]) for i in range(min(3, p))]
        + [X[:, i] * X[:, (i + 1) % p] for i in range(min(2, p))]
    )


def _outcome_nonlinear_part(X, U, dgp):
    beta_squared = dgp.get("beta_squared")
    beta_interact = dgp.get("beta_interact")
    if beta_squared is None or beta_interact is None:
        raise ValueError("Missing nonlinear outcome coefficients.")

    X_squared = X[:, : min(2, X.shape[1])] ** 2
    squared_term = X_squared @ beta_squared
    interact_coef = float(np.asarray(beta_interact).ravel()[0])
    U_arr = np.asarray(U)

    if U_arr.ndim == 1:
        return squared_term + interact_coef * (X[:, 0] * U_arr) + 0.5 * sigmoid(U_arr)
    return squared_term[:, np.newaxis] + interact_coef * (X[:, 0:1] * U_arr) + 0.5 * sigmoid(U_arr)


def _treatment_signal(X, U, dgp, cfg):
    U_arr = np.asarray(U)
    if cfg.linear_treatment:
        base = X @ dgp["alpha"]
        if U_arr.ndim == 1:
            return base + cfg.gamma_u_in_a * U_arr
        return base[:, np.newaxis] + cfg.gamma_u_in_a * U_arr

    features = _build_treatment_nonlinear_features(X)
    base = features @ dgp["alpha_nonlinear"]
    if U_arr.ndim == 1:
        return base + cfg.gamma_u_in_a * sigmoid(U_arr)
    return base[:, np.newaxis] + cfg.gamma_u_in_a * sigmoid(U_arr)


def _outcome_signal(X, U, dgp, cfg):
    U_arr = np.asarray(U)
    base = X @ dgp["beta_t"]
    if U_arr.ndim == 1:
        signal = base + cfg.beta_u_in_t * U_arr
    else:
        signal = base[:, np.newaxis] + cfg.beta_u_in_t * U_arr
    if not cfg.linear_outcome:
        signal = signal + _outcome_nonlinear_part(X, U_arr, dgp)
    return signal


@lru_cache(maxsize=None)
def _hermite_rule(order):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return nodes, weights / np.sqrt(np.pi)


def _gaussian_expectation(mu, var, fn, order=40):
    mu = np.asarray(mu, dtype=float)
    var = np.asarray(var, dtype=float)
    nodes, weights = _hermite_rule(order)
    sigma = np.sqrt(np.maximum(var, 0.0))
    support = mu[:, np.newaxis] + np.sqrt(2.0) * sigma[:, np.newaxis] * nodes[np.newaxis, :]
    values = fn(support)
    return np.sum(values * weights[np.newaxis, :], axis=1)


def recover_dgp_internals(cfg):
    rng = np.random.default_rng(cfg.seed)
    n, p = cfg.n, cfg.p_x

    X = rng.normal(size=(n, p))
    U = rng.normal(size=n)
    b_z = rng.normal(scale=0.3, size=(p, cfg.p_z))
    b_w = rng.normal(scale=0.3, size=(p, cfg.p_w))
    _ = rng.normal(scale=cfg.sigma_z, size=(n, cfg.p_z))
    _ = rng.normal(scale=cfg.sigma_w, size=(n, cfg.p_w))

    alpha = rng.normal(scale=0.5, size=p)
    alpha_nonlinear = None
    if cfg.linear_treatment:
        linpred = X @ alpha + cfg.gamma_u_in_a * U
    else:
        x_nonlinear = _build_treatment_nonlinear_features(X)
        alpha_nonlinear = rng.normal(scale=0.3, size=x_nonlinear.shape[1])
        linpred = x_nonlinear @ alpha_nonlinear + cfg.gamma_u_in_a * sigmoid(U)
    b0 = calibrate_intercept_for_prevalence(linpred, cfg.a_prevalence)
    p_a = sigmoid(b0 + linpred)
    A = rng.binomial(1, p_a, size=n).astype(int)

    beta_t = rng.normal(scale=0.4, size=p)
    u_t = rng.random(n)

    beta_squared = None
    beta_interact = None
    if cfg.linear_outcome:
        eta_t0 = X @ beta_t + cfg.beta_u_in_t * U
    else:
        X_squared = X[:, : min(2, p)] ** 2
        X_interact = X[:, 0:1] * U.reshape(-1, 1)
        beta_squared = rng.normal(scale=0.2, size=X_squared.shape[1])
        beta_interact = rng.normal(scale=0.2, size=X_interact.shape[1])
        nonlinear_part = X_squared @ beta_squared + X_interact @ beta_interact + 0.5 * sigmoid(U)
        eta_t0 = X @ beta_t + cfg.beta_u_in_t * U + nonlinear_part
    eta_t1 = eta_t0 + cfg.tau_log_hr
    T0 = weibull_ph_time_paper(u_t, k=cfg.k_t, lam=cfg.lam_t, eta=eta_t0)
    T1 = weibull_ph_time_paper(u_t, k=cfg.k_t, lam=cfg.lam_t, eta=eta_t1)

    beta_c = rng.normal(scale=0.3, size=p)
    u_c = rng.random(n)
    eta_c = X @ beta_c + cfg.beta_u_in_c * U
    T_obs = np.where(A == 1, T1, T0)

    lam_c = cfg.lam_c
    if lam_c is None:
        lo, hi = float(cfg.censor_lam_lo), float(cfg.censor_lam_hi)
        for _ in range(cfg.max_censor_calib_iter):
            mid = 0.5 * (lo + hi)
            c_mid = weibull_ph_time_paper(u_c, k=cfg.k_c, lam=mid, eta=eta_c)
            if (c_mid < T_obs).mean() < cfg.target_censor_rate:
                hi = mid
            else:
                lo = mid
        lam_c = 0.5 * (lo + hi)

    return {
        "X": X,
        "U": U,
        "A": A,
        "alpha": alpha,
        "alpha_nonlinear": alpha_nonlinear,
        "b0": b0,
        "beta_t": beta_t,
        "beta_c": beta_c,
        "beta_squared": beta_squared,
        "beta_interact": beta_interact,
        "b_z": b_z,
        "b_w": b_w,
        "lam_c": lam_c,
    }


def true_propensity_oracle(X, U, dgp, cfg):
    return sigmoid(dgp["b0"] + _treatment_signal(X, U, dgp, cfg))


def true_propensity_nc(X, Z, dgp, cfg):
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    b_z = np.asarray(dgp["b_z"], dtype=float)
    if b_z.ndim == 1:
        b_z = b_z.reshape(-1, 1)
    sz2 = cfg.sigma_z ** 2
    var_post = 1.0 / (1.0 + Z.shape[1] * (cfg.aZ ** 2) / sz2)
    mu_post = var_post * (cfg.aZ / sz2) * np.sum(Z - X @ b_z, axis=1)

    if cfg.linear_treatment:
        loc = dgp["b0"] + X @ dgp["alpha"] + cfg.gamma_u_in_a * mu_post
        return sigmoid(loc / np.sqrt(1 + np.pi / 8 * cfg.gamma_u_in_a ** 2 * var_post))

    return _gaussian_expectation(
        mu_post,
        np.full_like(mu_post, var_post, dtype=float),
        lambda u: sigmoid(dgp["b0"] + _treatment_signal(X, u, dgp, cfg)),
    )


def true_outcome_oracle(X, U, cfg, dgp):
    G = math.gamma(1.0 + 1.0 / cfg.k_t)
    signal = _outcome_signal(X, U, dgp, cfg)
    h0 = cfg.lam_t * G * np.exp(-signal / cfg.k_t)
    h1 = cfg.lam_t * G * np.exp(-(signal + cfg.tau_log_hr) / cfg.k_t)
    return h0, h1


def true_outcome_nc(X, W, cfg, dgp):
    W = np.asarray(W, dtype=float)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    b_w = np.asarray(dgp["b_w"], dtype=float)
    if b_w.ndim == 1:
        b_w = b_w.reshape(-1, 1)
    sw2 = cfg.sigma_w ** 2
    var_w = 1.0 / (1.0 + W.shape[1] * (cfg.aW ** 2) / sw2)
    mu_w = var_w * (cfg.aW / sw2) * np.sum(W - X @ b_w, axis=1)
    G = math.gamma(1.0 + 1.0 / cfg.k_t)

    if cfg.linear_outcome:
        c = -cfg.beta_u_in_t / cfg.k_t
        mgf = np.exp(c * mu_w + 0.5 * c ** 2 * var_w)
        xb = X @ dgp["beta_t"]
        h0 = cfg.lam_t * G * np.exp(-xb / cfg.k_t) * mgf
        h1 = cfg.lam_t * G * np.exp(-(xb + cfg.tau_log_hr) / cfg.k_t) * mgf
        return h0, h1

    var_vec = np.full_like(mu_w, var_w, dtype=float)
    h0 = _gaussian_expectation(
        mu_w,
        var_vec,
        lambda u: cfg.lam_t * G * np.exp(-_outcome_signal(X, u, dgp, cfg) / cfg.k_t),
    )
    h1 = _gaussian_expectation(
        mu_w,
        var_vec,
        lambda u: cfg.lam_t * G * np.exp(-(_outcome_signal(X, u, dgp, cfg) + cfg.tau_log_hr) / cfg.k_t),
    )
    return h0, h1


def oracle_q_from_proxy(X, proxy, dgp, cfg):
    return true_propensity_oracle(X, np.asarray(proxy).ravel(), dgp, cfg)


def oracle_h_from_proxy(X, proxy, cfg, dgp):
    return true_outcome_oracle(X, np.asarray(proxy).ravel(), cfg, dgp)


def nc_q_from_proxy(X, proxy, dgp, cfg):
    _, Z = proxy
    return true_propensity_nc(X, Z, dgp, cfg)


def nc_h_from_proxy(X, proxy, cfg, dgp):
    W, _ = proxy
    return true_outcome_nc(X, W, cfg, dgp)


def _format_seconds(seconds):
    return f"{seconds:.1f}s"


def _metric_row(name, preds, true_cate, elapsed):
    preds = np.asarray(preds, dtype=float).ravel()
    true_cate = np.asarray(true_cate, dtype=float).ravel()
    rmse = float(np.sqrt(np.mean((preds - true_cate) ** 2)))
    return {
        "name": name,
        "mean_pred": float(np.mean(preds)),
        "mean_true_cate": float(np.mean(true_cate)),
        "bias": float(np.mean(preds - true_cate)),
        "rmse": rmse,
        "pehe": rmse,
        "mae": float(np.mean(np.abs(preds - true_cate))),
        "pearson": float(np.corrcoef(preds, true_cate)[0, 1]),
        "sign_acc": float(np.mean(np.sign(preds) == np.sign(true_cate))),
        "time_sec": float(elapsed),
        "time_str": _format_seconds(elapsed),
    }


def _evaluate_baseline(name, X_train, A, Y, X_effect, true_cate, seed):
    start = time.time()
    model = BaselineCausalForestDML(n_estimators=200, min_samples_leaf=20, random_state=seed, cv=5)
    model.fit_baseline(X_train, A, Y)
    preds = model.effect(X_effect).ravel()
    return _metric_row(name, preds, true_cate, time.time() - start)


def _render_table_png(title, rows, output_path, columns, keys, *, dark=False, meta="8-variant benchmark", col_widths=None):
    font_title = ImageFont.truetype(FONT_PATH, 28)
    font_header = ImageFont.truetype(FONT_PATH, 20)
    font_body = ImageFont.truetype(FONT_PATH, 18)
    font_meta = ImageFont.truetype(FONT_PATH, 18)

    wrapped_title = textwrap.wrap(title, width=90)
    if col_widths is None:
        col_widths = [520] + [120] * (len(columns) - 2) + [90]
    left_margin = 24
    right_margin = 24
    top_margin = 24
    bottom_margin = 24
    title_line_h = 38
    meta_h = 30
    title_gap = 18
    header_h = 54
    row_h = 50

    width = left_margin + sum(col_widths) + right_margin
    height = top_margin + len(wrapped_title) * title_line_h + meta_h + title_gap + header_h + len(rows) * row_h + bottom_margin

    bg = "#06101f" if dark else "white"
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    if dark:
        header_bg = "#1b2533"
        header_fg = "white"
        stripe = "#08162b"
        stripe_alt = "#051224"
        text = "#f8fafc"
        meta_fg = "#cbd5e1"
        edge = "#334155"
    else:
        header_bg = "#1f2937"
        header_fg = "white"
        stripe = "#f3f4f6"
        stripe_alt = "white"
        text = "#111827"
        meta_fg = "#4b5563"
        edge = "#d1d5db"

    def text_size(value, font):
        box = draw.textbbox((0, 0), str(value), font=font)
        return box[2] - box[0], box[3] - box[1]

    y = top_margin
    for line in wrapped_title:
        tw, _ = text_size(line, font_title)
        draw.text(((width - tw) / 2, y), line, fill=text, font=font_title)
        y += title_line_h

    tw, _ = text_size(meta, font_meta)
    draw.text(((width - tw) / 2, y), meta, fill=meta_fg, font=font_meta)
    y += meta_h + title_gap

    x = left_margin
    for cw, head in zip(col_widths, columns):
        draw.rectangle([x, y, x + cw, y + header_h], fill=header_bg, outline=edge, width=2)
        tw, th = text_size(head, font_header)
        draw.text((x + (cw - tw) / 2, y + (header_h - th) / 2 - 1), head, fill=header_fg, font=font_header)
        x += cw

    for i, row in enumerate(rows):
        y0 = y + header_h + i * row_h
        fill = stripe if i % 2 else stripe_alt
        x = left_margin
        for j, (cw, key) in enumerate(zip(col_widths, keys)):
            value = row[key]
            if key.endswith("_str") or isinstance(value, str):
                value = str(value)
            elif key not in {"name", "time_str", "rank"}:
                if key == "avg_acc":
                    value = f"{100.0 * row[key]:.1f}%"
                else:
                    value = f"{row[key]:.4f}"
            else:
                value = str(value)
            draw.rectangle([x, y0, x + cw, y0 + row_h], fill=fill, outline=edge, width=1)
            tw, th = text_size(value, font_body)
            tx = x + 14 if j == 1 or key == "name" else x + (cw - tw) / 2
            if key == "rank":
                tx = x + (cw - tw) / 2
            draw.text((tx, y0 + (row_h - th) / 2 - 1), value, fill=text, font=font_body)
            x += cw

    img.save(output_path)


def _render_b2_c3_plot(case_df, summary_df, output_path):
    b2 = case_df[case_df["name"] == "B2  EconML baseline (X+W+Z)"].sort_values("case_id")
    c3 = case_df[case_df["name"] == "C3  NC-CSF (all estimated q/h)"].sort_values("case_id")

    labels = []
    for slug in b2["case_slug"]:
        parts = slug.split("_")
        labels.append(" / ".join(parts[:5]))

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.2])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])

    y_pos = np.arange(len(labels))
    height = 0.38
    ax_left.barh(y_pos - height / 2, c3["pehe"], height=height, color="#2a9d8f", label="C3 NC-CSF (all estimated q/h)")
    ax_left.barh(y_pos + height / 2, b2["pehe"], height=height, color="#e76f51", label="B2 EconML baseline (X+W+Z)")
    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(labels, fontsize=10)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("sqrt-PEHE")
    ax_left.set_title("Case-wise sqrt-PEHE comparison", fontsize=18, weight="bold")
    ax_left.grid(axis="x", alpha=0.25)
    ax_left.legend(loc="lower right")

    c3_wins = int((c3["pehe"].to_numpy() < b2["pehe"].to_numpy()).sum())
    b2_wins = int((b2["pehe"].to_numpy() < c3["pehe"].to_numpy()).sum())
    ax_right.bar(["C3", "B2"], [c3_wins, b2_wins], color=["#2a9d8f", "#e76f51"], width=0.6)
    ax_right.set_ylim(0, len(labels) + 2)
    ax_right.set_title("Who wins more cases?", fontsize=18, weight="bold")
    ax_right.set_ylabel("Number of cases won")
    for x, val in enumerate([c3_wins, b2_wins]):
        ax_right.text(x, val + 0.2, str(val), ha="center", va="bottom", fontsize=16, weight="bold")

    c3_summary = summary_df[summary_df["name"] == "C3  NC-CSF (all estimated q/h)"].iloc[0]
    b2_summary = summary_df[summary_df["name"] == "B2  EconML baseline (X+W+Z)"].iloc[0]
    note = (
        f"Primary criterion: sqrt-PEHE\n"
        f"C3 avg sqrt-PEHE {c3_summary['avg_pehe']:.4f}\n"
        f"B2 avg sqrt-PEHE {b2_summary['avg_pehe']:.4f}\n"
        f"C3 avg Pearson {c3_summary['avg_pearson']:.4f}\n"
        f"B2 avg Pearson {b2_summary['avg_pearson']:.4f}"
    )
    ax_right.text(0.06, 0.05, note, transform=ax_right.transAxes, fontsize=12, va="bottom", bbox=dict(boxstyle="round", facecolor="#f3f4f6", alpha=0.95))

    fig.suptitle("C3 vs B2 across all 12 cases\nsqrt-PEHE-based comparison", fontsize=24, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_case(cfg, case_spec):
    obs_df, truth_df, params = generate_synthetic_nc_cox(cfg)
    obs_df, truth_df = add_ground_truth_cate(obs_df, truth_df, cfg, params)
    dgp = recover_dgp_internals(cfg)
    x_cols = [f"X{j}" for j in range(cfg.p_x)]
    w_cols = ["W"] if "W" in obs_df.columns else sorted(col for col in obs_df.columns if col.startswith("W"))
    z_cols = ["Z"] if "Z" in obs_df.columns else sorted(col for col in obs_df.columns if col.startswith("Z"))
    X = obs_df[x_cols].to_numpy()
    return {
        "obs_df": obs_df,
        "truth_df": truth_df,
        "dgp": dgp,
        "cfg": cfg,
        "x_cols": x_cols,
        "w_cols": w_cols,
        "z_cols": z_cols,
        "case_spec": case_spec,
        "X": X,
        "W": obs_df[w_cols].to_numpy(),
        "Z": obs_df[z_cols].to_numpy(),
        "A": obs_df["A"].to_numpy(),
        "Y": truth_df["T"].to_numpy(),
        "U": truth_df["U"].to_numpy(),
        "true_cate": truth_df["CATE_XU_eq7"].to_numpy(),
    }


def _make_cfg(case_spec):
    cfg_kwargs = dict(BASE_CONFIG)
    cfg_kwargs.update(case_spec["cfg"])
    return SynthConfig(**cfg_kwargs)


def _evaluate_case_variant(case, variant, seed):
    X = case["X"]
    W = case["W"]
    Z = case["Z"]
    U = case["U"]
    A = case["A"]
    Y = case["Y"]
    true_cate = case["true_cate"]
    cfg = case["cfg"]
    dgp = case["dgp"]
    x_core_dim = X.shape[1]

    if variant["kind"] == "baseline_x":
        return _evaluate_baseline(variant["name"], X, A, Y, X, true_cate, seed)

    if variant["kind"] == "baseline_xwz":
        X_full = MildShrinkNCCausalForestDML.stack_final_features(X, W, Z)
        return _evaluate_baseline(variant["name"], X_full, A, Y, X_full, true_cate, seed)

    if variant["kind"] == "oracle":
        X_final = MildShrinkNCCausalForestDML.stack_final_features(X, U)
        estimator = MildShrinkNCCausalForestDML(
            n_estimators=200,
            min_samples_leaf=20,
            cv=5,
            random_state=seed,
            x_core_dim=x_core_dim,
            oracle=True,
            use_true_q=variant["use_true_q"],
            use_true_h=variant["use_true_h"],
            q_true_fn=partial(oracle_q_from_proxy, dgp=dgp, cfg=cfg),
            h_true_fn=partial(oracle_h_from_proxy, cfg=cfg, dgp=dgp),
        )
        start = time.time()
        estimator.fit_oracle(X_final, A, Y, U)
        preds = estimator.effect(X_final).ravel()
        return _metric_row(variant["name"], preds, true_cate, time.time() - start)

    X_final = MildShrinkNCCausalForestDML.stack_final_features(X, W, Z)
    estimator = MildShrinkNCCausalForestDML(
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=seed,
        x_core_dim=x_core_dim,
        oracle=False,
        use_true_q=variant["use_true_q"],
        use_true_h=variant["use_true_h"],
        q_true_fn=partial(nc_q_from_proxy, dgp=dgp, cfg=cfg),
        h_true_fn=partial(nc_h_from_proxy, cfg=cfg, dgp=dgp),
    )
    start = time.time()
    estimator.fit_nc(X_final, A, Y, Z, W)
    preds = estimator.effect(X_final).ravel()
    return _metric_row(variant["name"], preds, true_cate, time.time() - start)


def run_case_benchmark(case_spec, output_dir: Path):
    cfg = _make_cfg(case_spec)
    case = _build_case(cfg, case_spec)

    rows = []
    for variant in VARIANT_SPECS:
        rows.append(_evaluate_case_variant(case, variant, seed=cfg.seed))

    case_df = pd.DataFrame(rows)
    case_df.insert(0, "case_id", case_spec["case_id"])
    case_df.insert(1, "case_slug", case_spec["slug"])

    case_csv = output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.csv"
    case_png = output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.png"
    case_df.to_csv(case_csv, index=False)
    _render_table_png(case_spec["title"], case_df.to_dict("records"), case_png, TABLE_COLUMNS, TABLE_KEYS)
    return case_df


def summarize_results(all_results: pd.DataFrame):
    summary = (
        all_results.groupby("name", as_index=False)
        .agg(
            avg_pred_cate=("mean_pred", "mean"),
            avg_true_cate=("mean_true_cate", "mean"),
            avg_acc=("sign_acc", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_pehe=("pehe", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
        )
        .sort_values(["avg_pehe", "avg_mae", "avg_bias"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def run_b2_vs_c3_12case_comparison(output_dir: Path, case_ids=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = CASE_SPECS
    if case_ids:
        case_id_set = set(case_ids)
        selected = [case_spec for case_spec in CASE_SPECS if case_spec["case_id"] in case_id_set]

    selected_variants = [
        next(v for v in VARIANT_SPECS if v["name"] == "B2  EconML baseline (X+W+Z)"),
        next(v for v in VARIANT_SPECS if v["name"] == "C3  NC-CSF (all estimated q/h)"),
    ]

    all_rows = []
    for case_spec in selected:
        cfg = _make_cfg(case_spec)
        case = _build_case(cfg, case_spec)
        rows = []
        for variant in selected_variants:
            rows.append(_evaluate_case_variant(case, variant, seed=cfg.seed))
        case_df = pd.DataFrame(rows)
        case_df.insert(0, "case_id", case_spec["case_id"])
        case_df.insert(1, "case_slug", case_spec["slug"])
        all_rows.append(case_df)

        case_csv = output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.csv"
        case_png = output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.png"
        case_df.to_csv(case_csv, index=False)
        _render_table_png(
            case_spec["title"],
            case_df.to_dict("records"),
            case_png,
            TABLE_COLUMNS,
            TABLE_KEYS,
            meta="B2 vs C3 comparison",
        )

    all_results = pd.concat(all_rows, ignore_index=True)
    summary = summarize_results(all_results)

    all_results.to_csv(output_dir / "all_12case_b2_vs_c3_results.csv", index=False)
    summary.to_csv(output_dir / "all_12case_b2_vs_c3_summary.csv", index=False)
    _render_table_png(
        "Non-censored 12-case B2 vs C3 comparison",
        summary.to_dict("records"),
        output_dir / "all_12case_b2_vs_c3_summary.png",
        SUMMARY_COLUMNS,
        SUMMARY_KEYS,
        dark=True,
        meta="Average summary",
        col_widths=[90, 460, 150, 150, 120, 140, 140, 120, 150, 120],
    )
    _render_b2_c3_plot(all_results, summary, output_dir / "b2_vs_c3_sqrt_pehe_comparison.png")

    return all_results, summary


def write_implementation_audit(output_dir: Path):
    audit = {
        "benchmark_definition": {
            "A1": "oracle X+U final forest, true q/h",
            "A2": "oracle X+U final forest, true q and estimated h; non-censored analogue of partial oracle nuisance",
            "A3": "oracle X+U final forest, all estimated q/h with econml mild shrink learner",
            "B1": "econml CausalForestDML baseline with X only",
            "B2": "econml CausalForestDML baseline with X+W+Z",
            "C1": "NC X+W+Z final forest, true q/h",
            "C2": "NC X+W+Z final forest, true q and estimated h; non-censored analogue of partial oracle nuisance",
            "C3": "NC X+W+Z final forest, all estimated q/h with single-model econml mild shrink",
        },
        "metrics": ["Pred CATE", "True CATE", "Bias", "RMSE", "PEHE", "MAE", "Pearson", "Sign accuracy", "Time"],
        "noncensored_note": "Because the non-censored benchmark has no survival nuisance terms, A2/C2 are implemented as true q + estimated h.",
        "num_cases": 12,
    }
    (output_dir / "implementation_audit.json").write_text(json.dumps(audit, indent=2))


def run_all_12case_benchmarks(output_dir: Path, case_ids=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = CASE_SPECS
    if case_ids:
        case_id_set = set(case_ids)
        selected = [case_spec for case_spec in CASE_SPECS if case_spec["case_id"] in case_id_set]

    all_rows = []
    for case_spec in selected:
        print("=" * 100)
        print(f"Running case {case_spec['case_id']:02d}")
        print(case_spec["title"])
        print("=" * 100)
        case_df = run_case_benchmark(case_spec, output_dir)
        all_rows.append(case_df)
        print(f"Saved {output_dir / f'case_{case_spec['case_id']:02d}_{case_spec['slug']}.csv'}")
        print(f"Saved {output_dir / f'case_{case_spec['case_id']:02d}_{case_spec['slug']}.png'}")

    all_results = pd.concat(all_rows, ignore_index=True)
    summary = summarize_results(all_results)
    top5 = summary.head(5).copy()

    all_results.to_csv(output_dir / "all_12case_8variant_results.csv", index=False)
    summary.to_csv(output_dir / "all_12case_8variant_avg_summary.csv", index=False)
    top5.to_csv(output_dir / "all_12case_8variant_top5.csv", index=False)

    _render_table_png(
        "Non-censored 12-case 8-variant benchmark average summary",
        summary.to_dict("records"),
        output_dir / "all_12case_8variant_avg_summary.png",
        SUMMARY_COLUMNS,
        SUMMARY_KEYS,
        dark=True,
        meta="12-case average summary",
        col_widths=[90, 460, 150, 150, 120, 140, 140, 120, 150, 120],
    )
    _render_table_png(
        "Non-censored 12-case 8-variant benchmark top 5",
        top5.to_dict("records"),
        output_dir / "all_12case_8variant_top5.png",
        TOP5_COLUMNS,
        TOP5_KEYS,
        dark=True,
        meta="Top 5 by avg sqrt-PEHE",
        col_widths=[90, 520, 140, 140, 140, 160, 120],
    )
    _render_b2_c3_plot(all_results, summary, output_dir / "b2_vs_c3_sqrt_pehe_comparison.png")
    write_implementation_audit(output_dir)
    return all_results, summary, top5
