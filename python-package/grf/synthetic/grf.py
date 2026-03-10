from __future__ import annotations

from math import erf, sqrt

import numpy as np


def _beta_2_4_density(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = (x >= 0.0) & (x <= 1.0)
    out[mask] = 20.0 * x[mask] * (1.0 - x[mask]) ** 3
    return out


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def _toeplitz_correlation(p: int, rho: float) -> np.ndarray:
    first_row = rho ** np.arange(p, dtype=float)
    return np.fromfunction(lambda i, j: first_row[np.abs(i - j).astype(int)], (p, p), dtype=int)


def _make_covariates(
    n: int,
    p: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if rho == 0:
        return rng.random((n, p))
    mean = np.zeros(p, dtype=float)
    cov = _toeplitz_correlation(p, rho)
    return _normal_cdf(rng.multivariate_normal(mean, cov, size=n))


def generate_causal_survival_data(
    n: int,
    p: int,
    Y_max: float | None = None,
    y0: float | None = None,
    X: np.ndarray | None = None,
    rho: float = 0.0,
    n_mc: int = 10000,
    dgp: str = "simple1",
    seed: int | None = None,
) -> dict[str, np.ndarray | float | str]:
    min_p = {
        "simple1": 1,
        "type1": 5,
        "type2": 5,
        "type3": 5,
        "type4": 5,
        "type5": 5,
    }
    if dgp not in min_p:
        raise ValueError(f"Unsupported dgp: {dgp}")

    rng = np.random.default_rng(seed)

    if X is not None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        n, p = X.shape

    if p < min_p[dgp]:
        raise ValueError(f"Selected dgp {dgp} requires a minimum of {min_p[dgp]} variables.")

    if dgp == "simple1":
        Y_max = 1.0 if Y_max is None else float(Y_max)
        y0 = 0.6 if y0 is None else float(y0)
        X = _make_covariates(n, p, rho, rng) if X is None else X
        W = rng.binomial(1, 0.5, size=n)
        failure_time = np.minimum(rng.exponential(size=n) * X[:, 0] + W, Y_max)
        censor_time = 2.0 * rng.random(n)
        Y = np.minimum(failure_time, censor_time)
        D = (failure_time <= censor_time).astype(int)

        temp = rng.exponential(size=n_mc)
        cate = np.empty(n, dtype=float)
        cate_prob = np.empty(n, dtype=float)
        for index in range(n):
            cate[index] = np.mean(np.minimum(temp * X[index, 0] + 1.0, Y_max) - np.minimum(temp * X[index, 0], Y_max))
            cate_prob[index] = np.mean(temp * X[index, 0] + 1.0 > y0) - np.mean(temp * X[index, 0] > y0)
        cate_sign = np.ones(n, dtype=float)

    elif dgp == "type1":
        Y_max = 1.5 if Y_max is None else float(Y_max)
        y0 = 0.8 if y0 is None else float(y0)
        X = _make_covariates(n, p, rho, rng) if X is None else X
        e = (1.0 + _beta_2_4_density(X[:, 0])) / 4.0
        W = rng.binomial(1, e)
        i1 = X[:, 0] < 0.5
        ft = np.exp(
            -1.85
            - 0.8 * i1
            + 0.7 * np.sqrt(X[:, 1])
            + 0.2 * X[:, 2]
            + (0.7 - 0.4 * i1 - 0.4 * np.sqrt(X[:, 1])) * W
            + rng.normal(size=n)
        )
        failure_time = np.minimum(ft, Y_max)
        numerator = -np.log(rng.random(n))
        denominator = np.exp(
            -1.75
            - 0.5 * np.sqrt(X[:, 1])
            + 0.2 * X[:, 2]
            + (1.15 + 0.5 * i1 - 0.3 * np.sqrt(X[:, 1])) * W
        )
        censor_time = np.sqrt(numerator / denominator)
        Y = np.minimum(failure_time, censor_time)
        D = (failure_time <= censor_time).astype(int)

        cate = np.empty(n, dtype=float)
        cate_prob = np.empty(n, dtype=float)
        eps = rng.normal(size=n_mc)
        for index in range(n):
            ft0 = np.exp(-1.85 - 0.8 * i1[index] + 0.7 * np.sqrt(X[index, 1]) + 0.2 * X[index, 2] + eps)
            ft1 = np.exp(
                -1.85
                - 0.8 * i1[index]
                + 0.7 * np.sqrt(X[index, 1])
                + 0.2 * X[index, 2]
                + 0.7
                - 0.4 * i1[index]
                - 0.4 * np.sqrt(X[index, 1])
                + eps
            )
            cate[index] = np.mean(np.minimum(ft1, Y_max) - np.minimum(ft0, Y_max))
            cate_prob[index] = np.mean(ft1 > y0) - np.mean(ft0 > y0)
        cate_sign = np.sign(0.7 - 0.4 * i1 - 0.4 * np.sqrt(X[:, 1]))

    elif dgp == "type2":
        Y_max = 2.0 if Y_max is None else float(Y_max)
        y0 = 1.2 if y0 is None else float(y0)
        X = _make_covariates(n, p, rho, rng) if X is None else X
        e = (1.0 + _beta_2_4_density(X[:, 0])) / 4.0
        W = rng.binomial(1, e)
        numerator = -np.log(rng.random(n))
        cox_ft = (numerator / np.exp(X[:, 0] + (-0.5 + X[:, 1]) * W)) ** 2
        failure_time = np.minimum(cox_ft, Y_max)
        censor_time = 3.0 * rng.random(n)
        Y = np.minimum(failure_time, censor_time)
        D = (failure_time <= censor_time).astype(int)

        cate = np.empty(n, dtype=float)
        cate_prob = np.empty(n, dtype=float)
        numerator_mc = -np.log(rng.random(n_mc))
        for index in range(n):
            cox_ft0 = (numerator_mc / np.exp(X[index, 0])) ** 2
            cox_ft1 = (numerator_mc / np.exp(X[index, 0] + (-0.5 + X[index, 1]))) ** 2
            cate[index] = np.mean(np.minimum(cox_ft1, Y_max) - np.minimum(cox_ft0, Y_max))
            cate_prob[index] = np.mean(cox_ft1 > y0) - np.mean(cox_ft0 > y0)
        cate_sign = -np.sign(-0.5 + X[:, 1])

    elif dgp == "type3":
        Y_max = 15.0 if Y_max is None else float(Y_max)
        y0 = 10.0 if y0 is None else float(y0)
        X = _make_covariates(n, p, rho, rng) if X is None else X
        e = (1.0 + _beta_2_4_density(X[:, 0])) / 4.0
        W = rng.binomial(1, e)
        lambda_failure = X[:, 1] ** 2 + X[:, 2] + 6.0 + 2.0 * (np.sqrt(X[:, 0]) - 0.3) * W
        failure_time = np.minimum(rng.poisson(lam=lambda_failure), Y_max)
        lambda_censor = 12.0 + np.log1p(np.exp(X[:, 2]))
        censor_time = rng.poisson(lam=lambda_censor)
        Y = np.minimum(failure_time, censor_time)
        D = (failure_time <= censor_time).astype(int)

        cate = np.empty(n, dtype=float)
        cate_prob = np.empty(n, dtype=float)
        lambda_failure_0 = X[:, 1] ** 2 + X[:, 2] + 6.0
        lambda_failure_1 = X[:, 1] ** 2 + X[:, 2] + 6.0 + 2.0 * (np.sqrt(X[:, 0]) - 0.3)
        for index in range(n):
            ft0 = rng.poisson(lam=lambda_failure_0[index], size=n_mc)
            ft1 = rng.poisson(lam=lambda_failure_1[index], size=n_mc)
            cate[index] = np.mean(np.minimum(ft1, Y_max) - np.minimum(ft0, Y_max))
            cate_prob[index] = np.mean(ft1 > y0) - np.mean(ft0 > y0)
        cate_sign = np.sign(np.sqrt(X[:, 0]) - 0.3)

    elif dgp == "type4":
        Y_max = 3.0 if Y_max is None else float(Y_max)
        y0 = 2.0 if y0 is None else float(y0)
        X = _make_covariates(n, p, rho, rng) if X is None else X
        e = 1.0 / ((1.0 + np.exp(-X[:, 0])) * (1.0 + np.exp(-X[:, 1])))
        W = rng.binomial(1, e)
        lambda_failure = X[:, 1] + X[:, 2] + np.maximum(0.0, X[:, 0] - 0.3) * W
        failure_time = np.minimum(rng.poisson(lam=lambda_failure), Y_max)
        lambda_censor = 1.0 + np.log1p(np.exp(X[:, 2]))
        censor_time = rng.poisson(lam=lambda_censor)
        Y = np.minimum(failure_time, censor_time)
        D = (failure_time <= censor_time).astype(int)

        cate = np.empty(n, dtype=float)
        cate_prob = np.empty(n, dtype=float)
        lambda_failure_0 = X[:, 1] + X[:, 2]
        lambda_failure_1 = X[:, 1] + X[:, 2] + np.maximum(0.0, X[:, 0] - 0.3)
        for index in range(n):
            ft0 = rng.poisson(lam=lambda_failure_0[index], size=n_mc)
            ft1 = rng.poisson(lam=lambda_failure_1[index], size=n_mc)
            cate[index] = np.mean(np.minimum(ft1, Y_max) - np.minimum(ft0, Y_max))
            cate_prob[index] = np.mean(ft1 > y0) - np.mean(ft0 > y0)
        cate_sign = np.sign(np.maximum(0.0, X[:, 0] - 0.3))
        cate_sign = cate_sign.astype(float)
        cate_sign[X[:, 0] < 0.3] = np.nan

    else:
        Y_max = 2.0 if Y_max is None else float(Y_max)
        y0 = 0.17 if y0 is None else float(y0)
        X = _make_covariates(n, p, rho, rng) if X is None else X
        e = (1.0 + _beta_2_4_density(X[:, 0])) / 4.0
        W = rng.binomial(1, e)
        numerator = -np.log(rng.random(n))
        cox_ft = (numerator / np.exp(X[:, 0] + (-0.4 + X[:, 1]) * W)) ** 2
        failure_time = np.minimum(cox_ft, Y_max)
        recycled_first_row = np.resize(X[0, :], n)
        censor_time = np.exp(recycled_first_row - X[:, 2] * W + rng.normal(size=n))
        Y = np.minimum(failure_time, censor_time)
        D = (failure_time <= censor_time).astype(int)

        cate = np.empty(n, dtype=float)
        cate_prob = np.empty(n, dtype=float)
        numerator_mc = -np.log(rng.random(n_mc))
        for index in range(n):
            cox_ft0 = (numerator_mc / np.exp(X[index, 0])) ** 2
            cox_ft1 = (numerator_mc / np.exp(X[index, 0] + (-0.4 + X[index, 1]))) ** 2
            cate[index] = np.mean(np.minimum(cox_ft1, Y_max) - np.minimum(cox_ft0, Y_max))
            cate_prob[index] = np.mean(cox_ft1 > y0) - np.mean(cox_ft0 > y0)
        cate_sign = -np.sign(-0.4 + X[:, 1])

    return {
        "X": X,
        "Y": Y,
        "W": W,
        "D": D,
        "cate": cate,
        "cate.prob": cate_prob,
        "cate.sign": cate_sign,
        "dgp": dgp,
        "Y.max": float(Y_max),
        "y0": float(y0),
    }
