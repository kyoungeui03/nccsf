"""External comparison baselines, behind one import.

    from benchmarks import run_all, run_suite, run_bart, list_methods

Everything reports the SAME estimand set, so the rows stack into one table with whatever
method you are evaluating:

    {RMST, survival.probability} x {ATE, ATT}

Sign convention on BOTH scales: **negative = harm**.
  survival.probability tau = S1(h|x) - S0(h|x)
  RMST tau                 = RMST1(x) - RMST0(x)
  ATE = mean over the cohort;  ATT = mean over the treated (A == 1)

Methods
-------
  RSF-T      random survival forest, one per arm (T-learner)
  RSF-S      random survival forest with A as a feature (S-learner)
  AIPW       doubly-robust IPCW risk at h        (survival.probability only)
  TMLE       targeted MLE of the IPCW risk at h  (survival.probability only)
  BART       discrete-time survival BART, S-learner; posterior interval, not bootstrap

AIPW and TMLE are binary-at-h risk estimators, so RMST is undefined for them -- by
construction, not a missing feature.

The causal survival forest baseline is deliberately NOT here: this repository already ships
it as censored/censored_baseline.py (the R-grf CSF). A second copy would be free to drift
from the one the papers use.

These share one uncertainty contract so the numbers are comparable rather than merely
adjacent: a BCa interval over an (A, event)-STRATIFIED bootstrap in which the CATE learners
refit on the resample and predict on the FIXED original cohort, a method-agnostic
control-arm Kaplan-Meier risk0, and an E-value with a degeneracy guard. Matching the
resampling protocol matters -- a plain i.i.d. resample would widen these intervals by
construction and flatter whatever they are being compared against.
"""
from __future__ import annotations

import pandas as pd

from ._stats import bca_ci, e_value, evalue_multi, km_risk
from .bart_runner import find_bart_python, run_bart
from .suite import ESTIMATORS, ESTIMANDS, SCALES, run_suite

__all__ = [
    "run_all", "run_suite", "run_bart", "list_methods", "find_bart_python",
    "SCALES", "ESTIMANDS", "ESTIMATORS",
    "km_risk", "bca_ci", "e_value", "evalue_multi",
]

SUITE_METHODS = tuple(ESTIMATORS)          # RSF-T, RSF-S, AIPW, TMLE
ALL_METHODS = SUITE_METHODS + ("BART",)


def list_methods() -> dict[str, str]:
    """Available baselines -> one-line description."""
    return {
        "RSF-T": "random survival forest T-learner (one forest per arm)",
        "RSF-S": "random survival forest S-learner (A as a feature)",
        "AIPW": "doubly-robust IPCW risk at h (survival.probability only)",
        "TMLE": "targeted MLE of IPCW risk at h (survival.probability only)",
        "BART": "discrete-time survival BART S-learner (posterior interval; needs pymc-bart)",
    }


def run_all(df, horizon, *, x_cols, w_cols=(), z_cols=(), methods=None, n_boot=100, seed=0,
            adjust="xwz", verbose=True, bart_python=None, bart_kwargs=None):
    """Run every requested baseline on one cohort and return a single stacked DataFrame.

    Parameters
    ----------
    df        cohort with columns A (0/1), time, event (0/1) plus the named covariates
    horizon   h at which S(h) and RMST(h) are evaluated
    x_cols    confounders;  w_cols  outcome proxies (NCO);  z_cols  treatment proxies (NCE)
    methods   subset of ``ALL_METHODS``; default is everything available
    n_boot    bootstrap replicates for the non-BART baselines (BART uses its posterior)

    BART is skipped with a printed reason -- not an exception -- when no pymc-bart
    interpreter is available, so an optional dependency cannot take down the comparison.
    """
    wanted = list(methods) if methods is not None else list(ALL_METHODS)
    unknown = [m for m in wanted if m not in ALL_METHODS]
    if unknown:
        raise ValueError(f"unknown method(s) {unknown}; available: {list(ALL_METHODS)}")

    frames = []
    suite_wanted = [m for m in wanted if m in SUITE_METHODS]
    if suite_wanted:
        frames.append(run_suite(df, horizon=horizon, x_cols=x_cols, w_cols=w_cols,
                                z_cols=z_cols, methods=suite_wanted, n_boot=n_boot,
                                seed=seed, adjust=adjust, verbose=verbose))
    if "BART" in wanted:
        frames.append(run_bart(df, horizon, x_cols=x_cols, w_cols=w_cols, z_cols=z_cols,
                               adjust=adjust, seed=seed, python=bart_python,
                               verbose=verbose, **(bart_kwargs or {})))
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.attrs["horizon"] = float(horizon)
    out.attrs["adjust"] = adjust
    return out
