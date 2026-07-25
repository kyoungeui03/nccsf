"""Doubly-robust survival estimators (AIPW / TMLE), vendored VERBATIM from the DUET
pipeline (scripts/validation/estimators.py), as one CONTIGUOUS block.

Contiguous on purpose: an earlier function-by-function copy silently dropped the
_Nuisances dataclass that sits between two of them, and suite._dr's except-Exception
turned that into four missing AIPW/TMLE rows rather than an error. Copying the whole
span cannot lose a helper.

Copied rather than imported so this repository stands alone; copied rather than
REIMPLEMENTED so the DR baselines target exactly the estimand the NC-CSF comparison
assumes. DUET-specific helpers (nccsf_ate and the pair/FeatureSpec registry) are
deliberately excluded -- they depend on the DUET package.

These are RISK-scale estimators of P(event by h); survival.probability = -risk. Being
binary-at-h estimators, they have no RMST.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.linear_model import LogisticRegression


def ipcw_binary(time: np.ndarray, event: np.ndarray, h: float, X=None, censoring: str = "km"):
    """Binary event-by-h outcome Y and IPCW weights w.

    Y_i = 1{ T_i <= h and event_i = 1 }.
    A subject is INFORMATIVE (w>0) iff their status at h is known:
      * event before h            -> Y=1, w = 1 / G(T_i^-)
      * still at risk at h (T_i>h) -> Y=0, w = 1 / G(h^-)
      * censored before h          -> status unknown -> w = 0 (dropped; their
                                       information already lives in the model of G)

    ``censoring`` selects the censoring survival G used for the weights:
      * "km"  -- marginal Kaplan-Meier of the CENSORING indicator (1-event). Simple, but
                 assumes censoring is independent of X.
      * "cox" -- CONDITIONAL Cox on X: G(t | X_i) = S0(t) ** exp(beta.X_i). Use this so the
                 AIPW/TMLE censoring MATCHES the conditional NC-CSF (which fits a Cox
                 conditional-censoring model). Otherwise the NC-CSF-vs-ignorability GAP would
                 conflate the confounding the proxies remove with a censoring-MODEL difference --
                 a real concern here because death is treated as (informative) censoring. Requires
                 X; if the Cox fit fails (separation/convergence) it falls back to KM with a warning.
    G is evaluated just-before t so an event at its own censoring time is not divided by a
    post-drop G, and floored at 1e-3 to avoid exploding weights.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    n = len(time)

    Gfn = None
    if censoring == "cox" and X is not None:
        try:
            from lifelines import CoxPHFitter
            Xa = np.asarray(X, dtype=float)
            dfc = pd.DataFrame(Xa, columns=[f"x{i}" for i in range(Xa.shape[1])])
            dfc["_t"] = time
            dfc["_c"] = (1 - event).astype(int)            # censoring is the "event" for G
            cph = CoxPHFitter(penalizer=0.1).fit(dfc, duration_col="_t", event_col="_c")
            s0 = cph.baseline_survival_.iloc[:, 0]
            t0 = s0.index.values.astype(float)
            s0v = s0.values.astype(float)
            hr = np.asarray(cph.predict_partial_hazard(dfc), dtype=float).ravel()   # exp(beta.X_i)

            def Gfn(ts, idx):    # conditional G(ts_i | X_i), just-before ts, floored
                base = np.interp(np.clip(np.asarray(ts, float) - 1e-9, 0.0, None), t0, s0v,
                                 left=1.0, right=s0v[-1])
                return np.clip(base ** hr[idx], 1e-3, 1.0)
        except Exception as e:   # noqa: BLE001 -- Cox failed -> marginal KM fallback
            warnings.warn(f"IPCW: conditional Cox censoring failed ({type(e).__name__}); "
                          f"falling back to marginal KM.", RuntimeWarning)
            Gfn = None

    if Gfn is None:   # marginal Kaplan-Meier censoring
        kmf = KaplanMeierFitter().fit(time, event_observed=(1 - event))
        # Heavy-censoring guard: if G(h^-) is tiny the at-risk-through-h weights are huge and the
        # 1e-3 floor UNDER-weights those (Y=0) survivors -> biases risk UPWARD (anti-conservative).
        raw_Gh = float(np.asarray(kmf.predict(max(float(h) - 1e-9, 0.0)), float))
        if raw_Gh < 5e-2:
            warnings.warn(
                f"IPCW: censoring survival G(h^-)={raw_Gh:.3g} at h={h:g} is small; "
                f"at-risk weights are large and the 1e-3 floor may bias risk upward. "
                f"Prefer a horizon inside lighter-censoring follow-up.", RuntimeWarning)

        def Gfn(ts, idx=None):   # marginal (idx ignored)
            g = np.asarray(kmf.predict(np.clip(np.asarray(ts, float) - 1e-9, 0.0, None)), float)
            return np.clip(g, 1e-3, 1.0)

    Y = np.zeros(n)
    w = np.zeros(n)
    ev_before_h = (time <= h) & (event == 1)   # Y=1: event observed at or before h.
    # Y=0: event-free THROUGH h (time >= h so a subject admin-censored *at* h is counted event-free).
    reached_h = (time >= h) & ~ev_before_h
    Y[ev_before_h] = 1.0
    idx_ev = np.where(ev_before_h)[0]
    w[ev_before_h] = 1.0 / Gfn(time[ev_before_h], idx_ev)
    idx_rh = np.where(reached_h)[0]
    w[reached_h] = 1.0 / Gfn(np.full(len(idx_rh), float(h)), idx_rh)
    # only (time < h and event==0) = truly censored before h -> w stays 0
    return Y, w


# ---------------------------------------------------------------------------
# Shared doubly-robust nuisances (propensity + outcome regressions), IPCW-weighted
# ---------------------------------------------------------------------------
@dataclass
class _Nuisances:
    e: np.ndarray       # propensity P(A=1|X), clipped
    mu0: np.ndarray     # E[Y|A=0,X]
    mu1: np.ndarray     # E[Y|A=1,X]


def _fit_nuisances(X, A, Y, w, prop_clip=0.02):
    """IPCW-weighted logistic nuisances on the informative subset (w>0)."""
    m = w > 0
    Xm, Am, Ym, wm = X[m], A[m], Y[m], w[m]
    # propensity
    pe = LogisticRegression(max_iter=1000, C=1.0)
    pe.fit(Xm, Am, sample_weight=wm)
    e = np.clip(pe.predict_proba(X)[:, 1], prop_clip, 1 - prop_clip)
    # outcome models per arm (weighted). Guard degenerate (all-0/all-1) outcomes.
    def _mu(a):
        ma = m & (A == a)
        if ma.sum() < 5 or len(np.unique(Y[ma])) < 2:
            fill = float(np.average(Y[ma], weights=w[ma])) if ma.sum() else 0.0
            return np.full(len(X), fill)
        om = LogisticRegression(max_iter=1000, C=1.0)
        om.fit(X[ma], Y[ma], sample_weight=w[ma])
        return om.predict_proba(X)[:, 1]
    return _Nuisances(e=e, mu0=_mu(0), mu1=_mu(1))


def _wmean_se(score, w):
    """IPCW-weighted mean of an influence-function-style score + its SE.

    SE uses the linearized (sandwich) variance of a weighted mean,
    Var = sum(w_i^2 (s_i - ate)^2) / (sum w_i)^2, i.e. the weighted sample variance
    divided by the EFFECTIVE sample size n_eff = (sum w)^2 / sum(w^2) -- NOT the raw
    count n. Dividing by n understates the SE (narrows the CI) under the unequal
    IPCW weights this harness is built to use. Still an approximation: it does not
    propagate estimation of the censoring KM G, so the real run uses NC-CSF's
    bootstrap CI for primary inference.
    """
    m = w > 0
    s, ww = score[m], w[m]
    ate = np.average(s, weights=ww)
    sw = ww.sum()
    var = float(np.sum((ww ** 2) * (s - ate) ** 2) / (sw ** 2)) if sw > 0 else 0.0
    se = math.sqrt(max(var, 0.0))
    return float(ate), float(se)


# ---------------------------------------------------------------------------
# AIPW (doubly robust), risk scale, IPCW-adjusted
# ---------------------------------------------------------------------------
def aipw_risk_ate(X, A, time, event, h, censoring="cox"):
    # censoring="cox" (default) matches the conditional NC-CSF, so the gap isolates confounding
    Y, w = ipcw_binary(time, event, h, X=X, censoring=censoring)
    nu = _fit_nuisances(X, A, Y, w)
    e, mu0, mu1 = nu.e, nu.mu0, nu.mu1
    score = (mu1 - mu0) + A / e * (Y - mu1) - (1 - A) / (1 - e) * (Y - mu0)
    ate, se = _wmean_se(score, w)
    return {"risk_ate": ate, "se": se, "ci": (ate - 1.96 * se, ate + 1.96 * se),
            "p_treated": _wmean_se(mu1, w)[0], "p_control": _wmean_se(mu0, w)[0]}


# ---------------------------------------------------------------------------
# TMLE (targeted MLE), risk scale, IPCW-adjusted -- shares the same nuisances
# ---------------------------------------------------------------------------
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def tmle_risk_ate(X, A, time, event, h, censoring="cox"):
    import statsmodels.api as sm
    Y, w = ipcw_binary(time, event, h, X=X, censoring=censoring)   # cox = match NC-CSF (clean gap)
    nu = _fit_nuisances(X, A, Y, w)
    e, mu0, mu1 = nu.e, nu.mu0, nu.mu1
    m = w > 0
    muA = A * mu1 + (1 - A) * mu0
    H = A / e - (1 - A) / (1 - e)                 # clever covariate for the ATE
    offset = _logit(muA)
    # targeting: weighted logistic Y ~ H (no intercept) with offset logit(muA)
    glm = sm.GLM(Y[m], H[m][:, None], family=sm.families.Binomial(),
                 offset=offset[m], freq_weights=w[m])
    eps = float(glm.fit().params[0])
    q1 = 1 / (1 + np.exp(-(_logit(mu1) + eps * (1 / e))))
    q0 = 1 / (1 + np.exp(-(_logit(mu0) + eps * (-1 / (1 - e)))))
    ate, _ = _wmean_se(q1 - q0, w)
    # influence-function SE for TMLE
    ic = H * (Y - (A * q1 + (1 - A) * q0)) + (q1 - q0) - ate
    _, se = _wmean_se(ic, w)
    return {"risk_ate": float(ate), "se": float(se),
            "ci": (ate - 1.96 * se, ate + 1.96 * se),
            "p_treated": _wmean_se(q1, w)[0], "p_control": _wmean_se(q0, w)[0]}


# ---------------------------------------------------------------------------
# ATT versions (effect of the co-med on the TREATED, risk scale, IPCW-adjusted) --
# so the ignorability gap can be reported on the ATT scale too (apples-to-apples with
# NC-CSF's ATT). Both are doubly robust and share the same nuisances as the ATE versions.
# ---------------------------------------------------------------------------
def aipw_att_risk(X, A, time, event, h, censoring="cox"):
    """AIPW ATT on the risk scale. tau_ATT = (1/pi) E_w[ A(Y-mu0) - (1-A)(e/(1-e))(Y-mu0) ];
    DR (consistent if mu0 OR e is correct) and needs only the control-outcome model + propensity."""
    Y, w = ipcw_binary(time, event, h, X=X, censoring=censoring)   # cox = match NC-CSF (clean gap)
    nu = _fit_nuisances(X, A, Y, w)
    e, mu0 = nu.e, nu.mu0
    m = w > 0
    pi = float(np.average(A[m], weights=w[m])) if m.any() else 0.0
    if not (pi > 0):
        return {"risk_ate": float("nan"), "se": float("nan"), "ci": (float("nan"), float("nan"))}
    score = (A * (Y - mu0) - (1 - A) * (e / (1 - e)) * (Y - mu0)) / pi
    ate = float(np.average(score[m], weights=w[m]))
    ic = score - (A / pi) * ate                       # ATT influence function
    _, se = _wmean_se(ic, w)
    return {"risk_ate": ate, "se": float(se), "ci": (ate - 1.96 * se, ate + 1.96 * se)}


def tmle_att_risk(X, A, time, event, h, censoring="cox"):
    """TMLE ATT on the risk scale, IPCW-adjusted. Clever covariate H=(A-(1-A)e/(1-e))/pi;
    fluctuate Q, then average Q1*-Q0* over the treated."""
    import statsmodels.api as sm
    Y, w = ipcw_binary(time, event, h, X=X, censoring=censoring)   # cox = match NC-CSF (clean gap)
    nu = _fit_nuisances(X, A, Y, w)
    e, mu0, mu1 = nu.e, nu.mu0, nu.mu1
    m = w > 0
    pi = float(np.average(A[m], weights=w[m])) if m.any() else 0.0
    treated = m & (A == 1)
    if not (pi > 0) or not treated.any():
        return {"risk_ate": float("nan"), "se": float("nan"), "ci": (float("nan"), float("nan"))}
    muA = A * mu1 + (1 - A) * mu0
    H = (A - (1 - A) * (e / (1 - e))) / pi
    offset = _logit(muA)
    glm = sm.GLM(Y[m], H[m][:, None], family=sm.families.Binomial(),
                 offset=offset[m], freq_weights=w[m])
    eps = float(glm.fit().params[0])
    q1 = 1 / (1 + np.exp(-(_logit(mu1) + eps * (1.0 / pi))))
    q0 = 1 / (1 + np.exp(-(_logit(mu0) + eps * (-(e / (1 - e)) / pi))))
    att = float(np.average((q1 - q0)[treated], weights=w[treated]))
    ic = H * (Y - (A * q1 + (1 - A) * q0)) + (A / pi) * ((q1 - q0) - att)   # efficient ATT IC
    _, se = _wmean_se(ic, w)
    return {"risk_ate": att, "se": float(se), "ci": (att - 1.96 * se, att + 1.96 * se)}


# ---------------------------------------------------------------------------
# NC-CSF via the DUET wrapper (proximal, allows U) -- risk scale + RMST
# ---------------------------------------------------------------------------
def assert_horizon_inside_followup(time, h, min_beyond_frac=0.02):
    """Guard against evaluating the survival-probability estimand at/beyond the
    administrative-censoring boundary.

    grf's ``survival.probability`` CATE at horizon h collapses to ~0 when h is at
    or past the last observable time: the treated and control survival curves are
    both pinned to the KM tail there, so their difference vanishes -- a SILENT
    false-null. (RMST is integrated over [0,h] and does NOT share this failure, so
    the two estimands disagreeing at the boundary is itself the tell.) We require a
    minimum fraction of the cohort to still have follow-up strictly beyond h.
    """
    time = np.asarray(time, float)
    frac_beyond = float((time > h).mean())
    if frac_beyond < min_beyond_frac:
        raise ValueError(
            f"horizon h={h:g} is at/beyond the administrative-censoring boundary: "
            f"only {frac_beyond:.1%} of subjects have follow-up past h "
            f"(max observed time={time.max():g}). The survival.probability estimand "
            f"is degenerate here (would return a silent ~0). Choose h strictly inside "
            f"follow-up, e.g. h <= the {int((1-min_beyond_frac)*100)}th pctile of "
            f"observed times ({np.quantile(time, 1-min_beyond_frac):.0f})."
        )
    return frac_beyond
