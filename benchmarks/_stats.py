"""Shared statistics, vendored VERBATIM from the DUET pipeline (scripts/run_track_v2.py).

Copied rather than imported so this repository stands alone: a colleague cloning only nccsf
has no DUET checkout. Copied rather than REIMPLEMENTED so a baseline's E-value, control-arm
risk and BCa interval mean EXACTLY what they mean for the headline NC-CSF numbers.

Vendored: km_risk, e_value, evalue_multi, _norm_cdf, _norm_ppf, bca_ci.
Do not edit here; re-vendor from DUET if the source changes.
"""
from __future__ import annotations

import math

from typing import Any

import numpy as np


def km_risk(time: np.ndarray, event: np.ndarray, horizon: float) -> float:
    """Kaplan-Meier cumulative risk at `horizon` (censoring-aware baseline).

    Proper KM with TIE handling: at each distinct event time t, S *= (1 - d/at_risk)
    where d = events at t and at_risk = subjects with time >= t. Ties are collapsed
    so a lone late event does not spuriously drop S to 0, and the last censored
    subjects (event=0, which in the capped v2 endpoint sit exactly at the horizon)
    do not contribute a factor. Returns a value in [0, 1).
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if len(time) == 0:
        return float("nan")
    order = np.argsort(time, kind="mergesort")
    t, e = time[order], event[order]
    n = len(t)
    surv = 1.0
    i = 0
    while i < n:
        ti = t[i]
        if ti > horizon:
            break
        j = i
        d = 0  # events tied at ti
        while j < n and t[j] == ti:
            d += int(e[j] == 1)
            j += 1
        at_risk = n - i  # subjects with time >= ti
        if d > 0 and at_risk > 0:
            surv *= 1.0 - d / at_risk
        i = j
    # Guard: keep strictly < 1 so downstream ratios/E-values stay defined even if
    # a control arm is (near-)fully evented; a genuinely all-event baseline is
    # itself a red flag the caller can see via the reported risk0.
    return float(min(max(1.0 - surv, 0.0), 1.0 - 1e-9))


def e_value(rr: float) -> float:
    """VanderWeele-Ding E-value: E = RR + sqrt(RR * (RR - 1)); RR<1 is inverted.

    Reads as: an unmeasured confounder would need associations of at least this
    magnitude with BOTH the exposure and the outcome -- beyond the measured
    covariates -- to fully explain the observed association away.
    """
    rr = float(rr)
    if not np.isfinite(rr) or rr <= 0:
        return float("nan")
    r = rr if rr >= 1.0 else 1.0 / rr
    return float(r + math.sqrt(r * (r - 1.0)))


def evalue_multi(tau: float, ci_lo: float, ci_hi: float, risk0: float) -> dict[str, Any]:
    """Compute the E-value under SEVERAL pre-specified ratio definitions and pick,
    per track, the one giving the strongest (largest) conservative E-value.

    Methods (all on the same survival-probability contrast, so no cross-analysis
    borrowing -- only the ratio DEFINITION differs):
      * risk_ratio : RR   = risk1 / risk0                 (cumulative-risk ratio)
      * risk_diff  : RR*  = VanderWeele's sqrt-approx for a risk difference on a
                            binary outcome, RR* = (risk1/risk0) as above but with
                            the rare-outcome linearisation (kept explicit for the
                            report so reviewers see the alternative)
      * hazard_ratio: HR  = log(S1)/log(S0)               (proportional-hazards ratio)
      * odds_ratio : OR   = [risk1/(1-risk1)] / [risk0/(1-risk0)], then the
                            VanderWeele OR->E rule uses sqrt(OR)

    The selected method is the one whose E-value(CI) -- the number a claim is
    defended on -- is largest, but ALL methods are returned so the paper can show
    the range and name the chosen one.
    """
    risk0 = float(risk0)
    if not np.isfinite(risk0) or not (0.0 < risk0 < 1.0):
        return {"selected": None, "methods": {}}
    # DEGENERACY GUARD: the implied counterfactual risk risk1 = risk0 - tau MUST lie in
    # (0,1). If |tau| exceeds risk0 (or its complement), risk1 is negative/>1 -- physically
    # impossible -- which signals a DEGENERATE survival-probability estimate (e.g. the S(h)
    # estimand collapsed at a horizon beyond follow-up, or heavy-censoring IPCW blow-up).
    # Clipping risk1 to 1e-9 here and proceeding fabricates a gigantic spurious E-value
    # (observed 5.3e8 on clopidogrel+PPI, where survprob=+0.25 disagreed with RMST harm).
    # Refuse to emit an E-value and flag the degeneracy instead of reporting garbage.
    risk1_raw = risk0 - float(tau)
    if not (0.0 < risk1_raw < 1.0):
        return {"selected": None, "methods": {}, "degenerate": True,
                "reason": (f"implied risk1={risk1_raw:+.3f} out of (0,1): survprob tau={float(tau):+.3f} "
                           f"vs control risk0={risk0:.3f} -- degenerate S(h) estimate, E-value undefined")}
    risk1 = min(max(risk1_raw, 1e-9), 1.0 - 1e-9)
    s0, s1 = 1.0 - risk0, 1.0 - risk1
    spans_null = ci_lo <= 0 <= ci_hi
    null_side = ci_hi if tau < 0 else ci_lo
    # Same degeneracy check on the CI's null-side bound: if risk0-null_side is out of (0,1)
    # (a pathological inverted CI where the point sits outside its own interval on the harm
    # side), the CLIP below would revive the explosion on the CI-based E-value -- the number
    # a claim is actually defended on. Treat it like spanning-null (E_ci=1, the conservative
    # "no confounding required to reach null") instead of a fabricated huge e_value_ci.
    ci_degenerate = not (0.0 < (risk0 - float(null_side)) < 1.0)
    risk1_ci = min(max(risk0 - float(null_side), 1e-9), 1.0 - 1e-9)

    def _rr_methods(r1):
        s1_ = 1.0 - r1
        out = {"risk_ratio": r1 / risk0}
        if 0 < s0 < 1 and 0 < s1_ < 1:
            out["hazard_ratio"] = math.log(s1_) / math.log(s0)
        odds = (r1 / (1 - r1)) / (risk0 / (1 - risk0))
        out["odds_ratio_sqrt"] = math.sqrt(odds)  # VanderWeele OR rule pre-sqrt
        return out

    pt = _rr_methods(risk1)
    ci = _rr_methods(risk1_ci)
    methods = {}
    for name in pt:
        e_pt = e_value(pt[name])
        e_ci = 1.0 if (spans_null or ci_degenerate) else e_value(ci.get(name, pt[name]))
        methods[name] = {"rr": float(pt[name]), "e_value": float(e_pt),
                         "rr_ci_bound": float(ci.get(name, pt[name])), "e_value_ci": float(e_ci)}
    # pick the strongest conservative (CI) E-value; tie-break on point E-value
    selected = max(methods, key=lambda k: (methods[k]["e_value_ci"], methods[k]["e_value"]))
    return {"risk0": risk0, "selected": selected, **methods[selected],
            "selected_method": selected, "methods": methods}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation to the inverse normal CDF (sufficient here).
    p = min(max(p, 1e-9), 1 - 1e-9)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def bca_ci(reps: np.ndarray, point: float, alpha: float = 0.05) -> tuple[float, float]:
    """Bias-corrected & accelerated (BCa) bootstrap interval.

    BCa corrects for (i) median bias between the bootstrap distribution and the
    point estimate -- the z0 term -- and (ii) skew via a jackknife acceleration a.
    It converges faster and is more accurate than the naive percentile interval,
    which lets us use FEWER replicates, and it directly addresses the
    "point estimate falls outside its own percentile CI" pathology (that is exactly
    the median-bias z0 correction). Falls back to percentile if degrees are
    insufficient or the numbers degenerate.
    """
    reps = np.asarray(reps, dtype=float)
    n = len(reps)
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    pct = (float(np.percentile(reps, 100 * lo_q)), float(np.percentile(reps, 100 * hi_q)))
    if n < 20:
        return pct
    # z0: bias correction from the fraction of reps below the point estimate.
    prop = float(np.mean(reps < point))
    # Guard: if the point lies essentially OUTSIDE the bootstrap support, z0 is
    # extreme and the BCa quantile adjustment collapses (both bounds map to one
    # tail). That is itself the "point outside its interval" pathology -- BCa
    # cannot rescue it, so fall back to the percentile interval rather than emit a
    # degenerate one. (This is rare once the estimator/bootstrap are sound.)
    if prop <= 1.0 / n or prop >= 1.0 - 1.0 / n:
        return pct
    z0 = _norm_ppf(prop)
    # a: acceleration via jackknife on the REPLICATES (approximation to the
    #    statistic's jackknife when only the bootstrap draws are available).
    jack = np.array([np.mean(np.delete(reps, i)) for i in range(n)]) if n <= 400 else None
    if jack is None:  # cheap large-n jackknife of the mean
        s = reps.sum()
        jack = (s - reps) / (n - 1)
    jbar = jack.mean()
    denom = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a = float(np.sum((jbar - jack) ** 3) / denom) if denom > 0 else 0.0
    if not (np.isfinite(z0) and np.isfinite(a)):
        return pct

    def _adj(q):
        z = _norm_ppf(q)
        return _norm_cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    p_lo, p_hi = _adj(lo_q), _adj(hi_q)
    # Degenerate-adjustment guard: if the corrected quantiles collapse or invert,
    # the BCa transform blew up (extreme z0/a) -- fall back to percentile.
    if not (0.0 < p_lo < p_hi < 1.0) or (p_hi - p_lo) < 0.3 * (hi_q - lo_q):
        return pct
    return float(np.percentile(reps, 100 * p_lo)), float(np.percentile(reps, 100 * p_hi))
