"""External baseline suite: RSF T-learner, RSF S-learner, AIPW, TMLE.

The causal-survival-forest baseline is NOT here: this repository already ships it as
censored/censored_baseline.py (the R-grf CSF), and duplicating it would create a second
implementation free to drift from the one the papers use.

Every baseline emits the SAME estimand set as NC-CSF, so the rows drop straight into one
comparison table:

    {RMST, survival.probability} x {ATE, ATT}, each with a BCa bootstrap 95% CI,
    plus the E-value on the survival-probability scale.

  survival.probability tau = S1(h|x) - S0(h|x)      (harm < 0)   [primary scale]
  RMST tau                 = RMST1(x) - RMST0(x)    (harm < 0)   [curve methods]
  risk scale               = -(survprob tau)        (harm > 0)
  ATE = mean over the cohort;  ATT = mean over the TREATED (A == 1)

Design decisions that keep it apples-to-apples with NC-CSF (no logical drift):
  * risk0 = Kaplan-Meier CONTROL-ARM risk at h -- method-agnostic, and the same quantity the
    NC-CSF headline uses. NOT each estimator's own fitted S0. ATE and ATT share risk0 because
    the treated group's counterfactual baseline is that same control arm.
  * CI = the SAME BCa interval NC-CSF uses, over bootstrap replicates. The resample is
    STRATIFIED on the (A, event) 2x2 and CATE learners predict on the FIXED original cohort,
    matching the NC-CSF engine exactly -- a plain i.i.d. resample would inflate the baselines'
    intervals by construction and flatter NC-CSF.
  * E-value via the SAME evalue_multi, including the degeneracy guard that refuses to emit an
    E-value when the implied risk1 leaves (0, 1).
  * Adjustment defaults to the FULL observed stack [X, W, Z] as ORDINARY covariates -- the
    fairest ignorability baseline, since the proxies are legitimate pre-treatment covariates.
    A method-vs-baseline gap therefore isolates the PROXIMAL USE of the proxies, not mere
    access to them. Use adjust="x" for the older X-only convention.

BART lives in bart.py (separate environment) and is merged separately.

Usage:
    from benchmarks import run_suite
    res = run_suite(df, horizon=90, x_cols=[...], w_cols=[...], z_cols=[...], n_boot=100)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest

from ._dr import aipw_att_risk, aipw_risk_ate, tmle_att_risk, tmle_risk_ate
from ._stats import bca_ci, evalue_multi, km_risk

SCALES = ("survival.probability", "RMST")
ESTIMANDS = ("ATE", "ATT")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _num(df: pd.DataFrame, cols) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0))
    M = df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    M = np.where(np.isfinite(M), M, np.nan)
    for j in range(M.shape[1]):
        col = M[:, j]
        if np.isnan(col).any():
            med = np.nanmedian(col)
            col[np.isnan(col)] = med if np.isfinite(med) else 0.0
            M[:, j] = col
    return M


def _y_struct(time, event):
    return np.array([(bool(e), float(t)) for e, t in zip(np.asarray(event), np.asarray(time))],
                    dtype=[("event", "?"), ("time", "<f8")])


def _curve_arr(model, F, h: float):
    """VECTORIZED per-unit (S(h), RMST(h)) from an RSF's survival-probability array.

    predict_survival_function(return_array=True) -> (n_units x T) survival on
    model.unique_times_. S is a right-continuous step function (S=1 before the first
    grid time). Both quantities are computed with numpy over the whole cohort at once
    (no per-unit Python loop) so the bootstrap stays tractable at n=6000.
    """
    arr = np.asarray(model.predict_survival_function(F, return_array=True), float)  # (n, T)
    times = np.asarray(model.unique_times_, float)                                  # ascending
    n = arr.shape[0]
    jh = int(np.searchsorted(times, h, side="right") - 1)
    Sh = arr[:, jh] if jh >= 0 else np.ones(n)          # S(h): last grid time <= h
    grid = np.unique(np.clip(np.concatenate(([0.0], times[times <= h], [h])), 0.0, h))
    rmst = np.zeros(n)
    for a in range(len(grid) - 1):
        t0, t1 = grid[a], grid[a + 1]
        s = np.ones(n) if t0 < times[0] else arr[:, int(np.searchsorted(times, t0, side="right") - 1)]
        rmst += s * (t1 - t0)                            # exact step-integral of S over [0,h]
    return Sh, rmst


def _reduce(sp_unit, rm_unit, A):
    """Per-unit contrasts -> {scale: {ATE, ATT}}. ATT averages over the TREATED only."""
    tr = np.asarray(A, float) == 1
    out = {s: {} for s in SCALES}
    for scale, unit in (("survival.probability", sp_unit), ("RMST", rm_unit)):
        if unit is None:
            out[scale] = {"ATE": None, "ATT": None}
            continue
        unit = np.asarray(unit, float)
        out[scale]["ATE"] = float(np.mean(unit))
        out[scale]["ATT"] = float(np.mean(unit[tr])) if tr.any() else None
    return out


def _feat(df, cols, adjust):
    """Adjustment matrix: full observed stack [X,W,Z] (default) or X-only."""
    X = _num(df, cols["x"])
    if adjust == "x":
        return X
    W = _num(df, cols["w"]); Z = _num(df, cols["z"])
    blocks = [m for m in (X, W, Z) if m.shape[1] > 0]
    return np.hstack(blocks) if blocks else X


# ---------------------------------------------------------------------------
# estimators -- each returns {scale: {ATE, ATT}} on ONE cohort
# ---------------------------------------------------------------------------

def _rsf(feat, y, seed, max_features="sqrt", n_estimators=500):
    # 500 trees to match the NC-CSF engine's forest size; 200 gave the baselines strictly more
    # Monte-Carlo noise than our method.
    return RandomSurvivalForest(n_estimators=n_estimators, min_samples_leaf=15,
                                max_features=max_features,
                                random_state=seed, n_jobs=1).fit(feat, y)


def est_rsf_t(df, cols, h, adjust, seed=0, predict_df=None):
    """T-learner: a separate RSF per arm; predict BOTH counterfactuals for everyone."""
    pdf = df if predict_df is None else predict_df
    F = _feat(df, cols, adjust); A = df["A"].to_numpy(float)
    Fp = _feat(pdf, cols, adjust); Ap = pdf["A"].to_numpy(float)
    y = _y_struct(df["time"], df["event"])
    tr, ct = A == 1, A == 0
    if tr.sum() < 20 or ct.sum() < 20:
        return _reduce(None, None, Ap)
    m1 = _rsf(F[tr], y[tr], seed); m0 = _rsf(F[ct], y[ct], seed + 1)
    S1, R1 = _curve_arr(m1, Fp, h)     # each arm's model on ITS OWN time grid
    S0, R0 = _curve_arr(m0, Fp, h)
    return _reduce(S1 - S0, R1 - R0, Ap)


def est_rsf_s(df, cols, h, adjust, seed=0, predict_df=None):
    """S-learner: one RSF on [features, A]; predict with A forced to 1 and to 0.

    max_features="sqrt" leaves the single treatment column rarely eligible at a split among
    ~60 covariates, so the forest can largely ignore A -- the classic S-learner
    regularisation-induced confounding. Measured against a known truth this recovered only
    ~63% of the effect, about half of that attributable to the sqrt restriction. We therefore
    give the S-learner the whole feature set at every split so its attenuation reflects the
    S-learner PARAMETERISATION rather than a tuning choice we imposed on it.
    """
    pdf = df if predict_df is None else predict_df
    F = _feat(df, cols, adjust); A = df["A"].to_numpy(float)
    Fp = _feat(pdf, cols, adjust); Ap = pdf["A"].to_numpy(float)
    y = _y_struct(df["time"], df["event"])
    m = _rsf(np.hstack([F, A.reshape(-1, 1)]), y, seed, max_features=None)
    F1 = np.hstack([Fp, np.ones((len(pdf), 1))]); F0 = np.hstack([Fp, np.zeros((len(pdf), 1))])
    S1, R1 = _curve_arr(m, F1, h)
    S0, R0 = _curve_arr(m, F0, h)
    return _reduce(S1 - S0, R1 - R0, Ap)


DR_FAILURES: list[str] = []       # module-level so failures are visible, never silent


def _standardize(F: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance columns, dropping zero-variance ones.

    The doubly-robust nuisances are penalised LOGISTIC regressions (and a penalised Cox for
    the censoring model). On the raw [X, W, Z] stack -- which mixes 0/1 flags with lab values
    and log-transformed labs spanning orders of magnitude -- lbfgs does NOT converge in
    max_iter, and the resulting AIPW/TMLE estimates are unstable enough to FLIP SIGN once the
    design is standardised. Since penalised logistic regression is NOT equivariant under
    rescaling (the L2 penalty is applied on the raw coefficient scale), leaving this
    unstandardised silently mis-specifies the baselines -- which is exactly the direction that
    would flatter our method. Standardising is therefore a correctness fix, not a cosmetic one.
    Forests are scale-invariant, so this is applied ONLY on the DR path and cannot change
    CSF/RSF/BART.
    """
    F = np.asarray(F, float)
    if F.shape[1] == 0:
        return F
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    keep = sd > 1e-12
    if not keep.any():
        return np.zeros((F.shape[0], 0))
    return (F[:, keep] - mu[keep]) / sd[keep]


def _dr(df, cols, h, adjust, ate_fn, att_fn, tag=""):
    """AIPW/TMLE share the plumbing: risk-scale estimators, so survprob = -risk.

    Failures are RECORDED, not swallowed: a silently-missing baseline row shrinks the
    denominator of the downstream 'majority of baselines' vote, which biases the comparison
    toward our method.
    """
    F = _standardize(_feat(df, cols, adjust))
    A = df["A"].to_numpy(float); t = df["time"].to_numpy(float); e = df["event"].to_numpy(float)
    out = {"survival.probability": {}, "RMST": {"ATE": None, "ATT": None}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for est, fn in (("ATE", ate_fn), ("ATT", att_fn)):
            try:
                v = -float(fn(F, A, t, e, float(h), censoring="cox")["risk_ate"])
                # Saturation guard. A survival-probability contrast must lie in [-1, 1], but the
                # informative bound is STRICTLY inside it: the TMLE targeting step saturates on a
                # thin treated arm and emits values pinned at EXACTLY +/-1.0. Allowing |v| <= 1.0
                # let those through (observed bs_hi = 1.000000, dragging the bootstrap mean from
                # +0.086 to +0.211 and pushing ci_hi to exactly 1.0). A genuine contrast of 0.99
                # is not credible on these cohorts either, so reject at the boundary.
                out["survival.probability"][est] = v if (np.isfinite(v) and abs(v) < 0.99) else None
                if out["survival.probability"][est] is None:
                    DR_FAILURES.append(f"{tag}:{est}:saturated({v:+.3f})")
            except Exception as ex:
                out["survival.probability"][est] = None
                DR_FAILURES.append(f"{tag}:{est}:{type(ex).__name__}")
    return out


def est_aipw(df, cols, h, adjust, seed=0):
    return _dr(df, cols, h, adjust, aipw_risk_ate, aipw_att_risk, tag="AIPW")


def est_tmle(df, cols, h, adjust, seed=0):
    return _dr(df, cols, h, adjust, tmle_risk_ate, tmle_att_risk, tag="TMLE")


ESTIMATORS = {"RSF-T": est_rsf_t, "RSF-S": est_rsf_s,
              "AIPW": est_aipw, "TMLE": est_tmle}


# ---------------------------------------------------------------------------
# bootstrap + row assembly (same BCa + evalue_multi as the headline)
# ---------------------------------------------------------------------------
def _stratified_resample(A, event, rng):
    """Stratified bootstrap over the (A, event) 2x2, holding all four cell sizes FIXED.

    This is a byte-for-byte match of the NC-CSF engine's _stratified_resample_indices_np.
    It matters: plain i.i.d. row resampling lets the treated count and the event count
    fluctuate across replicates, adding a variance component that NC-CSF's design removes
    by construction. Leaving the two protocols different means every excludes_zero and
    e_value_ci comparison between NC-CSF and a baseline compares two different variance
    estimators rather than two methods -- and it inflates the BASELINES' intervals, which
    flatters us.
    """
    A = np.asarray(A); event = np.asarray(event)
    parts = []
    for a_val in (0, 1):
        for e_val in (0, 1):
            cell = np.where((A == a_val) & (event == e_val))[0]
            if cell.size:
                parts.append(rng.choice(cell, size=cell.size, replace=True))
    if not parts:
        return rng.integers(0, len(A), size=len(A))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


# Estimators that emit a PER-UNIT contrast can follow NC-CSF exactly: refit on the resample,
# then PREDICT AND AVERAGE ON THE FIXED ORIGINAL COHORT. The DR estimators are population-level
# and have no per-unit predict step, so they can only adopt the stratified resample.
CATE_LEARNERS = {"RSF-T", "RSF-S"}


def _bootstrap(fn, df, cols, h, adjust, n_boot, seed, fixed_predict=False):
    rng = np.random.default_rng(seed)
    A0 = df["A"].to_numpy(float); E0 = df["event"].to_numpy(float)
    acc = {(s, e): [] for s in SCALES for e in ESTIMANDS}
    for b in range(n_boot):
        idx = _stratified_resample(A0, E0, rng)
        bdf = df.iloc[idx].reset_index(drop=True)
        try:
            if fixed_predict:
                # fit on the resample, evaluate on the ORIGINAL cohort (NC-CSF's protocol)
                r = fn(bdf, cols, h, adjust, seed=seed + 1 + b, predict_df=df)
            else:
                r = fn(bdf, cols, h, adjust, seed=seed + 1 + b)
        except Exception:
            continue                     # a degenerate resample simply does not contribute
        for s in SCALES:
            for e in ESTIMANDS:
                v = (r.get(s) or {}).get(e)
                if v is None or not np.isfinite(v):
                    continue
                # Same saturation bound inside the bootstrap: a single pinned replicate
                # otherwise dominates both the bootstrap mean and the BCa endpoints.
                if s == "survival.probability" and abs(v) >= 0.99:
                    continue
                acc[(s, e)].append(v)
    return {k: np.asarray(v, float) for k, v in acc.items()}


def _row(method, scale, estimand, reps, point, risk0, n, n_ev, n_treated, n_boot=100, ci_mode="bca"):
    reps = np.asarray(reps, float)
    # Replicate-count floor MATCHING the NC-CSF convention (estimators.nccsf_ate requires
    # max(20, 0.5*n_boot) successful replicates). The old floor of 5 let a row built from six
    # replicates emit a full CI and E-value -- sometimes with a bootstrap mean of the OPPOSITE
    # sign to the point estimate.
    min_reps = max(20, int(0.5 * int(n_boot)))
    if point is None or not np.isfinite(point) or reps.size < min_reps:
        return None
    bs_mean = float(reps.mean())
    if ci_mode == "percentile":
        ci_lo, ci_hi = float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
    else:
        try:
            ci_lo, ci_hi = bca_ci(reps, float(point), 0.05)
        except Exception:
            ci_lo, ci_hi = float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
    # "excludes zero" must also mean the interval sits on the SAME SIDE as the estimate we
    # report. A BCa interval can land entirely on the opposite side of a bias-corrected point,
    # which would otherwise be read as a significant effect in the reported direction.
    excl = bool((ci_lo > 0 and bs_mean > 0) or (ci_hi < 0 and bs_mean < 0))
    row = {"method": method, "scale": scale, "estimand": estimand,
           "n": int(n), "n_ev": int(n_ev), "n_treated": int(n_treated),
           "point": float(point), "bs_mean": bs_mean,
           "bs_lo": float(np.min(reps)), "bs_hi": float(np.max(reps)),   # bs_range
           "bs_range": f"[{float(np.min(reps)):+.4f}, {float(np.max(reps)):+.4f}]",
           "ci_lo": ci_lo, "ci_hi": ci_hi,
           "n_reps": int(reps.size),
           "point_vs_bsmean": float(abs(float(point) - bs_mean)),
           "excludes_zero": excl}
    # E-value only on the survival-probability scale (the ratio is defined there).
    if scale == "survival.probability" and risk0 is not None and np.isfinite(risk0):
        ev = evalue_multi(bs_mean, ci_lo, ci_hi, risk0)
        if ev.get("selected") is not None:
            row.update({"risk0": ev["risk0"], "rr": ev["rr"], "e_value": ev["e_value"],
                        "e_value_ci": ev["e_value_ci"], "evalue_method": ev["selected"]})
        elif ev.get("degenerate"):
            row.update({"risk0": float(risk0), "e_value": float("nan"),
                        "degenerate_survprob": True, "degenerate_reason": ev.get("reason", "")})
    return row



def _resolve_cols(cfg, x_cols, w_cols, z_cols):
    """Column spec from explicit lists, else from a cfg exposing .x_cols/.w_cols/.z_cols.

    Explicit lists win. Refusing to silently default to "no covariates" matters: an empty
    adjustment set turns every baseline into an unadjusted comparison, which would look like
    a modelling result rather than a missing argument.
    """
    if x_cols is not None or w_cols is not None or z_cols is not None:
        return {"x": list(x_cols or []), "w": list(w_cols or []), "z": list(z_cols or [])}
    if cfg is None:
        raise ValueError("pass x_cols/w_cols/z_cols, or a cfg exposing .x_cols/.w_cols/.z_cols")
    missing = [a for a in ("x_cols", "w_cols", "z_cols") if not hasattr(cfg, a)]
    if missing:
        raise ValueError(f"cfg is missing {missing}; pass x_cols/w_cols/z_cols explicitly")
    return {"x": list(cfg.x_cols), "w": list(cfg.w_cols), "z": list(cfg.z_cols)}


def run_suite(df, cfg=None, horizon=None, methods=None, n_boot=100, seed=0, adjust="xwz",
              verbose=True, *, x_cols=None, w_cols=None, z_cols=None):
    """Run the baseline suite on one cohort.

    Columns come either from explicit ``x_cols``/``w_cols``/``z_cols`` (the plain way) or
    from any ``cfg`` object exposing ``.x_cols``/``.w_cols``/``.z_cols`` -- which keeps the
    DUET TrackConfig call-site working without importing DUET here.

    ``df`` needs columns A (0/1 treatment), time, event (0/1), plus the covariates named.
    Returns a DataFrame with one row per (method, scale, estimand).
    """
    if horizon is None:
        raise ValueError("horizon is required")
    cols = _resolve_cols(cfg, x_cols, w_cols, z_cols)
    A = df["A"].to_numpy(float); t = df["time"].to_numpy(float); e = df["event"].to_numpy(float)
    ctrl = A == 0
    # HORIZON GUARD: the IPCW risk-scale estimators (AIPW/TMLE) collapse when h sits at
    # the administrative-censoring boundary -- G(h) -> 0 makes the informative set tiny and
    # event-dominated, biasing risk sharply UPWARD (observed AIPW +0.54 at a boundary h vs
    # +0.06 well inside follow-up). Require a healthy risk set past h; warn loudly otherwise.
    from lifelines import KaplanMeierFitter
    frac_beyond = float((t > float(horizon)).mean())
    Gh = float(KaplanMeierFitter().fit(t, event_observed=(1 - e)).predict(float(horizon) - 1e-9))
    if frac_beyond < 0.05 or Gh < 0.05:
        print(f"  ** HORIZON WARNING: only {frac_beyond:.1%} of subjects have follow-up past "
              f"h={horizon:g} and censoring G(h)={Gh:.3f}. IPCW estimators (AIPW/TMLE) are "
              f"unstable/biased-upward here -- prefer h <= {np.quantile(t, 0.85):.0f}.", flush=True)
    risk0 = km_risk(t[ctrl], e[ctrl], float(horizon)) if ctrl.sum() else float("nan")
    n, n_ev, n_tr = len(df), int(e.sum()), int((A == 1).sum())
    rows = []
    for name in (methods or list(ESTIMATORS)):
        fn = ESTIMATORS[name]
        if verbose: print(f"  [{name}] point fit ...", flush=True)
        try:
            pt = fn(df, cols, float(horizon), adjust, seed=seed)
        except Exception as ex:
            print(f"    [{name}] POINT FAILED: {type(ex).__name__}: {str(ex)[:120]}", flush=True); continue
        if verbose:
            sp = pt.get("survival.probability", {})
            print(f"    survprob ATE={sp.get('ATE')} ATT={sp.get('ATT')} -> bootstrap x{n_boot}", flush=True)
        reps = _bootstrap(fn, df, cols, float(horizon), adjust, n_boot, seed,
                          fixed_predict=(name in CATE_LEARNERS))
        # The IPCW risk-scale estimators discard subjects whose status at h is unknown
        # (censored before h), so reporting the full cohort n beside the forests overstates
        # the data they actually used. Report the INFORMATIVE counts for AIPW/TMLE.
        n_m, nev_m, ntr_m = n, n_ev, n_tr
        if name in ("AIPW", "TMLE"):
            try:
                from ._dr import ipcw_binary
                Yb, wb = ipcw_binary(t, e, float(horizon), X=None, censoring="km")
                m = wb > 0
                n_m, nev_m, ntr_m = int(m.sum()), int(Yb[m].sum()), int(((A == 1) & m).sum())
            except Exception:
                pass
        for scale in SCALES:
            for est in ESTIMANDS:
                r = _row(name, scale, est, reps[(scale, est)], (pt.get(scale) or {}).get(est),
                         risk0, n_m, nev_m, ntr_m, n_boot=n_boot)
                if r:
                    r["n_cohort"] = int(n)          # full cohort, for comparability
                    rows.append(r)
                elif verbose:
                    print(f"    [{name}] {scale}/{est}: NO ROW "
                          f"(reps={len(reps[(scale, est)])}, need>={max(20,int(0.5*n_boot))})", flush=True)
    out = pd.DataFrame(rows)
    out.attrs["risk0"] = risk0; out.attrs["adjust"] = adjust
    out.attrs["dr_failures"] = list(DR_FAILURES)
    if DR_FAILURES and verbose:
        print(f"  DR failures ({len(DR_FAILURES)}): {DR_FAILURES[:6]}", flush=True)
    return out
