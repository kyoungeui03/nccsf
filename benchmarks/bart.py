"""BART survival treatment-effect baseline (discrete-time hazard S-learner).

Runs as a SUBPROCESS in an environment that has pymc + pymc-bart, so the heavy Bayesian
stack never has to be installed alongside the rest of the suite. Call it through
``benchmarks.run_bart`` (which writes the cohort/spec, shells out, and returns rows in the
same schema as the other baselines), or directly as a script.

Method = Sparapani et al. discrete-time survival BART:
  1. Bin follow-up into K intervals (distinct event times, capped for tractability).
  2. Person-period long format: subject i contributes rows for bins 1..k_i (through their
     event/censoring bin); binary y=1 only in their event bin.
  3. BART probit hazard  lambda(x,a,t_j) = Phi( f_BART([features, A, bin]) ).
  4. Survival S(h|x,a) = prod_{t_j<=h} (1 - lambda);  RMST = sum_j S(t_j|x,a)*dt_j.
  5. S-learner counterfactual: predict hazard for every subject x every bin under A=1 and A=0
     via posterior predictive, giving POSTERIOR DRAWS of the population ATE (its own Bayesian
     uncertainty -- no bootstrap). survprob ATE = mean_i[S1(h)-S0(h)]; harm < 0.

Emits JSON with the posterior draws so the caller builds the same {point, CI, e-value} row
(e-value via the shared evalue_multi on the survprob scale, using the shared KM risk0).
"""
import sys, json, argparse
import numpy as np, pandas as pd


def _num(df, cols):
    if not cols:
        return np.zeros((len(df), 0))
    M = df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    M = np.where(np.isfinite(M), M, np.nan)
    for j in range(M.shape[1]):
        c = M[:, j]
        if np.isnan(c).any():
            med = np.nanmedian(c); c[np.isnan(c)] = med if np.isfinite(med) else 0.0; M[:, j] = c
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)      # cohort CSV (already derived)
    ap.add_argument("--spec", required=True)        # JSON: x/w/z cols, horizon, adjust
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-bins", type=int, default=12)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--tune", type=int, default=200)
    ap.add_argument("--trees", type=int, default=50)
    ap.add_argument("--subsample", type=int, default=2500)   # BART is slow; cap rows
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import pymc as pm, pymc_bart as pmb
    spec = json.load(open(a.spec))
    df = pd.read_csv(a.cohort)
    rng = np.random.default_rng(a.seed)
    original_row = np.arange(len(df), dtype=int)
    if a.subsample and len(df) > a.subsample:
        keep = rng.choice(len(df), a.subsample, replace=False)
        original_row = original_row[keep]
        df = df.iloc[keep].reset_index(drop=True)

    h = float(spec["horizon"]); adjust = spec.get("adjust", "xwz")
    X = _num(df, spec["x_cols"])
    if adjust != "x":
        X = np.hstack([m for m in (X, _num(df, spec["w_cols"]), _num(df, spec["z_cols"])) if m.shape[1]])
    A = df["A"].to_numpy(float); t = df["time"].to_numpy(float); e = df["event"].to_numpy(int)
    n, p = X.shape

    # ---- time bins: distinct event times up to h, capped by quantile ----
    ev_times = np.unique(t[(e == 1) & (t <= h)])
    if len(ev_times) == 0:
        json.dump({"error": "no events before horizon"}, open(a.out, "w")); return
    if len(ev_times) > a.max_bins:
        ev_times = np.quantile(ev_times, np.linspace(0, 1, a.max_bins))
    bins = np.unique(np.round(ev_times, 6))
    # CLOSE THE INTERVAL AT THE HORIZON. The bins come from EVENT times <= h, so the last bin
    # ends at max(event time), not at h. Integrating RMST only over [0, max(bins)] silently
    # drops the tail [max(bins), h]. That does NOT cancel in the arm difference: the dropped
    # tail is weighted by each arm's OWN survival, S1 vs S0, so a genuine RMST gap over the
    # tail is discarded -- which shrinks BART's effect toward zero and flatters our method.
    if bins.size == 0 or bins[-1] < h - 1e-9:
        bins = np.concatenate([bins, [float(h)]])
    edges = np.concatenate(([0.0], bins))
    dt = np.diff(edges)                                  # width of each bin, now spanning [0, h]
    K = len(bins)
    # bin index reached by each subject (last bin with edge < their time), for expansion
    def last_bin(ti):  # number of bins the subject is observed through (1..K)
        return int(np.searchsorted(bins, min(ti, bins[-1]), side="left")) + 1 if ti >= bins[0] else 1

    # ---- person-period long design for FITTING ----
    rows_x, rows_bin, rows_y = [], [], []
    for i in range(n):
        ki = min(last_bin(t[i]), K)
        for j in range(ki):
            rows_x.append(i); rows_bin.append(j)
            # event in this bin iff subject had the event and this is their last observed bin
            y = 1 if (e[i] == 1 and j == ki - 1 and t[i] <= h) else 0
            rows_y.append(y)
    ridx = np.asarray(rows_x); rbin = np.asarray(rows_bin); Yl = np.asarray(rows_y, float)
    # design = [X_i, A_i, bin_center]
    bin_center = (edges[:-1] + edges[1:]) / 2.0
    Xl = np.hstack([X[ridx], A[ridx, None], bin_center[rbin, None]])

    # ---- BART probit hazard (Xd AND yd are Data so the design can be swapped for OOS pred) ----
    Ncf = n * K
    with pm.Model() as m:
        Xd = pm.Data("Xd", Xl)
        yd = pm.Data("yd", Yl)
        mu = pmb.BART("mu", Xd, yd, m=a.trees)
        p_ = pm.Deterministic("p", pm.math.invprobit(mu))
        pm.Bernoulli("y", p=p_, observed=yd, shape=Xd.shape[0])
        idata = pm.sample(draws=a.draws, tune=a.tune, chains=2, cores=1,
                          random_seed=a.seed, compute_convergence_checks=False,
                          progressbar=False)

    # ---- counterfactual design: every subject x every bin, under A=1 and A=0 ----
    sub = np.repeat(np.arange(n), K); bcol = np.tile(np.arange(K), n)
    def cf_design(aval):
        return np.hstack([X[sub], np.full((Ncf, 1), aval, float), bin_center[bcol, None]])

    def haz_draws(aval):
        with m:
            pm.set_data({"Xd": cf_design(aval), "yd": np.zeros(Ncf)})
            # sample_vars=["mu"] FORCES BART to re-evaluate its trees at the new design
            # (otherwise mu is implicitly frozen at the training design -> wrong OOS pred).
            try:
                pp = pm.sample_posterior_predictive(idata, var_names=["p"], sample_vars=["mu"],
                                                    progressbar=False, random_seed=a.seed)
            except TypeError:
                pp = pm.sample_posterior_predictive(idata, var_names=["mu", "p"],
                                                    progressbar=False, random_seed=a.seed)
        return np.asarray(pp.posterior_predictive["p"]).reshape(-1, n, K)   # (S, n, K)

    l1 = haz_draws(1.0); l0 = haz_draws(0.0)             # discrete hazards, (S,n,K)
    S = min(l1.shape[0], l0.shape[0]); l1, l0 = l1[:S], l0[:S]
    in_h = bins <= h
    # S(h) = prod_{bins<=h}(1-lambda); RMST = sum_j S(edge_j)*dt_j over bins<=h
    def surv_and_rmst(lam):
        surv_steps = np.cumprod(1.0 - lam, axis=2)       # (S,n,K) survival AFTER each bin
        Sh = np.where(in_h, surv_steps, 1.0)[:, :, in_h][:, :, -1] if in_h.any() else np.ones((S, n))
        # RMST: S at START of each bin (shift), times bin width, summed over bins<=h
        S_start = np.concatenate([np.ones((S, n, 1)), surv_steps[:, :, :-1]], axis=2)
        rmst = np.sum((S_start * dt[None, None, :])[:, :, in_h], axis=2)  # (S,n)
        return Sh, rmst
    S1h, R1 = surv_and_rmst(l1); S0h, R0 = surv_and_rmst(l0)
    # Per-posterior-draw population contrasts. ATE averages over everyone; ATT averages
    # over the TREATED only -- the same split the other baselines and the NC-CSF headline use.
    sp_unit = S1h - S0h                                   # (S, n) per-draw, per-subject
    rm_unit = R1 - R0
    tr = A == 1
    draws = {
        ("survival.probability", "ATE"): sp_unit.mean(axis=1),
        ("survival.probability", "ATT"): sp_unit[:, tr].mean(axis=1) if tr.any() else None,
        ("RMST", "ATE"): rm_unit.mean(axis=1),
        ("RMST", "ATT"): rm_unit[:, tr].mean(axis=1) if tr.any() else None,
    }

    out = {"method": "BART", "n": int(n), "n_ev": int(e.sum()), "n_treated": int(tr.sum()),
           "K_bins": int(K), "adjust": adjust, "uncertainty": "posterior (not bootstrap)",
           "estimands": {},
           # Posterior-mean subject-level contrasts are used by the simulation HTE runner.
           # row_index maps back to the input cohort when --subsample is nonzero.
           "row_index": original_row.tolist(),
           "unit_effects": {
               "survival.probability": sp_unit.mean(axis=0).tolist(),
               "RMST": rm_unit.mean(axis=0).tolist(),
           }}
    for (scale, est), d in draws.items():
        if d is None or not len(d):
            continue
        out["estimands"][f"{scale}|{est}"] = {
            "point": float(np.mean(d)),
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "n_draws": int(len(d)),
        }
    # keep the flat ATE keys for backward compatibility with the earlier smoke output
    sp_ate = draws[("survival.probability", "ATE")]; rm_ate = draws[("RMST", "ATE")]
    out.update({"survprob_ate": float(sp_ate.mean()),
                "survprob_ci": [float(np.percentile(sp_ate, 2.5)), float(np.percentile(sp_ate, 97.5))],
                "rmst_ate": float(rm_ate.mean())})
    json.dump(out, open(a.out, "w"))
    for k, v in out["estimands"].items():
        print(f"[BART] {k:28s} {v['point']:+.4f} [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}]")


if __name__ == "__main__":
    main()
