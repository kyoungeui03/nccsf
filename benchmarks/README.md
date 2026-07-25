# `benchmarks` — external comparison baselines

Four (optionally five) treatment-effect baselines for right-censored survival data, all
reporting the **same estimand set** so their rows stack into one table with whatever method
you are evaluating.

```
{RMST, survival.probability} x {ATE, ATT}
```

**Sign convention on both scales: negative = harm.**

| method | what it is | scales |
|---|---|---|
| `RSF-T` | random survival forest, one per arm (T-learner) | survprob + RMST |
| `RSF-S` | random survival forest with `A` as a feature (S-learner) | survprob + RMST |
| `AIPW` | doubly-robust IPCW risk at `h` | survprob only |
| `TMLE` | targeted MLE of the IPCW risk at `h` | survprob only |
| `BART` | discrete-time survival BART, S-learner (posterior interval) | survprob + RMST |

`AIPW` / `TMLE` estimate `P(event by h)`, a binary-at-`h` quantity, so **RMST is undefined for
them by construction** — a missing RMST row there is correct, not a failure.

The causal survival forest baseline is **not** here: this repository already ships it as
[`censored/censored_baseline.py`](../censored/censored_baseline.py) (the R-grf CSF). A second
copy would be free to drift from the one the papers use.

## Install

```bash
pip install numpy pandas scikit-learn scikit-survival lifelines
```

`BART` is optional and needs its own environment (pymc pulls a compiler toolchain):

```bash
python -m venv bart_venv && ./bart_venv/bin/pip install pymc pymc-bart numpy pandas
export BART_PYTHON=$PWD/bart_venv/bin/python
```

Without `BART_PYTHON` the BART row is **skipped with a printed reason**, never an exception —
an optional dependency should not take down the comparison.

## Use — Python

```python
from benchmarks import run_all

res = run_all(
    df,                      # DataFrame with A (0/1), time, event (0/1) + covariates
    horizon=90,
    x_cols=["age", "sex", ...],   # confounders
    w_cols=["W"],                 # outcome proxies  (NCO)   -- optional
    z_cols=["Z"],                 # treatment proxies (NCE)  -- optional
    n_boot=100,
    seed=0,
)
```

One row per `(method, scale, estimand)`, with:

| column | meaning |
|---|---|
| `point` | estimate on the original sample |
| `bs_mean` | mean over bootstrap replicates |
| `ci_lo`, `ci_hi` | **BCa 95 % interval** |
| `point_vs_bsmean` | bootstrap-bias diagnostic |
| `bs_lo`, `bs_hi`, `bs_range` | min/max of the replicates |
| `n_reps` | replicates that succeeded |
| `excludes_zero` | interval excludes 0 **and sits on the same side as the estimate** |
| `n`, `n_ev`, `n_treated`, `n_cohort` | analysed counts (see note below) |
| `risk0`, `rr`, `e_value`, `e_value_ci`, `evalue_method` | survprob scale only |

## Use — command line

```bash
python -m benchmarks.runner --csv cohort.csv --horizon 90 \
    --x-cols age,sex,ckd --w-cols W --z-cols Z --out results.csv
```

`--methods RSF-T,AIPW` runs a subset; `--adjust x` drops the proxies from the adjustment set.

## What keeps these comparable

Not merely adjacent — deliberately matched:

- **Stratified bootstrap.** The resample is stratified on the `(A, event)` 2×2 with all four
  cell sizes held fixed, and the CATE learners refit on the resample but **predict on the
  fixed original cohort**. A plain i.i.d. resample lets the treated and event counts wobble,
  adding a variance component and widening these intervals by construction — which would
  flatter whatever they are compared against.
- **One `risk0`.** The E-value baseline is the method-agnostic control-arm Kaplan-Meier risk
  at `h`, not each estimator's own fitted `S0`, so every E-value in the table means the same
  thing. It carries a degeneracy guard that refuses to emit an E-value when the implied
  `risk1` leaves `(0, 1)`.
- **Same-side `excludes_zero`.** A BCa interval can land entirely on the opposite side of a
  bias-corrected point estimate; reading that as significance in the reported direction would
  be wrong, so both conditions are required.
- **Honest counts.** `AIPW`/`TMLE` discard subjects whose status at `h` is unknown (censored
  before `h`), so their `n` is the **informative** count while `n_cohort` stays the full
  cohort. Their `n` being smaller than the forests' is expected.
- **Standardised design on the DR path only.** The DR nuisances are penalised logistic
  regressions, which are not equivariant under rescaling; on a raw covariate stack lbfgs does
  not converge and the estimates can flip sign. Forests are scale-invariant, so this touches
  `AIPW`/`TMLE` only.

`_stats.py` and `_dr.py` are **verbatim** copies from the DUET pipeline (`run_track_v2.py`,
`validation/estimators.py`) so this repository stands alone without importing DUET, and so the
shared machinery cannot drift from the version those results were produced with. Re-vendor
from DUET rather than editing them here.
