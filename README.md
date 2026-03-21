# csf_grf_new

Integrated research workspace for the finalized PCI+CSF pipelines.

The codebase now has a simple rule:

- **Default `C3` = New C3**
- **Old C3 = legacy comparator**

The default benchmark and RHC runners use the refactored `BestCurveLocal` family.

## Final proposed models

### Non-censored New C3

- class: `grf.non_censored.BestCurveLocalNCCausalForest`
- design: two-stage summary forest
- final learner: `econml.grf.CausalForest`
- final features: `X` plus bridge-derived summary features

### Censored New C3

- class: `grf.methods.BestCurveLocalCensoredPCISurvivalForest`
- design: two-stage summary survival forest
- final learner: `econml.grf.CausalForest`
- final features: `X` plus bridge- and survival-derived curve summaries

## Legacy models kept for reproducibility

### Non-censored Old C3

- class: `grf.non_censored.MildShrinkNCCausalForestDML`
- final learner: `econml.CausalForestDML`
- final features: raw `X+W+Z`

### Censored Old C3

- class: `grf.methods.EconmlMildShrinkNCSurvivalForest`
- final learner: `econml.CausalForestDML`
- final features: raw `X+W+Z`

## Canonical workflows

### Censored

Read [censored/README.md](/Users/kyoungeuihong/Desktop/csf_grf_new/censored/README.md).

- default New C3 8-variant benchmark:
  - [censored/scripts/run_12case_8variant_benchmark.py](/Users/kyoungeuihong/Desktop/csf_grf_new/censored/scripts/run_12case_8variant_benchmark.py)
- default New C3 RHC comparison:
  - [censored/scripts/run_rhc_b2_vs_c3.py](/Users/kyoungeuihong/Desktop/csf_grf_new/censored/scripts/run_rhc_b2_vs_c3.py)
- explicit Old C3 benchmark:
  - [censored/scripts/run_12case_old_c3_8variant_benchmark.py](/Users/kyoungeuihong/Desktop/csf_grf_new/censored/scripts/run_12case_old_c3_8variant_benchmark.py)
- explicit Old C3 RHC comparison:
  - [censored/scripts/run_rhc_b2_vs_old_c3.py](/Users/kyoungeuihong/Desktop/csf_grf_new/censored/scripts/run_rhc_b2_vs_old_c3.py)
- custom New C3 runner:
  - [censored/scripts/run_c3_bestcurve.py](/Users/kyoungeuihong/Desktop/csf_grf_new/censored/scripts/run_c3_bestcurve.py)
- custom Old C3 runner:
  - [censored/scripts/run_c3_econml_mild_shrink.py](/Users/kyoungeuihong/Desktop/csf_grf_new/censored/scripts/run_c3_econml_mild_shrink.py)

### Non-censored

Read [non_censored/README.md](/Users/kyoungeuihong/Desktop/csf_grf_new/non_censored/README.md).

- default New C3 8-variant benchmark:
  - [non_censored/scripts/run_12case_8variant_benchmark.py](/Users/kyoungeuihong/Desktop/csf_grf_new/non_censored/scripts/run_12case_8variant_benchmark.py)
- default New C3 RHC comparison:
  - [non_censored/scripts/run_rhc_b2_vs_c3.py](/Users/kyoungeuihong/Desktop/csf_grf_new/non_censored/scripts/run_rhc_b2_vs_c3.py)
- explicit Old C3 benchmark:
  - [non_censored/scripts/run_12case_old_c3_8variant_benchmark.py](/Users/kyoungeuihong/Desktop/csf_grf_new/non_censored/scripts/run_12case_old_c3_8variant_benchmark.py)
- explicit Old C3 RHC comparison:
  - [non_censored/scripts/run_rhc_b2_vs_old_c3.py](/Users/kyoungeuihong/Desktop/csf_grf_new/non_censored/scripts/run_rhc_b2_vs_old_c3.py)
- custom New C3 runner:
  - [non_censored/scripts/run_c3_bestcurve.py](/Users/kyoungeuihong/Desktop/csf_grf_new/non_censored/scripts/run_c3_bestcurve.py)
- custom Old C3 runner:
  - [non_censored/scripts/run_c3_econml_mild_shrink.py](/Users/kyoungeuihong/Desktop/csf_grf_new/non_censored/scripts/run_c3_econml_mild_shrink.py)

## Maintenance

### Baseline parity check

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
PYTHONPATH=python-package python3 scripts/compare_with_reference.py
```

### Rebuild the native library

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
python3 scripts/build_native.py --force
```

### RHC preprocessing

- [scripts/preprocess_rhc.py](/Users/kyoungeuihong/Desktop/csf_grf_new/scripts/preprocess_rhc.py)

Historical notebooks and raw tables are preserved under [notebooks/rhc](/Users/kyoungeuihong/Desktop/csf_grf_new/notebooks/rhc) and [data/rhc](/Users/kyoungeuihong/Desktop/csf_grf_new/data/rhc).
