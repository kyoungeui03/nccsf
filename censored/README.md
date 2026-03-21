# Censored NC-CSF

This folder is the user-facing entrypoint for the survival version of the project.

The implementation lives under `/Users/kyoungeuihong/Desktop/csf_grf_new/python-package/grf`, and the scripts here are thin wrappers around the canonical runners in `/Users/kyoungeuihong/Desktop/csf_grf_new/scripts`. This keeps the runtime behavior identical while exposing a cleaner workflow.

## Default C3

The default censored `C3` is the **New C3**:

- `BestCurveLocalCensoredPCISurvivalForest`
- final learner: `econml.grf.CausalForest`
- final representation: bridge- and survival-derived summary features
- targets: `RMST` and `survival.probability`

## Legacy Old C3

- `EconmlMildShrinkNCSurvivalForest`
- final learner: `econml.CausalForestDML`
- final features: `X+W+Z`
- nuisance: `q(Z,X)`, `h(W,X,A)`, censoring KM, arm-specific Cox
- targets: `RMST` and `survival.probability`

## Main scripts

- `scripts/run_12case_8variant_benchmark.py`
- `scripts/run_rhc_b2_vs_c3.py`
- `scripts/run_c3_bestcurve.py`

## Legacy scripts

- `scripts/run_12case_old_c3_8variant_benchmark.py`
- `scripts/run_rhc_b2_vs_old_c3.py`
- `scripts/run_c3_econml_mild_shrink.py`

## Run New C3 on a custom survival dataset

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
python3 censored/scripts/run_c3_bestcurve.py \
  --train-csv /path/to/train.csv \
  --x-cols X0,X1,X2,X3,X4 \
  --w-cols W \
  --z-cols Z \
  --treatment-col A \
  --time-col time \
  --event-col event \
  --output-dir censored/outputs/c3_custom_run
```

## Run the finalized 12-case 8-variant benchmark

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
python3 censored/scripts/run_12case_8variant_benchmark.py
```

## Run the default B2 vs C3 RHC comparison

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
python3 censored/scripts/run_rhc_b2_vs_c3.py
```

## Targets

- `RMST`: supported
- `survival.probability`: supported

Use `--target survival.probability --horizon <value>` when you want survival probability at a fixed time horizon.
