# Non-Censored NC-CSF

This folder is the user-facing entrypoint for the non-censored outcome version of the project.

The implementation lives under `/Users/kyoungeuihong/Desktop/csf_grf_new/python-package/grf/non_censored`, and the scripts here are the canonical entrypoints for that workflow.

## Default C3

The default non-censored `C3` is the **New C3**:

- class: `BestCurveLocalNCCausalForest`
- final learner: `econml.grf.CausalForest`
- final representation: bridge-derived summary features

## Legacy Old C3

- class: `MildShrinkNCCausalForestDML`
- final learner: `econml.CausalForestDML`
- final features: `X+W+Z`
- nuisance: `q(Z,X)` and `h(W,X,A)`
- stabilization: `q` clipping, outcome winsorization before `h` fit, residual clipping

## Main scripts

- `scripts/run_12case_8variant_benchmark.py`
- `scripts/run_rhc_b2_vs_c3.py`
- `scripts/run_c3_bestcurve.py`

## Legacy scripts

- `scripts/run_12case_old_c3_8variant_benchmark.py`
- `scripts/run_rhc_b2_vs_old_c3.py`
- `scripts/run_c3_econml_mild_shrink.py`

## Run New C3 on a custom outcome dataset

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
python3 non_censored/scripts/run_c3_bestcurve.py \
  --train-csv /path/to/train.csv \
  --x-cols X0,X1,X2,X3,X4 \
  --w-cols W \
  --z-cols Z \
  --treatment-col A \
  --outcome-col outcome \
  --output-dir non_censored/outputs/c3_custom_run
```

## Run the 12-case 8-variant benchmark

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
python3 non_censored/scripts/run_12case_8variant_benchmark.py
```

## Run the default RHC direct-outcome comparison

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
python3 non_censored/scripts/run_rhc_b2_vs_c3.py
```

This uses the historical RHC direct-outcome setup:

- treatment: `A = swang1`
- outcome: `Y = t3d30`
- proxies: `Z = [pafi1, paco21]`, `W = [ph1, hema1]`
- preprocessing is rebuilt from raw RHC through [preprocess_rhc.py](/Users/kyoungeuihong/Desktop/csf_grf_new/scripts/preprocess_rhc.py)
