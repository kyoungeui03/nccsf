# R Baseline Setup

This repository includes two direct R baselines:

- `R-CF Baseline` via `scripts/run_grf_cf_baseline.R`
- `R-CSF Baseline` via `scripts/run_grf_csf_baseline.R`

They are invoked from Python benchmark runners, so users need both:

- an R installation with a working `Rscript` executable
- the R package `grf`

## 1. Install R

Make sure `Rscript` is available.

Check:

```bash
Rscript --version
```

If `Rscript` is not on `PATH`, you can point the repo to it explicitly:

```bash
export RSCRIPT=/absolute/path/to/Rscript
```

The Python runners will resolve `Rscript` in this order:

1. `RSCRIPT` environment variable
2. `PATH`
3. `R_HOME/bin/Rscript`
4. common macOS/Linux install paths

## 2. Install `grf`

```bash
Rscript -e 'install.packages("grf", repos="https://cloud.r-project.org")'
```

For the censored baseline, the installed `grf` must provide:

- `causal_survival_forest()`

If your installed `grf` is too old, update it.

## 3. Quick smoke checks

### Non-censored direct R baseline

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
/Users/kyoungeuihong/Desktop/csf_grf_new/.mmenv311/bin/python \
  /Users/kyoungeuihong/Desktop/csf_grf_new/scripts/run_final_model_bundle_benchmark.py \
  --dataset basic12 \
  --domain non_censored \
  --case-ids 1
```

### Censored direct R baseline

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
/Users/kyoungeuihong/Desktop/csf_grf_new/.mmenv311/bin/python \
  /Users/kyoungeuihong/Desktop/csf_grf_new/scripts/run_final_model_bundle_benchmark.py \
  --dataset basic12 \
  --domain censored \
  --case-ids 1
```

## 4. Failure modes

### `Rscript executable not found`

Install R or set:

```bash
export RSCRIPT=/absolute/path/to/Rscript
```

### `R package 'grf' is required`

Install:

```bash
Rscript -e 'install.packages("grf", repos="https://cloud.r-project.org")'
```

### `causal_survival_forest() not exported`

Update `grf` to a recent version.
