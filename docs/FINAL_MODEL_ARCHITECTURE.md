# Final Model Architecture

This repository has been reduced to a final-model-centered layout.

## Canonical Entry Points

### Final five-model bundle

- `scripts/run_final_model_bundle_benchmark.py`

Runs:

- `Final Model`
- `EconML Baseline`
- `R-CF Baseline`
- `R-CSF Baseline`

Supports:

- `basic12`
- `structured14`
- `rhc`

### Final 7-variant benchmark

- non-censored: `non_censored/scripts/run_final_model_variant_benchmark.py`
- censored: `scripts/run_final_model_variant_benchmark.py`

Supports:

- `basic12`
- `structured14`

### Final PCI / no PCI / raw ablation

- non-censored: `non_censored/scripts/run_final_model_ablation_benchmark.py`
- censored: `scripts/run_final_model_ablation_benchmark.py`

Supports:

- `basic12`
- `structured14`

## Canonical Model Surfaces

### Non-censored

Defined in:

- `python-package/grf/non_censored/models.py`

Primary exported classes:

- `FinalModelNCCausalForest`
- `FinalModelNoPCINCCausalForest`
- `FinalModelRawNCCausalForest`
- `StrictEconmlXWZNCCausalForest`

### Censored

Defined in:

- `python-package/grf/methods/econml_oldc3_ablation_survival.py`

Primary exported classes:

- `FinalModelCensoredSurvivalForest`
- `FinalModelNoPCICensoredSurvivalForest`
- `FinalModelRawCensoredSurvivalForest`

## Data and Benchmark Infrastructure

### Synthetic benchmark sources

- `python-package/grf/non_censored/benchmarks.py`
- `python-package/grf/benchmarks/econml_8variant.py`
- `python-package/grf/synthetic/`
- `data/synthetic_scenarios/`

### RHC support

- `scripts/preprocess_rhc.py`
- `data/rhc/`

### Direct R baselines

- `scripts/run_grf_cf_baseline.R`
- `scripts/run_grf_csf_baseline.R`

## Validation Contract

The cleanup must not change the metric outputs of the five focus models on `basic12`.

Validation script:

- `scripts/validate_final_model_basic12.py`
