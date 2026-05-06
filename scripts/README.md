# Shared Scripts

This folder is intentionally small. Model-specific runners live in `censored/`,
`non_censored/`, or `Archive/`; this folder keeps only shared project utilities
and R helper scripts that are referenced by the active code.

## Python Utilities

- `build_native.py`: build the optional native C++ library used by the package backend.
- `materialize_synthetic_datasets.py`: generate canonical synthetic datasets under `data/synthetic_scenarios/`.
- `preprocess_rhc.py`: clean and encode the RHC dataset.
- `run_final_model_bundle_benchmark.py`: cross-domain validation bundle runner.
- `validate_final_model_basic12.py`: validation wrapper for the retained bundle runner.

## R Helpers

- `run_grf_cf_baseline.R`: direct `grf::causal_forest` non-censored baseline.
- `run_grf_cf_final_forest.R`: R `causal_forest` final-stage helper for non-censored variants.
- `run_grf_csf_baseline.R`: direct `grf::causal_survival_forest` censored baseline.
- `run_grf_csf_final_forest.R`: R `causal_forest` final-stage helper for censored residualized variants.
- `run_grf_csf_survival_final_forest.R`: R `causal_survival_forest` final-stage helper.
