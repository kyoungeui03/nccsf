# Final Model Cleanup Manifest

This manifest defines the canonical keep set for the repository cleanup.

## Cleanup Goal

Preserve only the code and data required to:

- run the finalized non-censored and censored `Final Model`
- run the direct baselines
  - `EconML Baseline`
  - `R-CF Baseline`
  - `R-CSF Baseline`
- run the finalized `Final Model` ablation study
  - `Final Model (PCI)`
  - `Final Model (No PCI)`
  - `Final Model (Raw)`
- run the finalized `Final Model` 7-variant benchmark
- preserve synthetic dataset testing infrastructure
- preserve reproducibility of the final basic runners

## Keep: Top-Level

- `README.md`
- `THIRD_PARTY_NOTICE.md`
- `pyproject.toml`
- `.gitignore`
- `core/`
- `native/`
- `data/rhc/`
- `data/synthetic_scenarios/`
- `python-package/`
- `python-package/grf/__init__.py`
- `scripts/build_native.py`
- `scripts/materialize_synthetic_datasets.py`
- `scripts/preprocess_rhc.py`
- `scripts/run_final_model_bundle_benchmark.py`
- `scripts/run_final_model_variant_benchmark.py`
- `scripts/run_final_model_ablation_benchmark.py`
- `scripts/run_grf_cf_baseline.R`
- `scripts/run_grf_csf_baseline.R`
- `non_censored/scripts/run_final_model_variant_benchmark.py`
- `non_censored/scripts/run_final_model_ablation_benchmark.py`
- `docs/FINAL_MODEL_KEEP_DELETE_MANIFEST.md`
- `docs/FINAL_MODEL_ARCHITECTURE.md`
- `docs/FINAL_MODEL_EXPECTED_BASIC12.json`
- `scripts/validate_final_model_basic12.py`

## Keep: Python Package Surfaces

- `python-package/grf/non_censored/models.py`
- `python-package/grf/non_censored/benchmarks.py`
- `python-package/grf/non_censored/data_generation.py`
- `python-package/grf/non_censored/__init__.py`
- `python-package/grf/censored/__init__.py`
- `python-package/grf/methods/econml_oldc3_ablation_survival.py`
- `python-package/grf/methods/econml_mild_shrink.py`
- `python-package/grf/methods/baseline.py`
- `python-package/grf/methods/__init__.py`
- `python-package/grf/benchmarks/econml_8variant.py`
- `python-package/grf/backends/`
- `python-package/grf/core/`
- `python-package/grf/synthetic/`

## Keep: Canonical Outputs Only

All historical outputs are deleted.

Only the following post-cleanup validation outputs may remain:

- `outputs/validation_postcleanup_final_model_bundle_basic12/`
- `outputs/validation_postcleanup_final_model_ablation_c_basic12/`
- `non_censored/outputs/validation_postcleanup_final_model_ablation_nc_basic12/`

Optional smoke outputs may be removed after validation.

## Delete

- all historical benchmark outputs under:
  - `outputs/`
  - `non_censored/outputs/`
  - except the post-cleanup validation directories listed above
- all historical render scripts
- all historical one-off benchmark scripts
- all overnight launcher scripts
- all comparison/report scripts unrelated to the finalized models
- all notebooks under `notebooks/`
- all obsolete wrapper files once canonical runners are in place

## Non-Negotiable Validation Rule

After cleanup, the preserved basic runner for the five focus models must reproduce the same metric outputs as before cleanup.

The five focus outputs are:

- non-censored `Final Model`
- non-censored `EconML Baseline`
- non-censored `R-CF Baseline`
- censored `Final Model`
- censored `R-CSF Baseline`

These are validated through:

- `scripts/run_final_model_bundle_benchmark.py --dataset basic12 --domain both`

The final-model ablation runners must also remain executable on:

- `basic12`
- `structured14`
