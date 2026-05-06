# Final Model Keep/Delete Manifest

This manifest records the current post-refactor keep set. It is intentionally
short; detailed result locations are maintained in
[`FINAL_REPRODUCIBILITY_NOTE.md`](FINAL_REPRODUCIBILITY_NOTE.md).

## Keep: Repository Structure

- `README.md`
- `requirements.txt`
- `pyproject.toml`
- `core/`
- `native/`
- `data/`
- `censored/`
- `non_censored/`
- `Archive/`
- `scripts/`
- `python-package/`
- `docs/`
- `outputs/`

## Keep: Data

- `data/synthetic_scenarios/`: materialized proximal synthetic scenarios.
- `data/csf_section4_synthetic/`: metadata for the prior work (2023) Section 4
  paper-replication DGP.
- `data/actg175/`: canonical ACTG175 inputs and metadata.
- `data/rhc/`: canonical RHC inputs and metadata.

## Keep: Censored Model Files

- `censored/final_censored_model.py`
- `censored/final_censored_oracle.py`
- `censored/censored_baseline.py`
- `censored/naive_censored_baseline.py`
- `censored/grf_censored_baseline.py`
- `censored/gpu_backends.py`
- `censored/plot_grouped_boxplot.py`

## Keep: Censored Runners

- `censored/run_5model_benchmark.py`
- `censored/run_synthetic_benchmark.py`
- `censored/run_conditional_ablation_addons.py`
- `censored/run_final_marginal_rcsf_full22.py`
- `censored/run_paper_style_horizon_conditional_triplet.py`
- `censored/run_paper_style_horizon_marginal_triplet.py`
- `censored/run_horizon_sweep_conditional_triplet.py`

## Keep: Non-Censored Model Files

- `non_censored/final_noncensored_model.py`
- `non_censored/final_noncensored_oracle.py`
- `non_censored/econml_noncensored_baseline.py`
- `non_censored/grf_noncensored_baseline.py`

## Keep: Non-Censored Runners

- `non_censored/run_variant_benchmark.py`
- `non_censored/run_ablation_benchmark.py`
- `non_censored/run_rcf_variant_benchmark.py`

## Keep: CSF Section 4 Replication

- `Archive/paper_csf_dgp.py`
- `Archive/run_paper_final_model_simulation.py`
- `Archive/run_paper_grf_csf_direct.R`
- `Archive/_direct_grf_imports.py`

## Keep: Shared Scripts

- `scripts/build_native.py`
- `scripts/materialize_synthetic_datasets.py`
- `scripts/preprocess_rhc.py`
- `scripts/run_final_model_bundle_benchmark.py`
- `scripts/validate_final_model_basic12.py`
- `scripts/run_grf_cf_baseline.R`
- `scripts/run_grf_cf_final_forest.R`
- `scripts/run_grf_csf_baseline.R`
- `scripts/run_grf_csf_final_forest.R`
- `scripts/run_grf_csf_survival_final_forest.R`

## Keep: Outputs

Canonical outputs are restricted to:

- `outputs/final_paper_results/`
- `outputs/appendix_results/`

New exploratory runs should go under:

- `outputs/reruns/`

## Delete Candidates

The following should not be reintroduced unless there is a specific reason:

- Local Python environments such as `.mmenv311/`.
- Matplotlib and Python caches.
- `.DS_Store`, `.Rhistory`, `__pycache__/`, and `*.pyc`.
- Historical one-off benchmark scripts.
- Pre-refactor output folders outside `outputs/final_paper_results/` and
  `outputs/appendix_results/`.
- Compatibility wrappers at repository root when canonical imports under
  `censored/` or `non_censored/` are available.

## Validation Rule

Refactoring must not alter paper-facing results. Minimum validation:

```bash
python3 -m compileall -q censored non_censored Archive scripts python-package
python3 censored/run_5model_benchmark.py --help
python3 non_censored/run_variant_benchmark.py --help
python3 Archive/run_paper_final_model_simulation.py --help
```

For result integrity, compare canonical output CSV row counts and table values
against [`FINAL_REPRODUCIBILITY_NOTE.md`](FINAL_REPRODUCIBILITY_NOTE.md).
