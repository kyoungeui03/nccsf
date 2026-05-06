# Final Model Architecture

This repository is organized around the finalized NC-CSF model surfaces. The
canonical implementation files now live in `censored/` and `non_censored/`.
Older package backends under `python-package/` are retained only when active
runners still need shared DGP, oracle, native, or R-runtime utilities.

## Canonical Model Surfaces

### Censored NC-CSF

Defined in:

- [`censored/final_censored_model.py`](../censored/final_censored_model.py)
- [`censored/final_censored_oracle.py`](../censored/final_censored_oracle.py)
- [`censored/censored_baseline.py`](../censored/censored_baseline.py)
- [`censored/naive_censored_baseline.py`](../censored/naive_censored_baseline.py)
- [`censored/grf_censored_baseline.py`](../censored/grf_censored_baseline.py)

Primary public class:

```python
from censored.final_censored_model import FinalCensoredModel
```

Main API:

```python
model.fit_components(X, A, time, event, Z, W)
tau_hat = model.effect_from_components(X_test, W_test, Z_test)
```

The same final model supports both `censoring_mode="conditional"` and
`censoring_mode="marginal"`. The paper's main censored model uses conditional
censoring.

### Non-Censored NC-CSF

Defined in:

- [`non_censored/final_noncensored_model.py`](../non_censored/final_noncensored_model.py)
- [`non_censored/final_noncensored_oracle.py`](../non_censored/final_noncensored_oracle.py)
- [`non_censored/econml_noncensored_baseline.py`](../non_censored/econml_noncensored_baseline.py)
- [`non_censored/grf_noncensored_baseline.py`](../non_censored/grf_noncensored_baseline.py)

Primary public class:

```python
from non_censored.final_noncensored_model import FinalNoncensoredModel
```

Main API:

```python
model.fit_components(X, A, Y, Z, W)
tau_hat = model.effect_from_components(X_test, W_test, Z_test)
```

## Active Runner Surfaces

### Censored

- [`censored/run_5model_benchmark.py`](../censored/run_5model_benchmark.py): main censored synthetic benchmark.
- [`censored/run_synthetic_benchmark.py`](../censored/run_synthetic_benchmark.py): smaller censored benchmark.
- [`censored/run_paper_style_horizon_conditional_triplet.py`](../censored/run_paper_style_horizon_conditional_triplet.py): paper-style RMST horizon comparison.
- [`censored/run_horizon_sweep_conditional_triplet.py`](../censored/run_horizon_sweep_conditional_triplet.py): survival horizon sweep.
- [`censored/run_final_marginal_rcsf_full22.py`](../censored/run_final_marginal_rcsf_full22.py): Final marginal / R-CSF full benchmark.

### Non-Censored

- [`non_censored/run_variant_benchmark.py`](../non_censored/run_variant_benchmark.py): final/oracle/EconML/R-CF comparison.
- [`non_censored/run_ablation_benchmark.py`](../non_censored/run_ablation_benchmark.py): final feature ablations.
- [`non_censored/run_rcf_variant_benchmark.py`](../non_censored/run_rcf_variant_benchmark.py): R final-forest variant benchmark.

### CSF Section 4 Replication

- [`Archive/run_paper_final_model_simulation.py`](../Archive/run_paper_final_model_simulation.py): prior work (2023) synthetic benchmark runner.
- [`Archive/paper_csf_dgp.py`](../Archive/paper_csf_dgp.py): paper-style synthetic DGP.
- [`Archive/run_paper_grf_csf_direct.R`](../Archive/run_paper_grf_csf_direct.R): direct R-CSF backend.

## Data and Output Boundaries

- [`data/`](../data/): canonical inputs and metadata only.
- [`outputs/final_paper_results/`](../outputs/final_paper_results/): final paper-facing outputs.
- [`outputs/appendix_results/`](../outputs/appendix_results/): appendix and robustness outputs.
- `outputs/reruns/`: recommended location for new exploratory reruns.

## Related Notes

- For applying the final censored and non-censored models to a new dataset, see
  [`USING_FINAL_MODELS.md`](USING_FINAL_MODELS.md).
- For paper result locations and table-source mappings, see
  [`FINAL_REPRODUCIBILITY_NOTE.md`](FINAL_REPRODUCIBILITY_NOTE.md).
- For R baseline setup, see
  [`R_BASELINE_SETUP.md`](R_BASELINE_SETUP.md).

## Validation Contract

Refactors must not change the saved paper-facing result values. After path or
runner changes, run:

```bash
python3 -m compileall -q censored non_censored Archive scripts python-package
python3 censored/run_5model_benchmark.py --help
python3 non_censored/run_variant_benchmark.py --help
python3 Archive/run_paper_final_model_simulation.py --help
```

Then verify that the canonical output CSV files in `outputs/final_paper_results/`
and `outputs/appendix_results/` still match the table values used in the paper.
