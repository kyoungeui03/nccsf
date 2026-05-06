# Anonymous Submission Codebase

Research workspace for Negative-Control Causal Survival Forests (NC-CSF).
The repository is organized around two finalized model families:

- `censored/`: right-censored NC-CSF models and censored benchmark runners.
- `non_censored/`: non-censored NC-CSF models and non-censored benchmark runners.

The refactored layout keeps model code, active runners, canonical data inputs,
and paper-facing outputs in separate folders so collaborators can reproduce or
inspect results without chasing pre-refactor paths.

## Setup

Install Python dependencies from the repository root:

```bash
cd <REPO_ROOT>
python3 -m pip install -r requirements.txt
```

Direct R baselines require `Rscript` and the R package `grf`.
On the cluster this may require:

```bash
module load r/4.4.2
```

If `Rscript` is not on `PATH`, set:

```bash
export RSCRIPT=/absolute/path/to/Rscript
```

See [docs/R_BASELINE_SETUP.md](docs/R_BASELINE_SETUP.md).

## Folder Map

- [data/](data): canonical input datasets and dataset metadata.
- [censored/](censored): censored final model, oracle, baselines, and censored runners.
- [non_censored/](non_censored): non-censored final model, oracle, baselines, and non-censored runners.
- [Archive/](Archive): CSF Section 4 paper-replication DGP and runner.
- [scripts/](scripts): shared utilities and R helper scripts used by active code.
- [outputs/](outputs): canonical final paper outputs and appendix outputs.
- [docs/](docs): architecture, reproducibility, setup, and usage notes.

## Final Model Entry Points

### Censored NC-CSF

Use [censored/final_censored_model.py](censored/final_censored_model.py).

Primary public class:

```python
from censored.final_censored_model import FinalCensoredModel
```

Main API:

```python
model.fit_components(X, A, time, event, Z, W)
tau_hat = model.effect_from_components(X_test, W_test, Z_test)
```

Use `censoring_mode="conditional"` for the main paper specification. The same
class also supports `censoring_mode="marginal"` for appendix comparisons.

### Non-Censored NC-CSF

Use [non_censored/final_noncensored_model.py](non_censored/final_noncensored_model.py).

Primary public class:

```python
from non_censored.final_noncensored_model import FinalNoncensoredModel
```

Main API:

```python
model.fit_components(X, A, Y, Z, W)
tau_hat = model.effect_from_components(X_test, W_test, Z_test)
```

Detailed instructions for applying both final models to a new dataset are in
[docs/USING_FINAL_MODELS.md](docs/USING_FINAL_MODELS.md).

## Canonical Workflows

### Censored Synthetic Experiments

Read [censored/README.md](censored/README.md).

Active runners:

- [censored/run_5model_benchmark.py](censored/run_5model_benchmark.py): main censored synthetic benchmark.
- [censored/run_synthetic_benchmark.py](censored/run_synthetic_benchmark.py): smaller censored benchmark.
- [censored/run_paper_style_horizon_conditional_triplet.py](censored/run_paper_style_horizon_conditional_triplet.py): paper-style conditional horizon benchmark.
- [censored/run_horizon_sweep_conditional_triplet.py](censored/run_horizon_sweep_conditional_triplet.py): survival horizon sweep.
- [censored/run_final_marginal_rcsf_full22.py](censored/run_final_marginal_rcsf_full22.py): marginal Final / R-CSF full benchmark.

### Non-Censored Synthetic Experiments

Read [non_censored/README.md](non_censored/README.md).

Active runners:

- [non_censored/run_variant_benchmark.py](non_censored/run_variant_benchmark.py): final/oracle/EconML/R-CF comparison.
- [non_censored/run_ablation_benchmark.py](non_censored/run_ablation_benchmark.py): final-model feature ablations.
- [non_censored/run_rcf_variant_benchmark.py](non_censored/run_rcf_variant_benchmark.py): R final-forest variant benchmark.

### CSF Section 4 Replication

Read [Archive/README.md](Archive/README.md).

Active files:

- [Archive/run_paper_final_model_simulation.py](Archive/run_paper_final_model_simulation.py): paper-style CSF simulation runner.
- [Archive/paper_csf_dgp.py](Archive/paper_csf_dgp.py): Section 4 synthetic DGP.
- [Archive/run_paper_grf_csf_direct.R](Archive/run_paper_grf_csf_direct.R): direct R-CSF backend.

## Reproducibility

For collaborator-facing result locations, exact output folders, and table-source
mapping, read:

- [docs/FINAL_REPRODUCIBILITY_NOTE.md](docs/FINAL_REPRODUCIBILITY_NOTE.md)

Canonical saved results are under:

- [outputs/final_paper_results/](outputs/final_paper_results/)
- [outputs/appendix_results/](outputs/appendix_results)

## Maintenance Smoke Check

After refactoring or path changes, run:

```bash
cd <REPO_ROOT>
python3 -m compileall -q censored non_censored Archive scripts python-package
python3 censored/run_5model_benchmark.py --help
python3 non_censored/run_variant_benchmark.py --help
python3 Archive/run_paper_final_model_simulation.py --help
```

Do not overwrite canonical outputs unless intentionally regenerating paper
tables. For exploratory reruns, write to `outputs/reruns/...`.
