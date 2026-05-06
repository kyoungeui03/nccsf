# Non-Censored NC-CSF

This folder is the canonical entrypoint for the non-censored outcome version of
the project. It mirrors the refactored `censored/` layout: model files live at
the folder root, and benchmark runners live directly under `non_censored/`.

## Core model files

- `final_noncensored_model.py`: final non-censored NC-CSF model used in the paper benchmarks.
- `final_noncensored_oracle.py`: oracle comparator that uses latent `U` for synthetic upper-bound experiments.
- `econml_noncensored_baseline.py`: raw `[X,W,Z]` EconML `CausalForestDML` baseline.
- `grf_noncensored_baseline.py`: direct R `grf::causal_forest` raw-feature baseline wrapper.

## Benchmark runners

- `run_variant_benchmark.py`: final/oracle/EconML/R-CF non-censored comparison.
- `run_ablation_benchmark.py`: non-censored final-model feature ablations.
- `run_rcf_variant_benchmark.py`: R final-forest variant comparison.

Default generated outputs now go under `outputs/non_censored/`.

## Example commands

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
python3 non_censored/run_variant_benchmark.py --suite basic12
```

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
python3 non_censored/run_ablation_benchmark.py --suite basic12
```

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
python3 non_censored/run_rcf_variant_benchmark.py --suite basic12
```

## Notes

The production single-file models are intentionally kept self-contained where
possible. The oracle and R baseline wrappers depend on the package helpers under
`python-package/grf/non_censored` because they are benchmark-only paths that
reuse existing DGP oracle and R-runtime utilities.

For direct use on a user-provided dataset, see
[`docs/USING_FINAL_MODELS.md`](/Users/kyoungeuihong/Desktop/csf_grf_new/docs/USING_FINAL_MODELS.md).

For canonical result locations and table-source mappings, see
[`docs/FINAL_REPRODUCIBILITY_NOTE.md`](/Users/kyoungeuihong/Desktop/csf_grf_new/docs/FINAL_REPRODUCIBILITY_NOTE.md).
