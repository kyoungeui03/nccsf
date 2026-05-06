# Censored Models

This folder contains the canonical censored-model implementations and runners
for the NC-CSF project.

## Core models

- `final_censored_model.py`: main Censored NC-CSF implementation. It supports
  both `censoring_mode="conditional"` and `censoring_mode="marginal"`.
- `final_censored_oracle.py`: oracle comparator for synthetic experiments with
  access to the latent confounder `U`.
- `censored_baseline.py`: matched no-PCI censored baseline.
- `naive_censored_baseline.py`: naive observed-feature censored baseline.
- `grf_censored_baseline.py`: direct R `grf` causal survival forest wrapper.

## Main runners

- `run_5model_benchmark.py`: canonical synthetic benchmark runner for censored
  model comparisons.
- `run_synthetic_benchmark.py`: smaller synthetic benchmark entry point.
- `run_paper_style_horizon_conditional_triplet.py`: paper-style fixed-horizon
  conditional comparison runner.
- `run_horizon_sweep_conditional_triplet.py`: horizon sweep runner.
- `run_final_marginal_rcsf_full22.py`: marginal Final / R-CSF comparison
  runner.

## Using the model on new data

For direct use on a user-provided censored dataset, call
`FinalCensoredModel.fit_components(X, A, time, event, Z, W)` and then
`effect_from_components(X_test, W_test, Z_test)`.

Detailed instructions are in
[`docs/USING_FINAL_MODELS.md`](/Users/kyoungeuihong/Desktop/csf_grf_new/docs/USING_FINAL_MODELS.md).

## Paper outputs

Canonical saved outputs and table-source mappings are documented in
[`docs/FINAL_REPRODUCIBILITY_NOTE.md`](/Users/kyoungeuihong/Desktop/csf_grf_new/docs/FINAL_REPRODUCIBILITY_NOTE.md).

The old `single_file_censored_models/` package name has been retired. Import
from `censored.*` instead.
