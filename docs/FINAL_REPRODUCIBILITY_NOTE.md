# Final Reproducibility Note

This note fixes the paper-facing experiment layout after the repository
refactor. Use it when checking where a result came from, which runner generated
it, or which files a collaborator should inspect.

## Canonical Result Locations

All final paper outputs live under `outputs/final_paper_results/` or
`outputs/appendix_results/`.

### Main Censored Synthetic Results

Path:

```text
outputs/final_paper_results/fixed_horizon_rmst3_q90/
```

Purpose:

- Main censored synthetic benchmark at RMST horizon `h=3.0`.
- Survival-probability target at the observed-follow-up `q90` horizon.

Most-used files:

- `benchmark/results_full.csv`
- `benchmark/basic12_conditional_suite_RMST_summary.csv`
- `benchmark/basic12_conditional_suite_survival_probability_summary.csv`
- `benchmark/run_metadata.json`

Paper table mapping:

- Main censored RMST/survival summary table.
- Subgroup-bias figures under `subgroup_bias/`.

### Curated Four-Model Censored Results

Path:

```text
outputs/final_paper_results/censored_synthetic_4model/
```

Models:

- `Final Conditional`
- `Revised Conditional`
- `Final Model Oracle`
- `R-CSF Baseline`

Most-used files:

- `combined_4model_results.csv`
- `final_conditional_results.csv`
- `revised_conditional_results.csv`
- `final_model_oracle_results.csv`
- `r_csf_results.csv`

Each model file has 528 rows: `12 cases x 22 settings x 2 targets`.

### Default `max(Y)` Censored Results

Path:

```text
outputs/final_paper_results/censored_synthetic_5model_source/
```

Purpose:

- Source results for the appendix robustness check where RMST uses the default
  `h=max(Y)` convention.
- Also contains the five-model source summaries used before the fixed-horizon
  appendix tables were finalized.

Most-used files:

- `results_full.csv`
- `basic12_conditional_suite_RMST_summary.csv`
- `basic12_conditional_suite_survival_probability_summary.csv`

### ACTG175 HIV Results

Path:

```text
outputs/final_paper_results/hiv_actg175_5model_10k_bootstrap/
```

Purpose:

- Final ACTG175 real-data analysis.
- Includes four Final Conditional specifications and the re-run R-CSF baseline.

Setup:

- Dataset: ACTG175 ddI vs ZDV+ddI subset.
- Target: RMST at `h=1000` days.
- Final-stage trees: `10000`.
- Bootstrap replications: `200`.

Most-used files:

- `combined/model_summary_5models.csv`
- `combined/age_profile_comparison_5models.csv`
- `combined/feature_splits_5models.csv`
- `combined/table4_style_comparison_5models.csv`
- `combined/table5_style_blp_comparison_5models.csv`
- `shared/cohort_summary.json`
- `shared/feature_splits.csv`
- `shared/followup_histogram.png`

Important note:

The final saved HIV outputs are canonical. The pre-refactor full HIV runner was
removed during cleanup, so collaborators should inspect the saved outputs unless
a fresh HIV rerun is explicitly required.

### CSF Section 4 Paper Replication

Path:

```text
outputs/final_paper_results/csf_section4_replication/
```

Purpose:

- External validation on the original Cui et al. CSF Section 4 synthetic
  benchmark.
- This benchmark has no latent unobserved confounder and no negative-control
  proxies; all confounding is through observed covariates.

Subfolders:

- `final/paper-exact/zero/`: Final Model with all observed covariates in `X`.
- `final/paper-exact/split-5-5-5/`: Final Model with observed covariates split
  into `X/W/Z` blocks as a structural sensitivity run.
- `r-csf/paper-exact/zero/`: direct R-CSF baseline.

Most-used files in each subfolder:

- `summary.csv`
- `paper_table.csv`
- `replication_results.csv`
- `metadata.json`

Metadata fixed by the paper-exact protocol:

- Settings: `1 2 3 4`.
- Replications: `250`.
- `n_train=2000`, `n_test=2000`, `p=15`.
- RMST horizons: `(1.5, 2.0, 15.0, 3.0)`.
- Survival horizons: `(0.8, 1.2, 10.0, 2.0)`.
- R-CSF trees: `2000`.
- Nuisance survival trees: `500`.

### Appendix Feature Ablations

Path:

```text
outputs/appendix_results/final_feature_ablation_q90_h3/
```

Purpose:

- Justifies passing PCI and nuisance summaries into the final learner.
- Also supports the proxy-specific vs proxy-enriched comparison.

Most-used files:

- `results_full.csv`
- `appendix_missing_conditional_ablation_q90_h3_summary.csv`
- `run_metadata.json`

### Appendix Horizon Sweeps

Paths:

```text
outputs/appendix_results/horizon_sweep/conditional_triplet_paper_horizons_q90/
outputs/appendix_results/horizon_sweep/conditional_triplet_q70_q80/
outputs/appendix_results/horizon_sweep/marginal_triplet_oracle/
```

Purpose:

- `conditional_triplet_paper_horizons_q90/`: Final, Revised, and Oracle under
  paper-style RMST horizons plus survival `q90`.
- `conditional_triplet_q70_q80/`: additional survival quantile rows for `q70`
  and `q80`.
- `marginal_triplet_oracle/`: marginal-censoring counterpart used for the
  marginal-vs-conditional appendix.

Most-used files:

- `paper_style_horizon_summary.csv`
- `paper_style_horizon_RMST_summary.csv`
- `paper_style_horizon_survival_probability_summary.csv`
- `results_full.csv`

### Appendix Direct Baselines

Paths:

```text
outputs/appendix_results/direct_baselines/r_csf_q90_h3_full22/
outputs/appendix_results/direct_baselines/naive_censored_q90_h3_full22/
```

Purpose:

- Direct R-CSF and naive censored baseline runs at RMST `h=3.0` and survival
  `q90`.

Each `results_full.csv` has 528 rows.

## Canonical Runners

### Main Censored Synthetic Runner

Use:

```bash
python3 censored/run_5model_benchmark.py --help
```

Canonical fixed-horizon run shape:

```bash
python3 censored/run_5model_benchmark.py \
  --target both \
  --rmst-horizon 3.0 \
  --horizon-quantile 0.9 \
  --output-dir outputs/reruns/fixed_horizon_rmst3_q90
```

Use `outputs/reruns/...` for new runs unless intentionally replacing canonical
paper outputs.

### Non-Censored Synthetic Runner

Use:

```bash
python3 non_censored/run_variant_benchmark.py \
  --suite basic12 \
  --output-dir outputs/reruns/non_censored_variant
```

### CSF Section 4 Paper Replication Runner

Use:

```bash
module load r/4.4.2

python3 Archive/run_paper_final_model_simulation.py \
  --model final \
  --protocol paper-exact \
  --proxy-layout zero \
  --target both \
  --settings 1 2 3 4 \
  --reps 250 \
  --n-train 2000 \
  --n-test 2000 \
  --p 15 \
  --final-num-trees 2000 \
  --final-min-node-size 5 \
  --h-trees 500 \
  --h-min-node-size 15 \
  --save-every 10 \
  --progress-every 5 \
  --output-dir outputs/reruns/csf_section4_replication
```

For direct R-CSF:

```bash
python3 Archive/run_paper_final_model_simulation.py \
  --model r-csf \
  --protocol paper-exact \
  --proxy-layout zero \
  --target both \
  --settings 1 2 3 4 \
  --reps 250 \
  --n-train 2000 \
  --n-test 2000 \
  --p 15 \
  --num-trees-r 2000 \
  --save-every 10 \
  --progress-every 5 \
  --output-dir outputs/reruns/csf_section4_replication
```

## Validation Checklist

After changing paths, model files, or runners, run:

```bash
python3 -m compileall -q censored non_censored Archive scripts python-package
python3 censored/run_5model_benchmark.py --help
python3 non_censored/run_variant_benchmark.py --help
python3 Archive/run_paper_final_model_simulation.py --help
```

Then verify that the canonical output files still exist and that the expected
row counts are unchanged:

- fixed horizon `results_full.csv`: 1584 rows.
- censored four-model `combined_4model_results.csv`: 2112 rows.
- five-model source `results_full.csv`: 2640 rows.
- feature ablation `results_full.csv`: 2112 rows.
- paper horizon sweep `results_full.csv`: 3960 rows.
- q70/q80 sweep `results_full.csv`: 1584 rows.
- direct baselines `results_full.csv`: 528 rows each.

The refactor smoke check verified these row counts and the paper-table values
stored in the canonical CSV files.
