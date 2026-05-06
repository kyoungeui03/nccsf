# Output Directory Layout

This directory keeps only current paper-facing outputs and appendix outputs.
Superseded, partial, smoke-test, and pre-refactor output folders were removed to avoid
confusing stale results with canonical experiment results.

For the collaborator-facing mapping from paper tables to output files, see
[`docs/FINAL_REPRODUCIBILITY_NOTE.md`](/Users/kyoungeuihong/Desktop/csf_grf_new/docs/FINAL_REPRODUCIBILITY_NOTE.md).

## Final paper results

- `final_paper_results/censored_synthetic_4model/`
  - Curated four-model censored synthetic benchmark results.
  - Models: Final Conditional, Revised Conditional, Final Model Oracle, R-CSF Baseline.
  - Each model has 528 rows, corresponding to 12 cases x 22 settings x 2 targets.
- `final_paper_results/censored_synthetic_5model_source/`
  - Source five-model censored synthetic results used for final and appendix tables.
  - Includes Final Conditional, Final Conditional Oracle, Revised Conditional, Revised Marginal, and Strict Conditional.
- `final_paper_results/hiv_actg175_5model_10k_bootstrap/`
  - Final ACTG175 HIV analysis outputs.
  - Contains four Final Conditional specifications plus the re-run R-CSF baseline.
  - Uses RMST horizon h=1000 days, 10000 final-stage trees, and 200 bootstrap replications.
- `final_paper_results/csf_section4_replication/`
  - Replication of the original CSF Section 4 synthetic benchmark.
  - Contains Final all-X, Final split-5-5-5, and direct R-CSF runs.
- `final_paper_results/fixed_horizon_rmst3_q90/`
  - Main fixed-horizon censored synthetic outputs.
  - Uses RMST h=3.0 and survival probability at the q90 horizon.

## Appendix results

- `appendix_results/final_feature_ablation_q90_h3/`
  - Final learner feature ablations at RMST h=3.0 and survival q90.
- `appendix_results/horizon_sweep/conditional_triplet_paper_horizons_q90/`
  - Conditional triplet results for paper-style RMST horizons and survival q90.
- `appendix_results/horizon_sweep/conditional_triplet_q70_q80/`
  - Additional survival horizon sweep outputs for q70 and q80.
- `appendix_results/horizon_sweep/marginal_triplet_oracle/`
  - Marginal-censoring horizon sweep with oracle comparator.
- `appendix_results/direct_baselines/r_csf_q90_h3_full22/`
  - Direct R-CSF baseline at RMST h=3.0 and survival q90.
- `appendix_results/direct_baselines/naive_censored_q90_h3_full22/`
  - Naive censored baseline at RMST h=3.0 and survival q90.
