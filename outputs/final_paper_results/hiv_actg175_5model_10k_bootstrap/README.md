# HIV ACTG175 5-Model 10k Bootstrap Final Outputs

Canonical folder for the final HIV ACTG175 comparison.

## Models

- `final_conditional_x14`: Final Model, all 14 covariates in X.
- `final_conditional_x12_w2_z2`: Final Model, X/W/Z = 12/2/2 clinical split.
- `final_conditional_rec_a`: Final Model Rec A.
- `final_conditional_rec_b`: Final Model Rec B.
- `r_csf_x14`: direct R-CSF baseline, all 14 covariates in X.

## Shared Setup

- Dataset: AIDS Clinical Trials Group Study 175
- Subset: ddI vs ZDV+ddI
- N: 1083 (561 control, 522 treated)
- Target: RMST at h = 1000.0 days
- Bootstrap repetitions: 200
- Sample seed: 42
- Table-4 reference subject IDs: 92, 95, 101, 217, 466, 472, 704, 753, 832, 926

## Hyperparameter Note

The four Final-model variants are collected from the `hiv_actg175_split_comparison_10k_bootstrap` run. The folder name identifies the 10k run, and the output metadata confirms the common ACTG175 setup and bootstrap procedure, but it does not store the Final `--num-trees` argument explicitly. The R-CSF fresh rerun records `--grf-num-trees 10000` in `shared/r_csf_wrapper_run_config.json`.

## Layout

- `models/`: complete per-model outputs.
- `shared/`: shared cohort metadata, feature splits, reference subjects, and R-CSF wrapper config.
- `combined/`: five-model combined CSV files for direct analysis.
- `MANIFEST.json`: machine-readable manifest.
