# ACTG175 Data

This folder contains the canonical ACTG175 inputs used by the HIV real-data
analysis.

- `actg175_ucimlrepo_cache.csv`: cached full ACTG175 table fetched from
  `ucimlrepo`.
- `actg175_paper_subset_filtered.csv`: ddI versus ZDV+ddI analysis subset used
  for the RMST at 1,000 days comparison.
- `actg175_feature_splits.csv`: covariate-routing specifications for the
  all-X, clinical split, Rec A, Rec B, and R-CSF analyses.
- `actg175_metadata.json`: dataset dimensions, treatment coding, target,
  horizon, covariates, hashes, and experiment defaults.

This directory is for data and metadata only. Result tables, age profiles,
bootstrap draws, and figures should stay under `outputs/`.
