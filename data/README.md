# Data Layout

This directory stores canonical datasets and dataset metadata. Model outputs,
figures, bootstrap draws, and benchmark summaries should stay under `outputs/`.

## `synthetic_scenarios/`

Main NC-CSF synthetic datasets used for the proximal benchmark. Each scenario
contains:

- `observed.csv`
- `truth.csv`
- `metadata.json`

The scenario catalog is stored in `catalog.csv` and `catalog.json`.

## `csf_section4_synthetic/`

Metadata for the external replication of the Cui et al. causal survival forest
Section 4 synthetic benchmark. The train/test samples are generated on the fly
from deterministic seeds, so this folder stores the DGP settings, horizons,
sample sizes, and hyperparameters rather than all replication-level CSV files.

## `actg175/`

Canonical inputs for the HIV ACTG175 real-data analysis:

- `actg175_ucimlrepo_cache.csv`
- `actg175_paper_subset_filtered.csv`
- `actg175_feature_splits.csv`
- `actg175_metadata.json`

## `rhc/`

Canonical inputs for the RHC real-data analysis:

- `raw_rhc.csv`
- `cleaned_rhc.csv`
- `rhc_30day_survival.csv`
- `rhc_30day_survival_metadata.json`
