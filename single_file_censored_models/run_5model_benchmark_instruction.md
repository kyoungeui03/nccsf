# run_5model_benchmark.py Guide

This folder contains a censored synthetic benchmark runner:

- run_5model_benchmark.py

It evaluates 5 methods on a grid of:

- 12 case templates (CASE_SPECS)
- 22 settings (S01-S22)
- target = RMST and/or survival.probability

Total model fits for a full run:

- 12 x 22 x number_of_targets x 5

If target is both, that is:

- 12 x 22 x 2 x 5 = 2640 model rows

## Methods Evaluated

The script runs these 5 methods per case-setting-target:

1. Final Model
2. Final Model Oracle
3. Strict Baseline
4. Strict Baseline Oracle
5. R-CSF Baseline

## Prerequisites

From repository root, install dependencies:

```bash
python -m pip install -r requirements.txt
```

R-CSF baseline also requires:

- R installation with Rscript available
- R package grf

If Rscript is not on PATH, set RSCRIPT environment variable.

## How To Run

Run from repository root:

```bash
python single_file_censored_models/run_5model_benchmark.py
```

This runs all cases, all settings, and both targets.

## Useful Flags

- --output-dir
  - Output folder path.
  - Default: outputs/single_file_censored_models_5model

- --case-ids
  - Run only selected case ids.
  - Example: --case-ids 1 2 3

- --case-slugs
  - Run only selected case slugs.

- --setting-ids
  - Run only selected settings.
  - Example: --setting-ids S01 S10 S22

- --target
  - Choices: RMST, survival.probability, both
  - Default: both

- --horizon-quantile
  - Horizon quantile used in target preparation.
  - Default: 0.60

- --random-state
  - Random seed.
  - Default: 42

- --num-trees-baseline
  - Trees for R-CSF baseline.
  - Default: 200

- --list-cases
  - Print available cases and exit.

- --list-settings
  - Print available settings and exit.

## Common Commands

List available cases:

```bash
python single_file_censored_models/run_5model_benchmark.py --list-cases
```

List available settings:

```bash
python single_file_censored_models/run_5model_benchmark.py --list-settings
```

Quick smoke run (small subset):

```bash
python single_file_censored_models/run_5model_benchmark.py \
  --case-ids 1 2 \
  --setting-ids S01 S02 \
  --target RMST \
  --output-dir outputs/smoke_run_5model
```

## Output Files

In the selected output dir, you will see:

- results_incremental.csv
  - Checkpoint file updated during run.

- results_full.csv
  - Full combined results table.

- run_metadata.json
  - Metadata used to validate resume compatibility.

- Per case-setting-target files:
  - case_XX_<case_slug>_<setting_id>_<target>.csv
  - case_XX_<case_slug>_<setting_id>_<target>.png

- Aggregated summaries per target:
  - basic12_RMST_summary.csv
  - basic12_RMST_top5.csv
  - basic12_survival_probability_summary.csv
  - basic12_survival_probability_top5.csv
  - and corresponding PNG tables

Note:

- summary and top5 are identical when only 5 models are present.

## Tips

- Start with a subset using --case-ids and --setting-ids.
- Use a dedicated output dir per experiment.
- Keep the same output dir when you want resume behavior.
