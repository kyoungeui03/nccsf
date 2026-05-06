# run_5model_benchmark.py Guide

This runner evaluates selected censored benchmark models on the
single-file censored synthetic benchmark. By default, it runs the main
conditional-censoring 3-model suite.

Models:

1. `Final Conditional`
2. `Final Conditional Oracle`
3. `Censored Baseline`

Optional models exposed through `--models`:

- `naive_censored_baseline`
- `r_csf_baseline`
- `final_conditional_x_only`
- `final_conditional_broaddup_raw_only`
- `final_conditional_dup_full`
- `final_conditional_dup_raw_only`

Benchmark grid:

- 12 case templates (`CASE_SPECS`)
- 22 settings (`S01` to `S22`)
- target = `RMST` and/or `survival.probability`

Total rows for a full run with both targets:

- default: `12 x 22 x 2 x 3 = 1584`
- with all optional models: `12 x 22 x 2 x 9 = 4752`

## Prerequisites

From the repository root:

```bash
python -m pip install -r requirements.txt
```

No R dependency is required for this runner.

Optional GPU path:

- install `xgboost` from `requirements.txt`
- use `--gpu auto` or `--gpu xgboost`
- the GPU path accelerates the nuisance learners only
- `lifelines` Cox fitting and EconML's final causal forest remain CPU-based

## Main Command

Run from the repository root:

```bash
python censored/run_5model_benchmark.py
```

Default output directory:

```bash
outputs/censored_5model_conditional_suite
```

## Useful Flags

- `--output-dir`
  - Custom output folder.
- `--case-ids`
  - Example: `--case-ids 1 2 3`
- `--case-slugs`
  - Run selected case slugs instead of numeric ids.
- `--setting-ids`
  - Example: `--setting-ids S01 S14 S22`
- `--target`
  - Choices: `RMST`, `survival.probability`, `both`
- `--horizon-quantile`
  - Default: `0.90`
- `--models`
  - Select any subset of supported model keys.
- `--random-state`
  - Default: `42`
- `--num-trees`
  - Forest size used across Python models.
  - Default: `200`
- `--grf-num-trees`
  - Forest size for `r_csf_baseline`.
  - Default: `200`
- `--gpu`
  - Choices: `off`, `auto`, `xgboost`
  - `auto` tries to enable CUDA-backed XGBoost nuisances when available
  - `xgboost` forces the runner to request CUDA-backed XGBoost nuisances
  - `off` disables GPU-backed nuisances
- `--list-cases`
  - Print case list and exit.
- `--list-settings`
  - Print setting list and exit.

## Common Commands

List cases:

```bash
python censored/run_5model_benchmark.py --list-cases
```

List settings:

```bash
python censored/run_5model_benchmark.py --list-settings
```

Small smoke run:

```bash
python censored/run_5model_benchmark.py \
  --case-ids 1 \
  --setting-ids S01 \
  --target both \
  --output-dir outputs/smoke_conditional_suite
```

GPU smoke run:

```bash
python censored/run_5model_benchmark.py \
  --case-ids 1 \
  --setting-ids S01 \
  --target RMST \
  --gpu auto \
  --output-dir outputs/smoke_conditional_suite_gpu
```

## Save / Load / Resume

The runner checkpoints after each completed model row.

Files written into the output directory:

- `results_incremental.csv`
- `results_full.csv`
- `run_metadata.json`

Resume rule:

- Re-run the exact same command with the same `--output-dir`.
- Completed rows are loaded from `results_incremental.csv`.
- The runner skips finished rows and continues from the remaining ones.

If the metadata in `run_metadata.json` does not match the new command, the
runner raises an error instead of silently mixing incompatible experiments.

## Output Files

Inside the output directory:

- `results_incremental.csv`
  - checkpoint table updated row by row
- `results_full.csv`
  - latest full results table
- `run_metadata.json`
  - run signature used for resume validation
- `case_XX_<case_slug>_<setting_id>_<target>.csv`
  - one table per case-setting-target
- `case_XX_<case_slug>_<setting_id>_<target>.png`
  - rendered PNG version of the same table
- `basic12_conditional_suite_RMST_summary.csv`
- `basic12_conditional_suite_RMST_summary.png`
- `basic12_conditional_suite_survival_probability_summary.csv`
- `basic12_conditional_suite_survival_probability_summary.png`

## Server Notes

Recommended workflow on the server:

```bash
cd /path/to/csf_grf_new
python -m pip install -r requirements.txt
python censored/run_5model_benchmark.py \
  --target both \
  --gpu auto \
  --output-dir /path/to/output_dir
```

If the process is interrupted, run the same command again to resume.
