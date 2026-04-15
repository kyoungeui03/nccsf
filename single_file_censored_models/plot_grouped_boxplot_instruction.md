## What It Plots

- X-axis: `setting_id` (S01-S22)
- Y-axis: selected metric (`rmse` or `mae`)
- Grouping: 5 model boxplots per setting
- Target subset: `RMST` or `survival.probability`

## Default Input and Output

If you run with no extra flags, the script uses:

- Input CSV:
  - `outputs/single_file_censored_models_5model/results_full.csv`
- Output PNG:
  - `outputs/single_file_censored_models_5model/grouped_rmse_boxplot_RMST.png`

## Run Commands

From repository root:

```bash
python single_file_censored_models/plot_grouped_boxplot.py
```

From inside `single_file_censored_models` folder:

```bash
python plot_grouped_boxplot.py
```

## Required/Important Flags

- `--target`
  - Choices: `RMST`, `survival.probability`

- `--metric`
  - Choices: `rmse`, `mae`

## Optional Flags

- `--input-csv <path>`
  - Custom path to `results_full.csv`

- `--output-png <path>`
  - Custom output image path

- `--figsize W H`
  - Figure size (default `18 8`)

- `--hide-fliers`
  - Hide outlier dots in boxplots

## Examples

RMST + RMSE:

```bash
python single_file_censored_models/plot_grouped_boxplot.py --target RMST --metric rmse
```

Survival probability + MAE:

```bash
python single_file_censored_models/plot_grouped_boxplot.py --target survival.probability --metric mae
```

Custom output filename:

```bash
python single_file_censored_models/plot_grouped_boxplot.py \
  --target survival.probability \
  --metric rmse \
  --output-png outputs/single_file_censored_models_5model/grouped_rmse_boxplot_survival_probability_custom.png
```

## Typical Workflow

1. Run benchmark first:
   - `python single_file_censored_models/run_5model_benchmark.py`
2. Then plot:
   - `python single_file_censored_models/plot_grouped_boxplot.py --target RMST --metric rmse`

If `results_full.csv` is missing, run the benchmark script first or pass `--input-csv` to a valid results file.
