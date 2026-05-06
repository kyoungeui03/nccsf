## What It Plots

- X-axis: `setting_id` (S01-S22)
- Y-axis: selected metric (`rmse` or `mae`)
- Grouping: 5 model boxplots per setting
- Target subset: `RMST` or `survival.probability`

## Default Input and Output

If you run with no extra flags, the script uses:

- Input CSV:
  - `outputs/censored_5model/results_full.csv`
- Output PNG:
  - `outputs/censored_5model/grouped_rmse_boxplot_RMST.png`

## Run Commands

From repository root:

```bash
python censored/plot_grouped_boxplot.py
```

From inside `censored` folder:

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
python censored/plot_grouped_boxplot.py --target RMST --metric rmse
```

Survival probability + MAE:

```bash
python censored/plot_grouped_boxplot.py --target survival.probability --metric mae
```

Custom output filename:

```bash
python censored/plot_grouped_boxplot.py \
  --target survival.probability \
  --metric rmse \
  --output-png outputs/censored_5model/grouped_rmse_boxplot_survival_probability_custom.png
```

## Typical Workflow

1. Run benchmark first:
   - `python censored/run_5model_benchmark.py`
2. Then plot:
   - `python censored/plot_grouped_boxplot.py --target RMST --metric rmse`

If `results_full.csv` is missing, run the benchmark script first or pass `--input-csv` to a valid results file.
