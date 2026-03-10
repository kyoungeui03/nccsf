# csf_grf_new

Minimal local Python + C++ extraction of the causal survival forest pipeline.

This clean copy keeps only the code and assets needed to:

- build the local native backend
- verify parity against the bundled GRF master reference
- generate standardized synthetic scenarios
- run the local 8-variant benchmark

What is intentionally excluded:

- all generated outputs
- cached files and notebooks
- ranking/render-only scripts
- legacy benchmark wrappers that are not needed for the main pipeline

Top-level layout:

- `core/`: copied GRF C++ core source and vendored headers
- `native/`: C API wrapper and compiled local library
- `python-package/grf/`: Python package for backends, methods, nuisances, and synthetic generators
- `scripts/build_native.py`: rebuild the native library
- `scripts/compare_with_reference.py`: compare local baseline output to bundled GRF reference predictions
- `scripts/materialize_synthetic_datasets.py`: emit the standardized synthetic scenario catalog
- `scripts/run_8variant_benchmark.py`: run the local 8-variant benchmark
- `scripts/run_12case_8variant_benchmark.py`: run the finalized 8-variant benchmark across the 12 notebook synthetic cases
- `scripts/run_c3_econml_mild_shrink.py`: run the finalized best C3 on any user-supplied survival dataset
- `scripts/run_grf_csf_baseline.R`: helper for installed R `grf` B1/B2 baselines
- `data/reference_input.csv`: bundled reference dataset
- `data/reference_grf_master_predictions.csv`: bundled GRF master reference predictions

Quick start:

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
PYTHONPATH=python-package python3 scripts/compare_with_reference.py
```

Rebuild native library:

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
python3 scripts/build_native.py --force
```

Run the local 8-variant benchmark:

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
PYTHONPATH=python-package python3 scripts/run_8variant_benchmark.py
```

Run the finalized 12-case 8-variant benchmark suite:

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
PYTHONPATH=python-package python3 scripts/run_12case_8variant_benchmark.py
```

Run the finalized best C3 on a custom dataset:

```bash
cd /Users/kyoungeuihong/Desktop/csf_grf_new
PYTHONPATH=python-package python3 scripts/run_c3_econml_mild_shrink.py \
  --train-csv /path/to/train.csv \
  --x-cols X0,X1,X2,X3,X4 \
  --w-cols W \
  --z-cols Z \
  --treatment-col A \
  --time-col time \
  --event-col event \
  --output-dir outputs/c3_custom_run
```
