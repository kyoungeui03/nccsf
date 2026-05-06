# CSF Section 4 Paper Replication

This folder is dedicated to reproducing the Section 4 synthetic simulation
benchmark from prior work (2023), the causal survival forest paper.

## Files

- `paper_csf_dgp.py`: paper-style CSF synthetic settings 1--4, including the
  RMST horizons and survival-probability horizons used in the paper.
- `run_paper_final_model_simulation.py`: main runner for comparing our Final
  Conditional Model, no-PCI/raw variants, and direct R-CSF on the paper-style
  DGP.
- `run_paper_grf_csf_direct.R`: direct R `grf::causal_survival_forest` backend
  used by the paper replication runner.
- `_direct_grf_imports.py`: import bootstrapper that pins Archive experiments to
  exact source modules under `python-package/grf`.

## Canonical Example

```bash
cd <REPO_ROOT>
module load r/4.4.2  # on the cluster, if R is module-managed

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
  --output-dir outputs/final_paper_results/csf_section4_replication
```

The paper-exact horizons are:

- RMST: setting 1 = 1.5, setting 2 = 2.0, setting 3 = 15.0, setting 4 = 3.0.
- Survival probability: setting 1 = 0.8, setting 2 = 1.2, setting 3 = 10.0,
  setting 4 = 2.0.
