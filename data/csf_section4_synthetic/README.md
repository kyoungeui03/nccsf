# CSF Section 4 Synthetic Benchmark

This folder records the canonical data-generating setup for the external
replication of the original causal survival forest synthetic benchmark from
Cui et al. (2023), Section 4.

The benchmark is different from the main NC-CSF synthetic benchmark:

- there is no latent confounder `U`;
- there are no designated negative-control proxies `W` and `Z`;
- all confounding is through observed covariates `X`;
- the purpose is to check that the Final Model remains competitive with direct
  R-CSF in a standard fully observed censored causal benchmark.

The actual train/test samples are generated on the fly by
`Archive/paper_csf_dgp.py` and `Archive/run_paper_final_model_simulation.py`.
This avoids storing large replication-level CSV files while preserving the
paper-exact horizons, settings, sample sizes, seeds, and hyperparameters in
`metadata.json` and `settings.json`.
