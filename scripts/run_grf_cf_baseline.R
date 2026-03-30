#!/usr/bin/env Rscript

if (!requireNamespace("grf", quietly = TRUE)) {
  stop(
    "R package 'grf' is required for the R-CF Baseline. Install it with install.packages('grf').",
    call. = FALSE
  )
}

suppressPackageStartupMessages(library(grf))

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop(
    paste(
      "Usage: run_grf_cf_baseline.R <input_csv> <feature_cols_csv> <num_trees> <output_csv> [seed]",
      call. = FALSE
    )
  )
}

input_csv <- args[[1]]
feature_cols <- strsplit(args[[2]], ",", fixed = TRUE)[[1]]
num_trees <- as.integer(args[[3]])
output_csv <- args[[4]]
seed <- if (length(args) >= 5) as.integer(args[[5]]) else 42L

obs <- read.csv(input_csv, check.names = FALSE)
X <- as.matrix(obs[, feature_cols, drop = FALSE])
Y <- obs[["Y"]]
W <- obs[["A"]]

fit <- causal_forest(
  X = X,
  Y = Y,
  W = W,
  num.trees = num_trees,
  seed = seed,
  num.threads = 1
)

pred <- predict(fit)$predictions
write.csv(data.frame(pred = pred), output_csv, row.names = FALSE)
