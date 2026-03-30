#!/usr/bin/env Rscript

if (!requireNamespace("grf", quietly = TRUE)) {
  stop(
    "R package 'grf' is required for the R-CSF Baseline. Install it with install.packages('grf').",
    call. = FALSE
  )
}

suppressPackageStartupMessages(library(grf))

if (!("causal_survival_forest" %in% getNamespaceExports("grf"))) {
  stop(
    "Your installed R package 'grf' does not export causal_survival_forest(). Install a recent version of grf.",
    call. = FALSE
  )
}

args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(5, 6))) {
  stop(
    "Usage: run_grf_csf_baseline.R <input_csv> <feature_cols_csv> <horizon> <num_trees> <output_csv> [target]",
    call. = FALSE
  )
}

input_path <- args[[1]]
feature_cols <- strsplit(args[[2]], ",", fixed = TRUE)[[1]]
horizon <- as.numeric(args[[3]])
num_trees <- as.integer(args[[4]])
output_path <- args[[5]]
target <- if (length(args) >= 6) args[[6]] else "RMST"

obs_df <- read.csv(input_path, check.names = FALSE)

missing_cols <- setdiff(c("time", "event", "A", feature_cols), names(obs_df))
if (length(missing_cols) > 0) {
  stop(
    sprintf("Missing columns in input CSV: %s", paste(missing_cols, collapse = ", ")),
    call. = FALSE
  )
}

X <- as.matrix(obs_df[, feature_cols, drop = FALSE])
Y <- obs_df$time
A <- obs_df$A
delta <- obs_df$event

forest <- causal_survival_forest(
  X = X,
  Y = Y,
  W = A,
  D = delta,
  target = target,
  horizon = horizon,
  num.trees = num_trees,
  num.threads = 1,
  seed = 42
)

preds <- predict(forest, X)$predictions
write.csv(data.frame(prediction = preds), output_path, row.names = FALSE)
