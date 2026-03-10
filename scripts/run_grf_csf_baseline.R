#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(grf))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop(
    "Usage: run_grf_csf_baseline.R <input_csv> <feature_cols_csv> <horizon> <num_trees> <output_csv>",
    call. = FALSE
  )
}

input_path <- args[[1]]
feature_cols <- strsplit(args[[2]], ",", fixed = TRUE)[[1]]
horizon <- as.numeric(args[[3]])
num_trees <- as.integer(args[[4]])
output_path <- args[[5]]

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
  target = "RMST",
  horizon = horizon,
  num.trees = num_trees,
  num.threads = 1,
  seed = 42
)

preds <- predict(forest, X)$predictions
write.csv(data.frame(prediction = preds), output_path, row.names = FALSE)
