#!/usr/bin/env Rscript

if (!requireNamespace("grf", quietly = TRUE)) {
  stop(
    "R package 'grf' is required for the censored R causal_forest final stage. Install it with install.packages('grf').",
    call. = FALSE
  )
}

suppressPackageStartupMessages(library(grf))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop(
    paste(
      "Usage:",
      "run_grf_csf_final_forest.R train <train_csv> <feature_cols_csv> <num_trees> <min_node_size> <model_rds> [seed]",
      "or",
      "run_grf_csf_final_forest.R predict <model_rds> <predict_csv> <feature_cols_csv> <output_csv>",
      call. = FALSE
    )
  )
}

mode <- args[[1]]

if (identical(mode, "train")) {
  if (length(args) < 6 || length(args) > 7) {
    stop(
      "Usage: run_grf_csf_final_forest.R train <train_csv> <feature_cols_csv> <num_trees> <min_node_size> <model_rds> [seed]",
      call. = FALSE
    )
  }

  train_csv <- args[[2]]
  feature_cols <- strsplit(args[[3]], ",", fixed = TRUE)[[1]]
  num_trees <- as.integer(args[[4]])
  min_node_size <- as.integer(args[[5]])
  model_rds <- args[[6]]
  seed <- if (length(args) >= 7) as.integer(args[[7]]) else 42L

  obs <- read.csv(train_csv, check.names = FALSE)
  required_cols <- c("Y", "A", feature_cols)
  missing_cols <- setdiff(required_cols, names(obs))
  if (length(missing_cols) > 0) {
    stop(
      sprintf("Missing columns in training CSV: %s", paste(missing_cols, collapse = ", ")),
      call. = FALSE
    )
  }

  X <- as.matrix(obs[, feature_cols, drop = FALSE])
  Y <- obs[["Y"]]
  W <- obs[["A"]]

  fit <- causal_forest(
    X = X,
    Y = Y,
    W = W,
    num.trees = num_trees,
    min.node.size = min_node_size,
    num.threads = 1,
    seed = seed
  )

  saveRDS(fit, model_rds)
} else if (identical(mode, "predict")) {
  if (length(args) != 5) {
    stop(
      "Usage: run_grf_csf_final_forest.R predict <model_rds> <predict_csv> <feature_cols_csv> <output_csv>",
      call. = FALSE
    )
  }

  model_rds <- args[[2]]
  predict_csv <- args[[3]]
  feature_cols <- strsplit(args[[4]], ",", fixed = TRUE)[[1]]
  output_csv <- args[[5]]

  fit <- readRDS(model_rds)
  obs <- read.csv(predict_csv, check.names = FALSE)
  missing_cols <- setdiff(feature_cols, names(obs))
  if (length(missing_cols) > 0) {
    stop(
      sprintf("Missing columns in prediction CSV: %s", paste(missing_cols, collapse = ", ")),
      call. = FALSE
    )
  }

  X <- as.matrix(obs[, feature_cols, drop = FALSE])
  pred <- predict(fit, X)$predictions
  write.csv(data.frame(prediction = pred), output_csv, row.names = FALSE)
} else {
  stop("mode must be one of: train, predict", call. = FALSE)
}
