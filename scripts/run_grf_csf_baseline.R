#!/usr/bin/env Rscript

# R-grf causal_survival_forest baseline for DUET.
#
# Proxy-blind: the caller stacks X + W + Z into one feature matrix, so grf sees
# every observed block as an ordinary covariate (no proximal distinction).
#
# Outputs (all needed by the DUET baseline pipeline to mirror NC-CSF artifacts):
#   <output_csv>            : per-patient `prediction` (RMST tau) + `variance`
#                             (grf infinitesimal-jackknife variance of tau-hat).
#   <output_csv>_ate.csv    : one row `estimate`,`std.err` from grf's doubly
#                             robust average_treatment_effect (AIPW ATE + SE).
#
# Native inference: we use grf's own IJ per-patient variance and AIPW ATE SE
# rather than a case bootstrap, matching the inference choice made for the
# EconML No-PCI baseline (its native Bootstrap-of-Little-Bags variance).

if (!requireNamespace("grf", quietly = TRUE)) {
  stop(
    "R package 'grf' is required for this causal_survival_forest baseline. Install it with install.packages('grf').",
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
if (!(length(args) %in% c(5, 6, 7, 8))) {
  stop(
    paste0(
      "Usage: run_grf_csf_baseline.R <input_csv> <feature_cols_csv> <horizon> ",
      "<num_trees> <output_csv> [target] [seed] [min_node_size]"
    ),
    call. = FALSE
  )
}

input_path <- args[[1]]
feature_cols <- strsplit(args[[2]], ",", fixed = TRUE)[[1]]
horizon <- as.numeric(args[[3]])
num_trees <- as.integer(args[[4]])
output_path <- args[[5]]
target <- if (length(args) >= 6) args[[6]] else "RMST"
seed <- if (length(args) >= 7) as.integer(args[[7]]) else 42
# min.node.size: matched to the NC-CSF / EconML final forest min_samples_leaf
# (20) so the R-grf forest is architecture-equal. If not supplied, fall back to
# grf's own default by leaving it unset.
min_node_size <- if (length(args) >= 8) as.integer(args[[8]]) else NA_integer_

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

forest_args <- list(
  X = X,
  Y = Y,
  W = A,
  D = delta,
  target = target,
  horizon = horizon,
  num.trees = num_trees,
  num.threads = 1,
  seed = seed
)
if (!is.na(min_node_size)) {
  forest_args[["min.node.size"]] <- min_node_size
}

forest <- do.call(causal_survival_forest, forest_args)

# Per-patient point estimate + infinitesimal-jackknife variance (native grf
# inference). estimate.variance=TRUE asks grf for the IJ variance of tau-hat(x).
pred <- predict(forest, X, estimate.variance = TRUE)
preds <- pred$predictions
variance <- pred$variance.estimates
if (is.null(variance)) {
  stop("grf did not return variance.estimates; check grf version (>= 2.0).", call. = FALSE)
}

out_df <- data.frame(prediction = preds, variance = variance)
write.csv(out_df, output_path, row.names = FALSE)

# Doubly robust average treatment effect + its standard error (native grf AIPW).
ate <- average_treatment_effect(forest)
ate_estimate <- unname(ate["estimate"])
ate_std_err <- unname(ate["std.err"])
ate_path <- sub("\\.csv$", "_ate.csv", output_path)
if (identical(ate_path, output_path)) {
  ate_path <- paste0(output_path, "_ate.csv")
}
ate_df <- data.frame(
  estimate = ate_estimate,
  std.err = ate_std_err,
  target = target,
  horizon = horizon,
  n = nrow(obs_df)
)
write.csv(ate_df, ate_path, row.names = FALSE)
