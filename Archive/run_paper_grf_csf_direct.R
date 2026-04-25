#!/usr/bin/env Rscript

if (!requireNamespace("grf", quietly = TRUE)) {
  stop(
    "R package 'grf' is required for the archive R-CSF experiment. Install it with install.packages('grf').",
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
if (length(args) < 7 || length(args) > 8) {
  stop(
    paste(
      "Usage:",
      "run_paper_grf_csf_direct.R <train_csv> <predict_csv> <feature_cols_csv> <target> <horizon> <num_trees> <output_csv> [seed]"
    ),
    call. = FALSE
  )
}

train_csv <- args[[1]]
predict_csv <- args[[2]]
feature_cols <- strsplit(args[[3]], ",", fixed = TRUE)[[1]]
target <- args[[4]]
horizon <- as.numeric(args[[5]])
num_trees <- as.integer(args[[6]])
output_csv <- args[[7]]
seed <- if (length(args) >= 8) as.integer(args[[8]]) else 42L

train_df <- read.csv(train_csv, check.names = FALSE)
predict_df <- read.csv(predict_csv, check.names = FALSE)

train_required <- c("time", "event", "A", feature_cols)
missing_train <- setdiff(train_required, names(train_df))
if (length(missing_train) > 0) {
  stop(
    sprintf("Missing training columns: %s", paste(missing_train, collapse = ", ")),
    call. = FALSE
  )
}

missing_predict <- setdiff(feature_cols, names(predict_df))
if (length(missing_predict) > 0) {
  stop(
    sprintf("Missing prediction columns: %s", paste(missing_predict, collapse = ", ")),
    call. = FALSE
  )
}

X_train <- as.matrix(train_df[, feature_cols, drop = FALSE])
Y_train <- train_df$time
W_train <- train_df$A
D_train <- train_df$event
X_predict <- as.matrix(predict_df[, feature_cols, drop = FALSE])

fit <- causal_survival_forest(
  X = X_train,
  Y = Y_train,
  W = W_train,
  D = D_train,
  target = target,
  horizon = horizon,
  num.trees = num_trees,
  min.node.size = 5,
  sample.fraction = 0.5,
  mtry = min(ncol(X_train), ceiling(sqrt(ncol(X_train))) + 20),
  num.threads = 1,
  seed = seed
)

pred <- predict(fit, X_predict)$predictions
write.csv(data.frame(prediction = pred), output_csv, row.names = FALSE)
