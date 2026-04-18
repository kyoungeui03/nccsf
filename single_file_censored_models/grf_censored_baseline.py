"""Single-file wrapper around installed R grf::causal_survival_forest.

This module keeps the same public API as the other standalone censored models:

    model.fit_components(X, A, time, event, Z, W)
    tau_hat = model.effect_from_components(X, W, Z)

It intentionally avoids repo-local helper imports. The only non-stdlib
dependency is NumPy; the actual forest fitting is delegated to an installed
R package `grf` via a tiny temporary R script written at runtime.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


R_RUNNER_SOURCE = r"""#!/usr/bin/env Rscript
if (!requireNamespace("grf", quietly = TRUE)) {
  stop(
    "R package 'grf' is required for the GRF censored baseline. Install it with install.packages('grf').",
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
if (length(args) < 1) {
  stop(
    paste(
      "Usage:",
      "grf_censored_baseline_runner.R train <train_csv> <feature_cols_csv> <target> <horizon> <num_trees> <min_node_size> <model_rds> [seed]",
      "or",
      "grf_censored_baseline_runner.R predict <model_rds> <predict_csv> <feature_cols_csv> <output_csv>",
      "or",
      "grf_censored_baseline_runner.R scores <model_rds> <output_csv>",
      call. = FALSE
    )
  )
}

mode <- args[[1]]

if (identical(mode, "train")) {
  if (length(args) < 8 || length(args) > 9) {
    stop(
      "Usage: grf_censored_baseline_runner.R train <train_csv> <feature_cols_csv> <target> <horizon> <num_trees> <min_node_size> <model_rds> [seed]",
      call. = FALSE
    )
  }

  train_csv <- args[[2]]
  feature_cols <- strsplit(args[[3]], ",", fixed = TRUE)[[1]]
  target <- args[[4]]
  horizon <- as.numeric(args[[5]])
  num_trees <- as.integer(args[[6]])
  min_node_size <- as.integer(args[[7]])
  model_rds <- args[[8]]
  seed <- if (length(args) >= 9) as.integer(args[[9]]) else 42L

  obs <- read.csv(train_csv, check.names = FALSE)
  required_cols <- c("time", "event", "A", feature_cols)
  missing_cols <- setdiff(required_cols, names(obs))
  if (length(missing_cols) > 0) {
    stop(
      sprintf("Missing columns in training CSV: %s", paste(missing_cols, collapse = ", ")),
      call. = FALSE
    )
  }

  X <- as.matrix(obs[, feature_cols, drop = FALSE])
  Y <- obs[["time"]]
  W <- obs[["A"]]
  D <- obs[["event"]]

  fit <- causal_survival_forest(
    X = X,
    Y = Y,
    W = W,
    D = D,
    target = target,
    horizon = horizon,
    num.trees = num_trees,
    min.node.size = min_node_size,
    num.threads = 1,
    seed = seed
  )

  saveRDS(fit, model_rds)
} else if (identical(mode, "predict")) {
  if (length(args) != 5) {
    stop(
      "Usage: grf_censored_baseline_runner.R predict <model_rds> <predict_csv> <feature_cols_csv> <output_csv>",
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
  pred <- predict(fit, X)[["predictions"]]
  write.csv(data.frame(prediction = pred), output_csv, row.names = FALSE)
} else if (identical(mode, "scores")) {
  if (length(args) != 3) {
    stop(
      "Usage: grf_censored_baseline_runner.R scores <model_rds> <output_csv>",
      call. = FALSE
    )
  }

  model_rds <- args[[2]]
  output_csv <- args[[3]]

  fit <- readRDS(model_rds)
  scores <- get_scores(fit)
  write.csv(data.frame(score = scores), output_csv, row.names = FALSE)
} else {
  stop("mode must be one of: train, predict, scores", call. = FALSE)
}
"""


def _ensure_2d(array):
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _resolve_rscript():
    candidates = [
        os.environ.get("RSCRIPT"),
        shutil.which("Rscript"),
        "/usr/local/bin/Rscript",
        "/opt/homebrew/bin/Rscript",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError("Could not find Rscript. Set RSCRIPT or install R.")


def _resolve_horizon(target, horizon, time):
    if target not in {"RMST", "survival.probability"}:
        raise ValueError("target must be one of {'RMST', 'survival.probability'}.")
    if target == "survival.probability":
        if horizon is None:
            raise ValueError("horizon is required when target='survival.probability'.")
        return float(horizon)
    if horizon is None:
        return float(np.max(np.asarray(time, dtype=float)))
    return float(horizon)


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _read_prediction_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return np.asarray([float(row["prediction"]) for row in reader], dtype=float)


def _read_score_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return np.asarray([float(row["score"]) for row in reader], dtype=float)


class GRFCensoredBaseline:
    """Standalone Python wrapper for installed R grf::causal_survival_forest."""

    def __init__(
        self,
        *,
        target="RMST",
        horizon=None,
        n_estimators=200,
        min_samples_leaf=20,
        random_state=42,
    ):
        self._target = target
        self._horizon = horizon
        self._n_estimators = int(n_estimators)
        self._min_samples_leaf = int(min_samples_leaf)
        self._random_state = int(random_state)
        self._tmpdir = None
        self._runner_path = None
        self._model_path = None
        self._feature_cols = None
        self._fitted_horizon = None

    @staticmethod
    def stack_final_features(*arrays):
        parts = [_ensure_2d(np.asarray(arr, dtype=float)) for arr in arrays]
        return np.hstack(parts)

    def _ensure_workspace(self):
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="grf_censored_baseline_")
            tmp_path = Path(self._tmpdir.name)
            self._runner_path = tmp_path / "grf_censored_baseline_runner.R"
            self._runner_path.write_text(R_RUNNER_SOURCE, encoding="utf-8")
            self._model_path = tmp_path / "grf_censored_model.rds"
        return Path(self._tmpdir.name)

    def _call_r(self, args):
        proc = subprocess.run(
            [_resolve_rscript(), str(self._runner_path), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Installed R grf baseline failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

    def fit_components(self, X, A, time, event, Z, W):
        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_full = self.stack_final_features(x, raw_w, raw_z)
        a = np.asarray(A, dtype=float).ravel()
        y_time = np.asarray(time, dtype=float).ravel()
        delta = np.asarray(event, dtype=float).ravel()

        if len(x_full) != len(a) or len(a) != len(y_time) or len(y_time) != len(delta):
            raise ValueError("X, A, time, and event must have matching lengths.")

        workspace = self._ensure_workspace()
        self._feature_cols = [f"f{j}" for j in range(x_full.shape[1])]
        self._fitted_horizon = _resolve_horizon(self._target, self._horizon, y_time)

        train_csv = workspace / "train.csv"
        header = ["time", "event", "A", *self._feature_cols]
        rows = [
            [float(y_time[i]), float(delta[i]), float(a[i]), *map(float, x_full[i])]
            for i in range(len(x_full))
        ]
        _write_csv(train_csv, header, rows)

        self._call_r(
            [
                "train",
                str(train_csv),
                ",".join(self._feature_cols),
                str(self._target),
                str(self._fitted_horizon),
                str(self._n_estimators),
                str(self._min_samples_leaf),
                str(self._model_path),
                str(self._random_state),
            ]
        )
        return self

    def effect_from_components(self, X, W, Z):
        if self._tmpdir is None or self._model_path is None or self._feature_cols is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")

        x = _ensure_2d(X).astype(float)
        raw_w = _ensure_2d(W).astype(float)
        raw_z = _ensure_2d(Z).astype(float)
        x_full = self.stack_final_features(x, raw_w, raw_z)

        if x_full.shape[1] != len(self._feature_cols):
            raise ValueError("Prediction features do not match the fitted feature dimension.")

        workspace = Path(self._tmpdir.name)
        predict_csv = workspace / "predict.csv"
        output_csv = workspace / "predictions.csv"
        rows = [[*map(float, row)] for row in x_full]
        _write_csv(predict_csv, self._feature_cols, rows)

        self._call_r(
            [
                "predict",
                str(self._model_path),
                str(predict_csv),
                ",".join(self._feature_cols),
                str(output_csv),
            ]
        )
        return _read_prediction_csv(output_csv)

    def dr_scores_training(self):
        """Return grf::get_scores(fit) for the fitted training sample."""

        if self._tmpdir is None or self._model_path is None:
            raise RuntimeError("Model must be fit before calling dr_scores_training.")

        workspace = Path(self._tmpdir.name)
        output_csv = workspace / "scores.csv"
        self._call_r(["scores", str(self._model_path), str(output_csv)])
        return _read_score_csv(output_csv)

    def cleanup(self):
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
            self._runner_path = None
            self._model_path = None
            self._feature_cols = None
            self._fitted_horizon = None

    def __del__(self):
        self.cleanup()


RGRFCensoredBaseline = GRFCensoredBaseline


__all__ = ["GRFCensoredBaseline", "RGRFCensoredBaseline"]
