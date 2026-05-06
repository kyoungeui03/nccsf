"""Thin Python wrapper for the direct R ``grf::causal_forest`` baseline."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.r_runtime import resolve_rscript  # noqa: E402


def _ensure_2d(array):
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


class GRFNoncensoredBaseline:
    """Run direct R-CSF-style causal forests on raw ``[X,W,Z]`` features.

    The bundled R script fits and predicts on the supplied sample. This wrapper
    is intended for benchmark parity with the existing non-censored runners.
    """

    def __init__(self, *, num_trees: int = 200, seed: int = 42):
        self.num_trees = int(num_trees)
        self.seed = int(seed)
        self.predictions_ = None
        self.elapsed_ = None

    def fit_components(self, X, A, Y, Z, W):
        x = _ensure_2d(X)
        w = _ensure_2d(W)
        z = _ensure_2d(Z)
        feature_cols = [f"X{j}" for j in range(x.shape[1])]
        feature_cols += [f"W{j}" for j in range(w.shape[1])]
        feature_cols += [f"Z{j}" for j in range(z.shape[1])]
        obs = pd.DataFrame(np.hstack([x, w, z]), columns=feature_cols)
        obs["A"] = np.asarray(A, dtype=float).ravel()
        obs["Y"] = np.asarray(Y, dtype=float).ravel()

        script = PROJECT_ROOT / "scripts" / "run_grf_cf_baseline.R"
        with tempfile.TemporaryDirectory(prefix="grf_nc_baseline_") as tmp_dir:
            tmp = Path(tmp_dir)
            input_csv = tmp / "input.csv"
            output_csv = tmp / "predictions.csv"
            obs.to_csv(input_csv, index=False)
            cmd = [
                resolve_rscript(),
                str(script),
                str(input_csv),
                ",".join(feature_cols),
                str(self.num_trees),
                str(output_csv),
                str(self.seed),
            ]
            start = time.time()
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
            self.elapsed_ = time.time() - start
            if proc.returncode != 0:
                raise RuntimeError(
                    "R grf non-censored baseline failed.\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                )
            self.predictions_ = pd.read_csv(output_csv)["pred"].to_numpy(dtype=float)
        return self

    def effect_from_components(self, X, W, Z):
        if self.predictions_ is None:
            raise RuntimeError("Model must be fit before calling effect_from_components.")
        n = _ensure_2d(X).shape[0]
        if n != len(self.predictions_):
            raise ValueError(
                "The direct R baseline wrapper only returns predictions on the fitted sample. "
                f"Requested {n} rows, fitted {len(self.predictions_)} rows."
            )
        return np.asarray(self.predictions_, dtype=float)


RCFNoncensoredBaseline = GRFNoncensoredBaseline


__all__ = ["GRFNoncensoredBaseline", "RCFNoncensoredBaseline"]
