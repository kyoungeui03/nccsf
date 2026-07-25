"""Wrapper that runs the BART baseline in a separate interpreter and returns suite-shaped rows.

pymc + pymc-bart pull a heavy stack (pytensor, a compiler toolchain) that most users of this
repository do not want installed next to scikit-survival. BART therefore runs as a SUBPROCESS
in its own environment, and this module is the bridge: it writes the cohort and adjustment
spec, invokes ``benchmarks/bart.py`` with the chosen interpreter, and converts the returned
posterior summary into rows with the SAME columns ``suite.run_suite`` emits.

The E-value uses the SAME method-agnostic control-arm KM risk0 and the SAME evalue_multi as
every other baseline, so BART's number means what everyone else's means. Its interval is its
own posterior, not a bootstrap, and is labelled ``uq="posterior"`` for that reason.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ._stats import evalue_multi, km_risk

_HERE = Path(__file__).resolve().parent


def find_bart_python(explicit: str | None = None) -> str | None:
    """Interpreter that can ``import pymc_bart``, or None.

    Order: explicit argument, then $BART_PYTHON, then the current interpreter. Returning None
    rather than raising lets a caller run the other five baselines and simply report BART as
    unavailable -- a hard failure here would take the whole suite down over an optional extra.
    """
    for cand in (explicit, os.environ.get("BART_PYTHON"), sys.executable):
        if not cand:
            continue
        exe = shutil.which(cand) or cand
        if not Path(exe).exists():
            continue
        probe = subprocess.run([exe, "-c", "import pymc_bart"], capture_output=True)
        if probe.returncode == 0:
            return exe
    return None


def run_bart(df, horizon, *, x_cols, w_cols=(), z_cols=(), adjust="xwz", seed=0,
             draws=500, tune=500, trees=50, max_bins=16, subsample=2500,
             python=None, timeout=5400, verbose=True):
    """Run the BART baseline; return a DataFrame shaped like ``run_suite``'s output.

    Returns an EMPTY DataFrame (never raises) when no pymc-bart interpreter is available or
    the subprocess fails, so a missing optional dependency degrades to a missing row rather
    than a crashed comparison. The reason is printed and carried in ``.attrs["bart_error"]``.
    """
    out = pd.DataFrame()
    exe = find_bart_python(python)
    if exe is None:
        msg = ("no interpreter with pymc-bart found (tried $BART_PYTHON and the current "
               "interpreter); set BART_PYTHON=/path/to/python to enable the BART baseline")
        if verbose:
            print(f"  [BART] SKIPPED: {msg}", flush=True)
        out.attrs["bart_error"] = msg
        return out

    A = df["A"].to_numpy(float)
    t = df["time"].to_numpy(float)
    e = df["event"].to_numpy(float)
    ctrl = A == 0
    risk0 = km_risk(t[ctrl], e[ctrl], float(horizon)) if ctrl.any() else float("nan")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cohort, spec, res = td / "cohort.csv", td / "spec.json", td / "out.json"
        df.to_csv(cohort, index=False)
        spec.write_text(json.dumps({"x_cols": list(x_cols), "w_cols": list(w_cols),
                                    "z_cols": list(z_cols), "horizon": float(horizon),
                                    "adjust": adjust}))
        cmd = [exe, "-u", str(_HERE / "bart.py"), "--cohort", str(cohort), "--spec", str(spec),
               "--out", str(res), "--draws", str(draws), "--tune", str(tune),
               "--trees", str(trees), "--max-bins", str(max_bins),
               "--subsample", str(subsample), "--seed", str(seed)]
        if verbose:
            print(f"  [BART] {exe} draws={draws} tune={tune} bins={max_bins} ...", flush=True)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            msg = f"timed out after {timeout}s"
            if verbose:
                print(f"  [BART] FAILED: {msg}", flush=True)
            out.attrs["bart_error"] = msg
            return out
        if proc.returncode != 0 or not res.exists():
            msg = f"rc={proc.returncode}: {(proc.stderr or '')[-300:]}"
            if verbose:
                print(f"  [BART] FAILED: {msg}", flush=True)
            out.attrs["bart_error"] = msg
            return out
        payload = json.loads(res.read_text())

    rows = []
    for key, v in (payload.get("estimands") or {}).items():
        scale, estimand = key.split("|")
        point = float(v["point"])
        ci_lo, ci_hi = float(v["ci_lo"]), float(v["ci_hi"])
        row = {"method": "BART", "scale": scale, "estimand": estimand,
               "n": int(payload.get("n", len(df))), "n_ev": int(payload.get("n_ev", int(e.sum()))),
               "n_treated": int((A == 1).sum()),
               "point": point, "bs_mean": point,
               "bs_lo": np.nan, "bs_hi": np.nan, "bs_range": "",
               "ci_lo": ci_lo, "ci_hi": ci_hi,
               "n_reps": int(payload.get("draws", draws)),
               "point_vs_bsmean": 0.0,
               # Same same-side rule the bootstrap rows use: an interval on the opposite side
               # of the reported estimate is not evidence in the reported direction.
               "excludes_zero": bool((ci_lo > 0 and point > 0) or (ci_hi < 0 and point < 0)),
               "n_cohort": int(len(df)), "uq": "posterior"}
        if scale == "survival.probability" and np.isfinite(risk0):
            ev = evalue_multi(point, ci_lo, ci_hi, risk0)
            if ev.get("selected") is not None:
                row.update({"risk0": ev["risk0"], "rr": ev["rr"], "e_value": ev["e_value"],
                            "e_value_ci": ev["e_value_ci"], "evalue_method": ev["selected"]})
            elif ev.get("degenerate"):
                row.update({"risk0": float(risk0), "e_value": float("nan"),
                            "degenerate_survprob": True,
                            "degenerate_reason": ev.get("reason", "")})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["risk0"] = risk0
    return out
