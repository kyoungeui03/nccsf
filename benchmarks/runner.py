"""Command-line entry point: run the baselines on one cohort CSV and write a results CSV.

    python -m benchmarks.runner --csv cohort.csv --horizon 90 \
        --x-cols X0,X1,X2 --w-cols W --z-cols Z --out results.csv

The cohort needs columns A (0/1 treatment), time, event (0/1), plus the covariates named.
Column names are given explicitly rather than guessed: silently defaulting to "everything
else is a confounder" would change the estimand without anyone noticing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import ALL_METHODS, run_all


def _cols(s: str) -> list[str]:
    return [c.strip() for c in (s or "").split(",") if c.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.runner",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="cohort CSV with A, time, event + covariates")
    ap.add_argument("--horizon", required=True, type=float, help="h for S(h) and RMST(h)")
    ap.add_argument("--out", required=True, help="destination CSV")
    ap.add_argument("--x-cols", required=True, help="comma-separated confounders")
    ap.add_argument("--w-cols", default="", help="comma-separated outcome proxies (NCO)")
    ap.add_argument("--z-cols", default="", help="comma-separated treatment proxies (NCE)")
    ap.add_argument("--methods", default="", help=f"comma-separated subset of {list(ALL_METHODS)}")
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adjust", choices=["xwz", "x"], default="xwz",
                    help="xwz (default) puts the proxies in as ordinary covariates; x drops them")
    ap.add_argument("--subsample", type=int, default=0, help="cap rows (0 = use all)")
    ap.add_argument("--bart-python", default=None,
                    help="interpreter with pymc-bart; also read from $BART_PYTHON")
    a = ap.parse_args(argv)

    df = pd.read_csv(a.csv)
    missing = [c for c in ("A", "time", "event") if c not in df.columns]
    if missing:
        print(f"cohort is missing required column(s): {missing}", file=sys.stderr)
        return 2
    if a.subsample and len(df) > a.subsample:
        df = df.sample(a.subsample, random_state=a.seed).reset_index(drop=True)

    x, w, z = _cols(a.x_cols), _cols(a.w_cols), _cols(a.z_cols)
    unknown = [c for c in x + w + z if c not in df.columns]
    if unknown:
        print(f"column(s) not in the cohort: {unknown}", file=sys.stderr)
        return 2

    print(f"n={len(df)} treated={int(df['A'].sum())} events={int(df['event'].sum())} "
          f"h={a.horizon:g} X={len(x)} W={len(w)} Z={len(z)} adjust={a.adjust}", flush=True)
    res = run_all(df, a.horizon, x_cols=x, w_cols=w, z_cols=z,
                  methods=_cols(a.methods) or None, n_boot=a.n_boot, seed=a.seed,
                  adjust=a.adjust, bart_python=a.bart_python)
    if not len(res):
        print("no rows produced", file=sys.stderr)
        return 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    show = [c for c in ("method", "scale", "estimand", "n", "point", "bs_mean",
                        "ci_lo", "ci_hi", "excludes_zero", "e_value") if c in res]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\n" + res[show].to_string(index=False))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
