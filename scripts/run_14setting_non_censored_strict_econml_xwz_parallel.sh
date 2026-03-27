#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/kyoungeuihong/Desktop/csf_grf_new"
SCRIPT="$ROOT/scripts/run_14setting_non_censored_strict_econml_xwz.py"
OUT="$ROOT/non_censored/outputs/benchmark_structured_14settings_strict_econml_xwz"
LOGDIR="$ROOT/non_censored/outputs/logs_strict_econml_xwz"

mkdir -p "$OUT" "$LOGDIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONPATH="$ROOT/python-package:${PYTHONPATH:-}"

python3 "$SCRIPT" --setting-ids S01 S05 S09 S13 --output-dir "$OUT" > "$LOGDIR/q1.log" 2>&1 &
P1=$!
python3 "$SCRIPT" --setting-ids S02 S06 S10 --output-dir "$OUT" > "$LOGDIR/q2.log" 2>&1 &
P2=$!
python3 "$SCRIPT" --setting-ids S03 S07 S11 --output-dir "$OUT" > "$LOGDIR/q3.log" 2>&1 &
P3=$!
python3 "$SCRIPT" --setting-ids S04 S08 S12 S14 --output-dir "$OUT" > "$LOGDIR/q4.log" 2>&1 &
P4=$!

{
  echo "$P1 q1"
  echo "$P2 q2"
  echo "$P3 q3"
  echo "$P4 q4"
} > "$LOGDIR/pids.txt"

wait "$P1" "$P2" "$P3" "$P4"
