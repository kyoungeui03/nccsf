#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/kyoungeuihong/Desktop/csf_grf_new"
NC_RUNNER="$ROOT/scripts/run_14setting_non_censored_unified_b2sum_family.py"
C_RUNNER="$ROOT/scripts/run_14setting_censored_unified_b2sum_family.py"
NC_OUT="$ROOT/non_censored/outputs/benchmark_structured_14settings_unified_b2sum_final5"
C_OUT="$ROOT/outputs/benchmark_structured_14settings_unified_b2sum_final5"
LOGDIR="$ROOT/outputs/logs_unified_b2sum_final5_14setting"

mkdir -p "$LOGDIR" "$NC_OUT" "$C_OUT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export RCPP_PARALLEL_NUM_THREADS=1

Q_A=(S14 S07 S10 S03 S12 S01)
Q_B=(S11 S13 S04 S09 S06 S08 S05 S02)

python3 "$NC_RUNNER" --output-dir "$NC_OUT" --setting-ids "${Q_A[@]}" > "$LOGDIR/nc_q1.log" 2>&1 &
NC_Q1=$!
python3 "$NC_RUNNER" --output-dir "$NC_OUT" --setting-ids "${Q_B[@]}" > "$LOGDIR/nc_q2.log" 2>&1 &
NC_Q2=$!
python3 "$C_RUNNER" --output-dir "$C_OUT" --setting-ids "${Q_A[@]}" > "$LOGDIR/c_q1.log" 2>&1 &
C_Q1=$!
python3 "$C_RUNNER" --output-dir "$C_OUT" --setting-ids "${Q_B[@]}" > "$LOGDIR/c_q2.log" 2>&1 &
C_Q2=$!

{
  echo "nc_q1 $NC_Q1"
  echo "nc_q2 $NC_Q2"
  echo "c_q1 $C_Q1"
  echo "c_q2 $C_Q2"
} > "$LOGDIR/pids.txt"

echo "Started unified B2Sum 14-setting overnight run."
echo "NC output: $NC_OUT"
echo "C output:  $C_OUT"
echo "Logs:      $LOGDIR"
echo "PIDs saved to $LOGDIR/pids.txt"
