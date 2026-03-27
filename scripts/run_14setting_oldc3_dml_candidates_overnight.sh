#!/bin/zsh
set -euo pipefail

ROOT="/Users/kyoungeuihong/Desktop/csf_grf_new"
LOG_DIR="$ROOT/outputs/logs_oldc3_dml_candidates"
PYTHON="python3"
RUNNER="$ROOT/scripts/run_14setting_oldc3_dml_candidates.py"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$LOG_DIR"

run_queue() {
  local label="$1"
  shift
  echo "[$label] $(date '+%Y-%m-%d %H:%M:%S') start" | tee -a "$LOG_DIR/${label}.log"
  "$PYTHON" "$RUNNER" "$@" 2>&1 | tee -a "$LOG_DIR/${label}.log"
  echo "[$label] $(date '+%Y-%m-%d %H:%M:%S') done" | tee -a "$LOG_DIR/${label}.log"
}

worker_a() {
  run_queue nc_q1 --domains non_censored --setting-ids S01 S12 S03 S10 S07 S14
  run_queue c_q2 --domains censored --setting-ids S02 S05 S08 S06 S09 S04 S11 S13
}

worker_b() {
  run_queue c_q1 --domains censored --setting-ids S01 S12 S03 S10 S07 S14
  run_queue nc_q2 --domains non_censored --setting-ids S02 S05 S08 S06 S09 S04 S11 S13
}

worker_a &
PID_A=$!
worker_b &
PID_B=$!

wait "$PID_A"
wait "$PID_B"

echo "[overnight] $(date '+%Y-%m-%d %H:%M:%S') all queues done" | tee -a "$LOG_DIR/overnight_master.log"
