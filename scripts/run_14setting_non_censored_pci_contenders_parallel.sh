#!/bin/zsh
set -euo pipefail

ROOT="/Users/kyoungeuihong/Desktop/csf_grf_new"
LOG_DIR="$ROOT/non_censored/outputs/logs_pci_contenders"
PYTHON="python3"
RUNNER="$ROOT/scripts/run_14setting_non_censored_pci_contenders.py"

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

# Three workers is still well below machine capacity with thread counts pinned to 1.
# Early queues prioritize quicker settings so the first summary becomes available sooner.
worker_a() {
  run_queue q1 --setting-ids S01 S05 S10 S14
}

worker_b() {
  run_queue q2 --setting-ids S02 S06 S11
}

worker_c() {
  run_queue q3 --setting-ids S12 S03 S08 S07 S09 S04 S13
}

worker_a &
PID_A=$!
worker_b &
PID_B=$!
worker_c &
PID_C=$!

wait "$PID_A"
wait "$PID_B"
wait "$PID_C"

echo "[parallel] $(date '+%Y-%m-%d %H:%M:%S') all queues done" | tee -a "$LOG_DIR/master.log"
