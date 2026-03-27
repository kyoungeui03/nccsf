#!/bin/zsh
set -euo pipefail

ROOT="/Users/kyoungeuihong/Desktop/csf_grf_new"
OUT="$ROOT/non_censored/outputs/benchmark_structured_14settings_pci_contenders"
LOGDIR="$ROOT/non_censored/outputs/logs_pci_contenders"
RUNNER="$ROOT/scripts/run_14setting_non_censored_pci_contenders.py"
CTRLLOG="$LOGDIR/reassign_q1_follow.log"
Q1BLOG="$LOGDIR/q1b.log"

Q1_PID="${1:?q1 pid required}"
Q3_PID="${2:?q3 pid required}"
Q4_PID="${3:?q4 pid required}"

S12="$OUT/S12_n1000_px10_pw5_pz5/summary.csv"
S08="$OUT/S08_n2000_px10_pw1_pz1/summary.csv"

mkdir -p "$LOGDIR"

echo "[reassign] $(date '+%Y-%m-%d %H:%M:%S') watcher start" >> "$CTRLLOG"

while [ ! -f "$S12" ]; do
  sleep 20
done
echo "[reassign] $(date '+%Y-%m-%d %H:%M:%S') S12 done; pausing q4 ($Q4_PID) before S13" >> "$CTRLLOG"
kill -STOP "$Q4_PID" 2>/dev/null || true

while [ ! -f "$S08" ]; do
  sleep 20
done
echo "[reassign] $(date '+%Y-%m-%d %H:%M:%S') S08 done; pausing q3 ($Q3_PID) before S09" >> "$CTRLLOG"
kill -STOP "$Q3_PID" 2>/dev/null || true

while kill -0 "$Q1_PID" 2>/dev/null; do
  sleep 20
done
echo "[reassign] $(date '+%Y-%m-%d %H:%M:%S') q1 done; launching q1b for S09 S13" >> "$CTRLLOG"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[q1b] $(date '+%Y-%m-%d %H:%M:%S') start" | tee -a "$Q1BLOG"
python3 "$RUNNER" --setting-ids S09 S13 2>&1 | tee -a "$Q1BLOG"
echo "[q1b] $(date '+%Y-%m-%d %H:%M:%S') done" | tee -a "$Q1BLOG"
