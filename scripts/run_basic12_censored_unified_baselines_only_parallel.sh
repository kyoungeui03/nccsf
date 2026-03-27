#!/bin/zsh
set -euo pipefail

ROOT="/Users/kyoungeuihong/Desktop/csf_grf_new"
PYTHON="python3"
RUNNER="$ROOT/scripts/run_basic12_censored_unified_baselines_only.py"
OUT="$ROOT/outputs/benchmark_basic12_censored_unified_baselines_only_rerun"
LOG_DIR="$ROOT/outputs/logs_basic12_censored_unified_baselines_only_rerun"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export RCPP_PARALLEL_NUM_THREADS=1

mkdir -p "$OUT" "$LOG_DIR"

run_logged() {
  local label="$1"
  shift
  echo "[$label] $(date '+%Y-%m-%d %H:%M:%S') start" | tee -a "$LOG_DIR/${label}.log"
  "$@" 2>&1 | tee -a "$LOG_DIR/${label}.log"
  echo "[$label] $(date '+%Y-%m-%d %H:%M:%S') done" | tee -a "$LOG_DIR/${label}.log"
}

run_logged q1 "$PYTHON" "$RUNNER" --output-dir "$OUT" --tag q1 --case-ids 1 2 3 4 5 6 &
PID_A=$!
run_logged q2 "$PYTHON" "$RUNNER" --output-dir "$OUT" --tag q2 --case-ids 7 8 9 10 11 12 &
PID_B=$!

wait "$PID_A"
wait "$PID_B"

"$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd

out = Path("/Users/kyoungeuihong/Desktop/csf_grf_new/outputs/benchmark_basic12_censored_unified_baselines_only_rerun")
results = pd.concat(
    [pd.read_csv(out / "results_q1.csv"), pd.read_csv(out / "results_q2.csv")],
    ignore_index=True,
).sort_values(["case_id", "name"]).reset_index(drop=True)
results.to_csv(out / "results_combined.csv", index=False)
summary = (
    results.groupby("name", as_index=False)
    .agg(
        avg_rmse=("rmse", "mean"),
        avg_mae=("mae", "mean"),
        avg_pearson=("pearson", "mean"),
        avg_time=("total_time", "mean"),
    )
    .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
    .reset_index(drop=True)
)
summary.insert(0, "rank", range(1, len(summary) + 1))
summary.to_csv(out / "summary_combined.csv", index=False)
PY

echo "[parallel] $(date '+%Y-%m-%d %H:%M:%S') all queues done" | tee -a "$LOG_DIR/master.log"
