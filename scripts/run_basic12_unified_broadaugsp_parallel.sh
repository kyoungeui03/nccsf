#!/bin/zsh
set -euo pipefail

ROOT="/Users/kyoungeuihong/Desktop/csf_grf_new"
PYTHON="python3"
LOG_DIR="$ROOT/outputs/logs_basic12_unified_broadaugsp"
NONC_OUT="$ROOT/non_censored/outputs/benchmark_basic12_unified_broadaugsp_non_censored"
CENS_OUT="$ROOT/outputs/benchmark_basic12_unified_broadaugsp_censored"
NONC_RUNNER="$ROOT/scripts/run_basic12_unified_broadaugsp_non_censored.py"
CENS_RUNNER="$ROOT/scripts/run_basic12_unified_broadaugsp_censored.py"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export RCPP_PARALLEL_NUM_THREADS=1

mkdir -p "$LOG_DIR" "$NONC_OUT" "$CENS_OUT"

run_logged() {
  local label="$1"
  shift
  echo "[$label] $(date '+%Y-%m-%d %H:%M:%S') start" | tee -a "$LOG_DIR/${label}.log"
  "$@" 2>&1 | tee -a "$LOG_DIR/${label}.log"
  echo "[$label] $(date '+%Y-%m-%d %H:%M:%S') done" | tee -a "$LOG_DIR/${label}.log"
}

run_logged nonc_q1 "$PYTHON" "$NONC_RUNNER" --output-dir "$NONC_OUT" --tag q1 --case-ids 1 2 3 4 5 6 &
PID_A=$!
run_logged nonc_q2 "$PYTHON" "$NONC_RUNNER" --output-dir "$NONC_OUT" --tag q2 --case-ids 7 8 9 10 11 12 &
PID_B=$!
run_logged cens_q1 "$PYTHON" "$CENS_RUNNER" --output-dir "$CENS_OUT" --tag q1 --case-ids 1 2 3 4 5 6 &
PID_C=$!
run_logged cens_q2 "$PYTHON" "$CENS_RUNNER" --output-dir "$CENS_OUT" --tag q2 --case-ids 7 8 9 10 11 12 &
PID_D=$!

wait "$PID_A"
wait "$PID_B"
wait "$PID_C"
wait "$PID_D"

"$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd

def combine(output_dir: Path, time_col: str, metric_cols: list[str]):
    parts = []
    for tag in ("q1", "q2"):
        parts.append(pd.read_csv(output_dir / f"results_{tag}.csv"))
    results = pd.concat(parts, ignore_index=True).sort_values(["case_id", "name"]).reset_index(drop=True)
    results.to_csv(output_dir / "results_combined.csv", index=False)
    agg = {col: "mean" for col in metric_cols}
    agg[time_col] = "mean"
    summary = (
        results.groupby("name", as_index=False)
        .agg(**{
            "avg_rmse": ("rmse", "mean"),
            "avg_mae": ("mae", "mean"),
            "avg_pearson": ("pearson", "mean"),
            "avg_time": (time_col, "mean"),
        })
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    summary.to_csv(output_dir / "summary_combined.csv", index=False)

combine(
    Path("/Users/kyoungeuihong/Desktop/csf_grf_new/non_censored/outputs/benchmark_basic12_unified_broadaugsp_non_censored"),
    "time_sec",
    ["rmse", "mae", "pearson"],
)
combine(
    Path("/Users/kyoungeuihong/Desktop/csf_grf_new/outputs/benchmark_basic12_unified_broadaugsp_censored"),
    "total_time",
    ["rmse", "mae", "pearson"],
)
PY

echo "[parallel] $(date '+%Y-%m-%d %H:%M:%S') all queues done" | tee -a "$LOG_DIR/master.log"
