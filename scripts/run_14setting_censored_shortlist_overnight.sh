#!/bin/zsh
set -euo pipefail

ROOT="/Users/kyoungeuihong/Desktop/csf_grf_new"
RUNNER="$ROOT/scripts/run_14setting_censored_shortlist.py"
OUT="$ROOT/outputs/benchmark_structured_14settings_censored_shortlist"
LOGDIR="$ROOT/outputs/logs_censored_shortlist"

mkdir -p "$OUT" "$LOGDIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export RCPP_PARALLEL_NUM_THREADS=1

echo "[launch] output dir: $OUT"
echo "[launch] logs dir: $LOGDIR"

python3 "$RUNNER" --output-dir "$OUT" --setting-ids S01 S05 S09 S13 > "$LOGDIR/q1.log" 2>&1 &
P1=$!
python3 "$RUNNER" --output-dir "$OUT" --setting-ids S02 S06 S10 > "$LOGDIR/q2.log" 2>&1 &
P2=$!
python3 "$RUNNER" --output-dir "$OUT" --setting-ids S03 S07 S11 > "$LOGDIR/q3.log" 2>&1 &
P3=$!
python3 "$RUNNER" --output-dir "$OUT" --setting-ids S04 S08 S12 S14 > "$LOGDIR/q4.log" 2>&1 &
P4=$!

echo "$P1 q1 S01 S05 S09 S13" > "$LOGDIR/pids.txt"
echo "$P2 q2 S02 S06 S10" >> "$LOGDIR/pids.txt"
echo "$P3 q3 S03 S07 S11" >> "$LOGDIR/pids.txt"
echo "$P4 q4 S04 S08 S12 S14" >> "$LOGDIR/pids.txt"

echo "[launch] pids recorded in $LOGDIR/pids.txt"

wait "$P1" "$P2" "$P3" "$P4"

python3 - <<'PY'
import pandas as pd
from pathlib import Path

SETTINGS = [
    {"setting_id": "S01", "n": 1000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S02", "n": 2000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S03", "n": 4000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S04", "n": 8000, "p_x": 5, "p_w": 1, "p_z": 1},
    {"setting_id": "S05", "n": 2000, "p_x": 5, "p_w": 3, "p_z": 3},
    {"setting_id": "S06", "n": 2000, "p_x": 5, "p_w": 5, "p_z": 5},
    {"setting_id": "S07", "n": 2000, "p_x": 5, "p_w": 10, "p_z": 10},
    {"setting_id": "S08", "n": 2000, "p_x": 10, "p_w": 1, "p_z": 1},
    {"setting_id": "S09", "n": 2000, "p_x": 20, "p_w": 1, "p_z": 1},
    {"setting_id": "S10", "n": 2000, "p_x": 10, "p_w": 5, "p_z": 5},
    {"setting_id": "S11", "n": 2000, "p_x": 20, "p_w": 5, "p_z": 5},
    {"setting_id": "S12", "n": 1000, "p_x": 10, "p_w": 5, "p_z": 5},
    {"setting_id": "S13", "n": 2000, "p_x": 10, "p_w": 10, "p_z": 10},
    {"setting_id": "S14", "n": 4000, "p_x": 20, "p_w": 10, "p_z": 10},
]

def setting_slug(setting):
    return f"{setting['setting_id']}_n{setting['n']}_px{setting['p_x']}_pw{setting['p_w']}_pz{setting['p_z']}"

out = Path("/Users/kyoungeuihong/Desktop/csf_grf_new/outputs/benchmark_structured_14settings_censored_shortlist")
frames = []
for setting in SETTINGS:
    path = out / setting_slug(setting) / "results.csv"
    if path.exists():
        frames.append(pd.read_csv(path))
if not frames:
    raise SystemExit("No results found to combine.")
df = pd.concat(frames, ignore_index=True)
df.to_csv(out / "all_settings_results.csv", index=False)
summary = (
    df.groupby("name", as_index=False)
      .agg(
          avg_rmse=("rmse", "mean"),
          avg_mae=("mae", "mean"),
          avg_pearson=("pearson", "mean"),
          avg_bias=("bias", "mean"),
          avg_sign_acc=("sign_acc", "mean"),
          avg_time=("total_time", "mean"),
      )
      .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
      .reset_index(drop=True)
)
summary.insert(0, "rank", range(1, len(summary) + 1))
summary.to_csv(out / "all_settings_summary.csv", index=False)
print(summary.to_string(index=False))
PY

echo "[done] combined outputs written under $OUT"
