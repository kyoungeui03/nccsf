from __future__ import annotations

import csv
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

from grf.non_censored.benchmarks import CASE_SPECS, _build_case, _make_cfg, _metric_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]
R_CF_SCRIPT = PROJECT_ROOT / "scripts" / "run_grf_cf_baseline.R"
OUTPUT_DIR = PROJECT_ROOT / "non_censored" / "outputs" / "benchmark_basic12_r_grf_b2_clean"
MODEL_NAME = "R-CF Baseline"
NUM_TREES = 200
SEED = 42


def _write_summary(results_df: pd.DataFrame, output_path: Path) -> None:
    summary = (
        results_df.groupby("name", as_index=False)
        .agg(
            avg_pred_cate=("mean_pred", "mean"),
            avg_true_cate=("mean_true_cate", "mean"),
            avg_acc=("sign_acc", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_pehe=("pehe", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
            avg_time=("time_sec", "mean"),
            n_cases=("case_id", "count"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    summary.to_csv(output_path, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    for case_spec in CASE_SPECS:
        case = _build_case(_make_cfg(case_spec), case_spec)
        feature_cols = [*case["x_cols"], *case["w_cols"], *case["z_cols"]]

        with tempfile.TemporaryDirectory(dir=OUTPUT_DIR, prefix=f"case_{case_spec['case_id']:02d}_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            input_path = tmp_dir_path / "input.csv"
            output_path = tmp_dir_path / "predictions.csv"

            obs_df = case["obs_df"].loc[:, [*feature_cols, "A"]].copy()
            obs_df["Y"] = case["Y"]
            obs_df.to_csv(input_path, index=False)

            cmd = [
                "Rscript",
                str(R_CF_SCRIPT),
                str(input_path),
                ",".join(feature_cols),
                str(NUM_TREES),
                str(output_path),
                str(SEED),
            ]
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
            elapsed = time.time() - t0
            if proc.returncode != 0:
                raise RuntimeError(
                    f"R causal_forest baseline failed for case {case_spec['case_id']}.\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                )

            preds = pd.read_csv(output_path)["pred"].to_numpy(dtype=float)
            row = _metric_row(MODEL_NAME, preds, case["true_cate"], elapsed)
            row["case_id"] = int(case_spec["case_id"])
            row["slug"] = str(case_spec["slug"])
            rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_DIR / "results_full.csv", index=False)
    _write_summary(results_df, OUTPUT_DIR / "summary_full.csv")


if __name__ == "__main__":
    main()
