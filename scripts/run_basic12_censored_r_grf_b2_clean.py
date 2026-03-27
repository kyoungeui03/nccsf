from __future__ import annotations

import pandas as pd
from pathlib import Path

from grf.benchmarks.econml_8variant import CASE_SPECS, evaluate_r_csf_variant, prepare_case


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmark_basic12_r_csf_b2_clean"
MODEL_NAME = "R-CSF Baseline"
NUM_TREES = 200
TARGET = "RMST"


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
            avg_time=("total_time", "mean"),
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
        case = prepare_case(case_spec, target=TARGET, horizon_quantile=0.60)
        feature_cols = [*case.x_cols, *case.w_cols, *case.z_cols]
        row = evaluate_r_csf_variant(
            MODEL_NAME,
            case.obs_df,
            feature_cols,
            case.true_cate,
            case.horizon,
            num_trees=NUM_TREES,
            target=TARGET,
        )
        row["case_id"] = int(case_spec["case_id"])
        row["slug"] = str(case_spec["slug"])
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_DIR / "results_full.csv", index=False)
    _write_summary(results_df, OUTPUT_DIR / "summary_full.csv")


if __name__ == "__main__":
    main()
