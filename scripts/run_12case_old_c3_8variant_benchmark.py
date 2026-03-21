#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks import (  # noqa: E402
    CASE_SPECS,
    render_avg_summary_png,
    render_b2_vs_c3_png,
    render_case_table_png,
    render_top5_png,
    run_case_benchmark,
    summarize_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the legacy Old C3 8-variant benchmark across the standardized censored synthetic cases."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    parser.add_argument("--num-trees-b2", type=int, default=200)
    parser.add_argument("--target", choices=["RMST", "survival.probability"], default="RMST")
    parser.add_argument("--horizon-quantile", type=float, default=0.60)
    parser.add_argument("--case-ids", nargs="*", type=int, help="Optional subset of case ids.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        if args.target == "RMST":
            output_dir = PROJECT_ROOT / "outputs" / "benchmark_old_c3_8variant_12case"
        else:
            output_dir = PROJECT_ROOT / "outputs" / "benchmark_old_c3_8variant_12case_survival_probability"
    else:
        output_dir = args.output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = set(args.case_ids) if args.case_ids else None
    selected_cases = [case for case in CASE_SPECS if selected_ids is None or case["case_id"] in selected_ids]

    case_frames = []
    for case in selected_cases:
        print("=" * 100)
        print(f"Running case {case['case_id']:02d}")
        print(case["title"])
        print(f"target={args.target}, horizon_quantile={args.horizon_quantile:.2f}")
        print("=" * 100)
        case_df = run_case_benchmark(
            case,
            num_trees_b2=args.num_trees_b2,
            verbose=True,
            target=args.target,
            horizon_quantile=args.horizon_quantile,
        )
        case_frames.append(case_df)

        base_name = f"case_{case['case_id']:02d}_{case['slug']}"
        case_csv = output_dir / f"{base_name}.csv"
        case_png = output_dir / f"{base_name}.png"
        case_df.to_csv(case_csv, index=False)
        render_case_table_png(case_df, case_png)
        print(f"Saved {case_csv}")
        print(f"Saved {case_png}")

    combined_df = pd.concat(case_frames, ignore_index=True)
    combined_csv = output_dir / "all_12case_8variant_results.csv"
    combined_df.to_csv(combined_csv, index=False)

    summary_df, top5_df = summarize_results(combined_df)
    summary_csv = output_dir / "all_12case_8variant_avg_summary.csv"
    top5_csv = output_dir / "all_12case_8variant_top5.csv"
    summary_png = output_dir / "all_12case_8variant_avg_summary.png"
    top5_png = output_dir / "all_12case_8variant_top5.png"
    compare_png = output_dir / "b2_vs_c3_sqrt_pehe_comparison.png"

    summary_df.to_csv(summary_csv, index=False)
    top5_df.to_csv(top5_csv, index=False)
    render_avg_summary_png(summary_df, summary_png)
    render_top5_png(top5_df, top5_png)
    render_b2_vs_c3_png(combined_df, compare_png)

    audit = {
        "benchmark_definition": {
            "B1": "installed R grf baseline with X only",
            "B2": "installed R grf baseline with X+W+Z",
            "A1": "oracle X+U final forest, true S/lambda/K_tau/q/h",
            "A2": "oracle X+U final forest, true survival nuisance and estimated q/h",
            "A3": "oracle X+U final forest, all estimated with econml mild shrink",
            "C1": "NC X+W+Z final forest, true S/lambda/K_tau/q/h",
            "C2": "NC X+W+Z final forest, true survival nuisance and estimated q/h",
            "C3": "legacy Old C3: NC X+W+Z final forest, all estimated with econml mild shrink",
        },
        "metrics": [
            "Pred CATE",
            "True CATE",
            "Bias",
            "RMSE",
            "PEHE",
            "MAE",
            "Pearson",
            "Sign accuracy",
            "Time",
        ],
        "num_cases": int(len(selected_cases)),
        "target": args.target,
        "horizon_quantile": float(args.horizon_quantile),
    }
    audit_path = output_dir / "implementation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"Saved {combined_csv}")
    print(f"Saved {summary_csv}")
    print(f"Saved {top5_csv}")
    print(f"Saved {summary_png}")
    print(f"Saved {top5_png}")
    print(f"Saved {compare_png}")
    print(f"Saved {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
