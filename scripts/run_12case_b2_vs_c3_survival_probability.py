#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.benchmarks.econml_8variant import (  # noqa: E402
    CASE_SPECS,
    R_CSF_SCRIPT,
    _event_eta,
    _make_forest_kwargs,
    _weibull_scale,
    build_case_cfg,
    recover_dgp_internals,
    render_avg_summary_png,
    render_b2_vs_c3_png,
)
from grf.methods import EconmlMildShrinkNCSurvivalForest  # noqa: E402
from grf.synthetic import add_ground_truth_cate, generate_synthetic_nc_cox  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 12-case survival.probability comparison for B2 vs C3."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benchmark_b2_vs_c3_survival_probability_12case",
    )
    parser.add_argument("--num-trees-b2", type=int, default=200)
    parser.add_argument("--horizon-quantile", type=float, default=0.60)
    parser.add_argument("--case-ids", nargs="*", type=int)
    return parser.parse_args()


def true_survival_probability_cate(X, U, horizon, cfg, dgp):
    a0 = np.zeros(X.shape[0], dtype=float)
    a1 = np.ones(X.shape[0], dtype=float)
    eta0 = _event_eta(X, U, a0, cfg, dgp)
    eta1 = _event_eta(X, U, a1, cfg, dgp)
    scale0 = _weibull_scale(cfg.lam_t, cfg.k_t, eta0)
    scale1 = _weibull_scale(cfg.lam_t, cfg.k_t, eta1)
    s0 = np.exp(-((float(horizon) / scale0) ** cfg.k_t))
    s1 = np.exp(-((float(horizon) / scale1) ** cfg.k_t))
    return s1 - s0


def evaluate_predictions(name, preds, true_cate, elapsed, backend):
    preds = np.asarray(preds, dtype=float)
    true_cate = np.asarray(true_cate, dtype=float)
    pehe = float(np.sqrt(np.mean((preds - true_cate) ** 2)))
    return dict(
        name=name,
        mean_pred=float(np.mean(preds)),
        std_pred=float(np.std(preds)),
        mean_true_cate=float(np.mean(true_cate)),
        std_true_cate=float(np.std(true_cate)),
        bias=float(np.mean(preds - true_cate)),
        rmse=pehe,
        pehe=pehe,
        mae=float(np.mean(np.abs(preds - true_cate))),
        pearson=float(np.corrcoef(preds, true_cate)[0, 1]),
        sign_acc=float(np.mean(np.sign(preds) == np.sign(true_cate))),
        total_time=float(elapsed),
        backend=backend,
    )


def evaluate_r_csf_variant(obs_df, feature_cols, true_cate, horizon, num_trees):
    (PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs", prefix="survprob_r_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        input_path = tmp_dir / "input.csv"
        output_path = tmp_dir / "predictions.csv"
        obs_df.loc[:, ["time", "event", "A", *feature_cols]].to_csv(input_path, index=False)
        cmd = [
            "Rscript",
            str(R_CSF_SCRIPT),
            str(input_path),
            ",".join(feature_cols),
            str(float(horizon)),
            str(int(num_trees)),
            str(output_path),
            "survival.probability",
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"Installed R grf baseline failed.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        preds = pd.read_csv(output_path)["prediction"].to_numpy(dtype=float)
    return preds, elapsed


def evaluate_c3_variant(X, W, Z, A, Y, D, true_cate, horizon):
    model = EconmlMildShrinkNCSurvivalForest(
        target="survival.probability",
        horizon=float(horizon),
        **_make_forest_kwargs(),
    )
    t0 = time.time()
    model.fit_components(X, A, Y, D, Z, W)
    preds = model.effect_from_components(X, W, Z).ravel()
    elapsed = time.time() - t0
    return preds, elapsed


def summarize_results(combined_df):
    summary = (
        combined_df.groupby("name", as_index=False)
        .agg(
            avg_pred_cate=("mean_pred", "mean"),
            avg_true_cate=("mean_true_cate", "mean"),
            avg_acc=("sign_acc", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_pehe=("pehe", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
        )
        .sort_values(["avg_pehe", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = set(args.case_ids) if args.case_ids else None
    selected_cases = [case for case in CASE_SPECS if selected_ids is None or case["case_id"] in selected_ids]

    rows = []
    for case_spec in selected_cases:
        cfg = build_case_cfg(case_spec)
        obs_df, truth_df, params = generate_synthetic_nc_cox(cfg)
        obs_df, truth_df = add_ground_truth_cate(obs_df, truth_df, cfg, params)
        dgp = recover_dgp_internals(cfg)

        x_cols = [f"X{j}" for j in range(cfg.p_x)]
        X = obs_df[x_cols].to_numpy()
        W = obs_df[["W"]].to_numpy()
        Z = obs_df[["Z"]].to_numpy()
        A = obs_df["A"].to_numpy(dtype=float)
        Y = obs_df["time"].to_numpy(dtype=float)
        D = obs_df["event"].to_numpy(dtype=float)
        U = truth_df["U"].to_numpy(dtype=float)

        horizon = float(np.quantile(Y, args.horizon_quantile))
        true_cate = true_survival_probability_cate(X, U, horizon, cfg, dgp)

        print("=" * 100)
        print(f"Running case {case_spec['case_id']:02d}")
        print(case_spec["title"])
        print(f"target=survival.probability, horizon_quantile={args.horizon_quantile:.2f}, horizon={horizon:.6f}")
        print("=" * 100)

        b2_preds, b2_elapsed = evaluate_r_csf_variant(obs_df, x_cols + ["W", "Z"], true_cate, horizon, args.num_trees_b2)
        c3_preds, c3_elapsed = evaluate_c3_variant(X, W, Z, A, Y, D, true_cate, horizon)

        for name, preds, elapsed, backend in [
            ("B2  R-CSF baseline (X+W+Z)", b2_preds, b2_elapsed, "installed R grf"),
            ("C3  NC-CSF (all estimated)", c3_preds, c3_elapsed, "econml mild shrink"),
        ]:
            row = evaluate_predictions(name, preds, true_cate, elapsed, backend)
            row.update(
                case_id=case_spec["case_id"],
                case_slug=case_spec["slug"],
                case_title=case_spec["title"],
                target="survival.probability",
                horizon=horizon,
                horizon_quantile=float(args.horizon_quantile),
                n=cfg.n,
                p_x=cfg.p_x,
                seed=cfg.seed,
                target_censor_rate=cfg.target_censor_rate,
                actual_censor_rate=float(1.0 - D.mean()),
                linear_treatment=cfg.linear_treatment,
                linear_outcome=cfg.linear_outcome,
                tau_log_hr=cfg.tau_log_hr,
            )
            rows.append(row)

    combined_df = pd.DataFrame(rows)
    combined_csv = output_dir / "all_12case_b2_vs_c3_survival_probability_results.csv"
    summary_csv = output_dir / "all_12case_b2_vs_c3_survival_probability_summary.csv"
    summary_png = output_dir / "all_12case_b2_vs_c3_survival_probability_summary.png"
    compare_png = output_dir / "b2_vs_c3_survival_probability_comparison.png"

    combined_df.to_csv(combined_csv, index=False)
    summary_df = summarize_results(combined_df)
    summary_df.to_csv(summary_csv, index=False)
    render_avg_summary_png(summary_df, summary_png)
    render_b2_vs_c3_png(combined_df, compare_png)

    print(f"Saved {combined_csv}")
    print(f"Saved {summary_csv}")
    print(f"Saved {summary_png}")
    print(f"Saved {compare_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
