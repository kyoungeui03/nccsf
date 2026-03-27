#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.non_censored.benchmarks import CASE_SPECS, _build_case, _make_cfg, _metric_row  # noqa: E402
from grf.non_censored.models import _BaseSinglePassBridgeFeatureNCCausalForest  # noqa: E402


def build_model_specs() -> list[tuple[str, dict[str, object]]]:
    base = {
        "prediction_nuisance_mode": "full_refit",
        "observed_only": False,
        "n_estimators": 200,
        "min_samples_leaf": 20,
        "cv": 5,
        "random_state": 42,
        "nuisance_feature_mode": "broad_dup",
        "q_clip": 0.02,
        "y_clip_quantile": 0.99,
        "y_res_clip_percentiles": (1.0, 99.0),
        "n_jobs": 1,
    }
    combos = [
        ("aug_full", "q=logit", {"q_kind": "logit"}, "h=extra400x10", {"h_kind": "extra", "h_n_estimators": 400, "h_min_samples_leaf": 10}),
        ("aug_full", "q=logit", {"q_kind": "logit"}, "h=extra1200x3", {"h_kind": "extra", "h_n_estimators": 1200, "h_min_samples_leaf": 3}),
        ("aug_full", "q=poly2", {"q_kind": "poly2"}, "h=extra400x10", {"h_kind": "extra", "h_n_estimators": 400, "h_min_samples_leaf": 10}),
        ("aug_full", "q=poly2", {"q_kind": "poly2"}, "h=extra1200x3", {"h_kind": "extra", "h_n_estimators": 1200, "h_min_samples_leaf": 3}),
        ("aug_compact", "q=logit", {"q_kind": "logit"}, "h=extra400x10", {"h_kind": "extra", "h_n_estimators": 400, "h_min_samples_leaf": 10}),
        ("aug_compact", "q=logit", {"q_kind": "logit"}, "h=extra1200x3", {"h_kind": "extra", "h_n_estimators": 1200, "h_min_samples_leaf": 3}),
        ("aug_compact", "q=poly2", {"q_kind": "poly2"}, "h=extra400x10", {"h_kind": "extra", "h_n_estimators": 400, "h_min_samples_leaf": 10}),
        ("aug_compact", "q=poly2", {"q_kind": "poly2"}, "h=extra1200x3", {"h_kind": "extra", "h_n_estimators": 1200, "h_min_samples_leaf": 3}),
        ("aug_compact_stats", "q=logit", {"q_kind": "logit"}, "h=extra400x10", {"h_kind": "extra", "h_n_estimators": 400, "h_min_samples_leaf": 10}),
        ("aug_compact_stats", "q=logit", {"q_kind": "logit"}, "h=extra1200x3", {"h_kind": "extra", "h_n_estimators": 1200, "h_min_samples_leaf": 3}),
        ("aug_compact_stats", "q=poly2", {"q_kind": "poly2"}, "h=extra400x10", {"h_kind": "extra", "h_n_estimators": 400, "h_min_samples_leaf": 10}),
        ("aug_compact_stats", "q=poly2", {"q_kind": "poly2"}, "h=extra1200x3", {"h_kind": "extra", "h_n_estimators": 1200, "h_min_samples_leaf": 3}),
        ("summary_compact_stats", "q=logit", {"q_kind": "logit"}, "h=extra400x10", {"h_kind": "extra", "h_n_estimators": 400, "h_min_samples_leaf": 10}),
        ("summary_compact_stats", "q=logit", {"q_kind": "logit"}, "h=extra1200x3", {"h_kind": "extra", "h_n_estimators": 1200, "h_min_samples_leaf": 3}),
        ("summary_compact_stats", "q=poly2", {"q_kind": "poly2"}, "h=extra400x10", {"h_kind": "extra", "h_n_estimators": 400, "h_min_samples_leaf": 10}),
        ("summary_compact_stats", "q=poly2", {"q_kind": "poly2"}, "h=extra1200x3", {"h_kind": "extra", "h_n_estimators": 1200, "h_min_samples_leaf": 3}),
        ("aug_noq", "q=logit", {"q_kind": "logit"}, "h=extra400x10", {"h_kind": "extra", "h_n_estimators": 400, "h_min_samples_leaf": 10}),
        ("aug_noq", "q=logit", {"q_kind": "logit"}, "h=extra1200x3", {"h_kind": "extra", "h_n_estimators": 1200, "h_min_samples_leaf": 3}),
    ]
    specs = []
    for feature_mode, q_label, q_kwargs, h_label, h_kwargs in combos:
        kwargs = dict(base)
        kwargs["final_feature_mode"] = feature_mode
        kwargs.update(q_kwargs)
        kwargs.update(h_kwargs)
        specs.append((f"EliteNC[{feature_mode}|{q_label}|{h_label}]", kwargs))
    return specs


MODEL_SPECS = build_model_specs()


def parse_args():
    parser = argparse.ArgumentParser(description="Run basic12 non-censored elite single-pass search.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p-w", type=int, default=1)
    parser.add_argument("--p-z", type=int, default=1)
    return parser.parse_args()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("name", as_index=False)
        .agg(
            avg_pred_cate=("mean_pred", "mean"),
            avg_true_cate=("mean_true_cate", "mean"),
            avg_acc=("sign_acc", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_bias=("bias", "mean"),
            avg_time=("time_sec", "mean"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.case_ids) if args.case_ids else None
    case_specs = [case for case in CASE_SPECS if selected is None or int(case["case_id"]) in selected]

    frames = []
    for case_spec in case_specs:
        cfg = _make_cfg(case_spec)
        cfg.n = int(args.n)
        cfg.p_w = int(args.p_w)
        cfg.p_z = int(args.p_z)
        case = _build_case(cfg, case_spec)
        rows = []
        total_models = len(MODEL_SPECS)
        print(f"[case {case_spec['case_id']:02d}] start: {case_spec['slug']} with {total_models} models", flush=True)
        for model_index, (name, model_kwargs) in enumerate(MODEL_SPECS, start=1):
            print(f"[case {case_spec['case_id']:02d}] model {model_index}/{total_models}: {name}", flush=True)
            start = time.time()
            model = _BaseSinglePassBridgeFeatureNCCausalForest(**model_kwargs)
            model.fit_components(case["X"], case["A"], case["Y"], case["Z"], case["W"])
            preds = model.effect_from_components(case["X"], case["W"], case["Z"]).ravel()
            row = _metric_row(name, preds, case["true_cate"], time.time() - start)
            row["model_kwargs"] = repr(model_kwargs)
            rows.append(row)
            partial_case_df = pd.DataFrame(rows)
            partial_case_df.insert(0, "case_id", case_spec["case_id"])
            partial_case_df.insert(1, "case_slug", case_spec["slug"])
            partial_case_df.insert(2, "case_title", case_spec["title"])
            partial_case_df["n"] = cfg.n
            partial_case_df["p_x"] = cfg.p_x
            partial_case_df["p_w"] = cfg.p_w
            partial_case_df["p_z"] = cfg.p_z
            pd.concat(frames + [partial_case_df], ignore_index=True).to_csv(args.output_dir / f"results_{args.tag}.csv", index=False)

        case_df = pd.DataFrame(rows)
        case_df.insert(0, "case_id", case_spec["case_id"])
        case_df.insert(1, "case_slug", case_spec["slug"])
        case_df.insert(2, "case_title", case_spec["title"])
        case_df["n"] = cfg.n
        case_df["p_x"] = cfg.p_x
        case_df["p_w"] = cfg.p_w
        case_df["p_z"] = cfg.p_z
        frames.append(case_df)
        pd.concat(frames, ignore_index=True).to_csv(args.output_dir / f"results_{args.tag}.csv", index=False)
        print(f"[case {case_spec['case_id']:02d}] done", flush=True)

    results = pd.concat(frames, ignore_index=True).sort_values(["case_id", "name"]).reset_index(drop=True)
    results.to_csv(args.output_dir / f"results_{args.tag}.csv", index=False)
    summarize(results).to_csv(args.output_dir / f"summary_{args.tag}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
