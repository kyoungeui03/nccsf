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
from grf.non_censored.models import (  # noqa: E402
    UnifiedB2SumBaselineNCCausalForest,
    UnifiedB2SumMildShrinkNCCausalForest,
    UnifiedB2SumSinglePassBaselineNCCausalForest,
    _BaseSinglePassBridgeFeatureNCCausalForest,
)


REFERENCE_MODELS = [
    ("UnifiedB2SumBaseline (NC)", UnifiedB2SumBaselineNCCausalForest, {}),
    ("UnifiedB2SumMildShrink (NC)", UnifiedB2SumMildShrinkNCCausalForest, {}),
    ("UnifiedB2SumSinglePassBaseline (NC)", UnifiedB2SumSinglePassBaselineNCCausalForest, {}),
]

Q_PROFILES = [
    ("q=logit", {"q_kind": "logit"}),
    ("q=poly2", {"q_kind": "poly2"}),
    ("q=rf100x20", {"q_kind": "rf", "q_trees": 100, "q_leaf": 20}),
]

H_PROFILES = [
    ("h=rf100x20", {"h_kind": "rf", "h_n_estimators": 100, "h_min_samples_leaf": 20}),
    ("h=extra200x10", {"h_kind": "extra", "h_n_estimators": 200, "h_min_samples_leaf": 10}),
    ("h=extra400x10", {"h_kind": "extra", "h_n_estimators": 400, "h_min_samples_leaf": 10}),
    ("h=extra600x5", {"h_kind": "extra", "h_n_estimators": 600, "h_min_samples_leaf": 5}),
    ("h=extra1200x3", {"h_kind": "extra", "h_n_estimators": 1200, "h_min_samples_leaf": 3}),
]

FINAL_FEATURE_MODES = [
    "aug_full",
    "aug_compact",
    "aug_compact_stats",
    "aug_compact_qcenter_agreement",
    "summary_compact",
    "summary_compact_stats",
    "summary_compact_qcenter_agreement",
]

CLIP_PROFILES = [
    ("clip=base", {"q_clip": 0.02, "y_clip_quantile": 0.99, "y_res_clip_percentiles": (1.0, 99.0)}),
    ("clip=strong", {"q_clip": 0.03, "y_clip_quantile": 0.98, "y_res_clip_percentiles": (2.0, 98.0)}),
]

FOREST_PROFILES = [
    ("forest=200x20", {"n_estimators": 200, "min_samples_leaf": 20}),
    ("forest=300x10", {"n_estimators": 300, "min_samples_leaf": 10}),
]


def build_model_specs() -> list[tuple[str, type, dict[str, object]]]:
    specs: list[tuple[str, type, dict[str, object]]] = list(REFERENCE_MODELS)
    for feature_mode in FINAL_FEATURE_MODES:
        for q_label, q_kwargs in Q_PROFILES:
            for h_label, h_kwargs in H_PROFILES:
                for clip_label, clip_kwargs in CLIP_PROFILES:
                    for forest_label, forest_kwargs in FOREST_PROFILES:
                        kwargs: dict[str, object] = {
                            "final_feature_mode": feature_mode,
                            "prediction_nuisance_mode": "full_refit",
                            "observed_only": False,
                            "cv": 5,
                            "random_state": 42,
                            "nuisance_feature_mode": "broad_dup",
                            "n_jobs": 1,
                        }
                        kwargs.update(q_kwargs)
                        kwargs.update(h_kwargs)
                        kwargs.update(clip_kwargs)
                        kwargs.update(forest_kwargs)
                        name = f"SPv2NC[{feature_mode}|{q_label}|{h_label}|{clip_label}|{forest_label}]"
                        specs.append((name, _BaseSinglePassBridgeFeatureNCCausalForest, kwargs))
    return specs


MODEL_SPECS = build_model_specs()


def parse_args():
    parser = argparse.ArgumentParser(description="Run basic12 non-censored single-pass v2 search.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--name-contains", type=str)
    parser.add_argument("--allow-names-file", type=Path)
    parser.add_argument("--max-models", type=int)
    parser.add_argument("--grid-only", action="store_true")
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
    model_specs = MODEL_SPECS
    if args.grid_only:
        model_specs = model_specs[len(REFERENCE_MODELS):]
    if args.name_contains:
        needle = args.name_contains.lower()
        model_specs = [spec for spec in model_specs if needle in spec[0].lower()]
    if args.allow_names_file:
        allowed = {
            line.strip()
            for line in args.allow_names_file.read_text().splitlines()
            if line.strip()
        }
        model_specs = [spec for spec in model_specs if spec[0] in allowed]
    if args.max_models is not None:
        model_specs = model_specs[: int(args.max_models)]

    frames = []
    for case_spec in case_specs:
        cfg = _make_cfg(case_spec)
        cfg.n = int(args.n)
        cfg.p_w = int(args.p_w)
        cfg.p_z = int(args.p_z)
        case = _build_case(cfg, case_spec)

        rows = []
        total_models = len(model_specs)
        print(f"[case {case_spec['case_id']:02d}] start: {case_spec['slug']} with {total_models} models", flush=True)
        for model_index, (name, model_cls, model_kwargs) in enumerate(model_specs, start=1):
            print(f"[case {case_spec['case_id']:02d}] model {model_index}/{total_models}: {name}", flush=True)
            start = time.time()
            model = model_cls(**model_kwargs)
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
    summary = summarize(results)
    summary.to_csv(args.output_dir / f"summary_{args.tag}.csv", index=False)
    summary.head(50).to_csv(args.output_dir / f"summary_top50_{args.tag}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
