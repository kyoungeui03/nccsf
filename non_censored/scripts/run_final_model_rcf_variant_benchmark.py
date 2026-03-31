#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import tempfile

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.non_censored.benchmarks import (  # noqa: E402
    CASE_SPECS,
    SUMMARY_COLUMNS,
    SUMMARY_KEYS,
    TABLE_COLUMNS,
    TABLE_KEYS,
    TOP5_COLUMNS,
    TOP5_KEYS,
    _build_case,
    _make_cfg,
    _metric_row,
    _render_table_png,
    nc_h_from_proxy,
    nc_q_from_proxy,
    oracle_h_from_proxy,
    oracle_q_from_proxy,
)
from grf.non_censored.models import (  # noqa: E402
    BaselineCausalForestDML,
    _ConfigurableNCNuisance,
    _build_oldc3_ablation_features,
    _ensure_2d,
    _predict_r_cf_final_forest,
    _train_r_cf_final_forest,
    make_h_model,
    make_q_model,
    SinglePassBridgeFeatureNCCausalForestDML,
)
from grf.r_runtime import resolve_rscript  # noqa: E402

matplotlib.use("Agg")


VARIANT_SPECS = [
    {"name": "A1  RCF Final Model Oracle (all true q/h)", "kind": "oracle", "use_true_q": True, "use_true_h": True},
    {"name": "A3  RCF Final Model Oracle (all estimated q/h)", "kind": "oracle", "use_true_q": False, "use_true_h": False},
    {"name": "EconML Baseline", "kind": "baseline"},
    {"name": "R-CF Baseline", "kind": "baseline_r"},
    {"name": "C3  Final Model (all estimated q/h)", "kind": "original_c3"},
    {"name": "C1  RCF Final Model (all true q/h)", "kind": "proxy", "use_true_q": True, "use_true_h": True},
    {"name": "C3  RCF Final Model (all estimated q/h)", "kind": "proxy", "use_true_q": False, "use_true_h": False},
]

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

R_CF_SCRIPT = PROJECT_ROOT / "scripts" / "run_grf_cf_baseline.R"


@dataclass
class _RCFVariantOwner:
    cv: int
    random_state: int
    x_core_dim: int
    q_model_template: object
    h_model_template: object
    q_clip: float
    y_clip_quantile: float | None
    y_res_clip_percentiles: tuple[float, float]
    nuisance_feature_mode: str
    oracle: bool
    use_true_q: bool
    use_true_h: bool
    q_true_fn: object
    h_true_fn: object
    include_raw_proxy: bool = True
    include_bridge_stats: bool = False
    extra_bridge_features: tuple[str, ...] = ()

    def _make_nuisance(self):
        return _ConfigurableNCNuisance(
            q_model=self.q_model_template,
            h_model=self.h_model_template,
            q_clip=self.q_clip,
            y_clip_quantile=self.y_clip_quantile,
            y_res_clip_percentiles=self.y_res_clip_percentiles,
            x_core_dim=self.x_core_dim,
            duplicate_proxies_in_nuisance=True,
            nuisance_feature_mode=self.nuisance_feature_mode,
            oracle=self.oracle,
            use_true_q=self.use_true_q,
            use_true_h=self.use_true_h,
            q_true_fn=self.q_true_fn,
            h_true_fn=self.h_true_fn,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the non-censored R grf final-forest variant benchmark for basic12 or structured14."
    )
    parser.add_argument("--suite", choices=["basic12", "structured14"], default="basic12")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--setting-ids", nargs="*")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p-x", type=int, default=None)
    parser.add_argument("--p-w", type=int, default=1)
    parser.add_argument("--p-z", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


def _resolve_output_dir(args) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    if args.suite == "structured14":
        return (PROJECT_ROOT / "non_censored" / "outputs" / "benchmark_structured14_final_model_rcf_variants").resolve()
    return (PROJECT_ROOT / "non_censored" / "outputs" / "benchmark_final_model_rcf_variants_12case").resolve()


def _case_with_overrides(case_spec, *, n=None, p_x=None, p_w=None, p_z=None):
    case_copy = dict(case_spec)
    cfg_updates = dict(case_spec["cfg"])
    if n is not None:
        cfg_updates["n"] = int(n)
    if p_x is not None:
        cfg_updates["p_x"] = int(p_x)
    if p_w is not None:
        cfg_updates["p_w"] = int(p_w)
    if p_z is not None:
        cfg_updates["p_z"] = int(p_z)
    case_copy["cfg"] = cfg_updates
    return case_copy


def _setting_slug(setting: dict[str, int | str]) -> str:
    return (
        f"{setting['setting_id']}_n{setting['n']}_px{setting['p_x']}"
        f"_pw{setting['p_w']}_pz{setting['p_z']}"
    )


def _format_case_title(case_spec, cfg):
    base = str(case_spec["title"]).split(", n=", 1)[0]
    return (
        f"{base}, n={cfg.n}, p_x={cfg.p_x}, p_w={cfg.p_w}, "
        f"p_z={cfg.p_z}, seed={cfg.seed}, censoring rate=0%"
    )


def _summarize(combined_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            avg_time=("time_sec", "mean"),
        )
        .sort_values(["avg_rmse", "avg_mae", "avg_pearson"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary, summary.head(5).copy()


def _build_owner(
    *,
    x_core_dim: int,
    seed: int,
    oracle: bool,
    use_true_q: bool,
    use_true_h: bool,
    q_true_fn,
    h_true_fn,
) -> _RCFVariantOwner:
    return _RCFVariantOwner(
        cv=5,
        random_state=int(seed),
        x_core_dim=int(x_core_dim),
        q_model_template=make_q_model("logit", random_state=seed, n_estimators=300, min_samples_leaf=20),
        h_model_template=make_h_model(
            "extra",
            random_state=seed,
            n_estimators=600,
            min_samples_leaf=5,
            n_jobs=1,
        ),
        q_clip=0.02,
        y_clip_quantile=0.99,
        y_res_clip_percentiles=(1.0, 99.0),
        nuisance_feature_mode="broad_dup",
        oracle=bool(oracle),
        use_true_q=bool(use_true_q),
        use_true_h=bool(use_true_h),
        q_true_fn=q_true_fn,
        h_true_fn=h_true_fn,
    )


def _make_original_final_model_kwargs(
    x_core_dim: int,
    seed: int,
    *,
    oracle: bool,
    use_true_q: bool,
    use_true_h: bool,
    q_true_fn,
    h_true_fn,
):
    return dict(
        final_feature_mode="aug_full",
        n_estimators=200,
        min_samples_leaf=20,
        cv=5,
        random_state=seed,
        q_kind="logit",
        h_kind="extra",
        h_n_estimators=600,
        h_min_samples_leaf=5,
        q_clip=0.02,
        y_clip_quantile=0.99,
        y_res_clip_percentiles=(1.0, 99.0),
        n_jobs=1,
        x_core_dim=x_core_dim,
        duplicate_proxies_in_nuisance=True,
        nuisance_feature_mode="broad_dup",
        oracle=oracle,
        use_true_q=use_true_q,
        use_true_h=use_true_h,
        q_true_fn=q_true_fn,
        h_true_fn=h_true_fn,
    )


def _crossfit_rcf_variant(
    owner: _RCFVariantOwner,
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    *,
    raw_w: np.ndarray,
    raw_z: np.ndarray,
    nuisance_w,
    nuisance_z,
):
    x = np.asarray(X, dtype=float)
    a = np.asarray(A, dtype=float).ravel()
    y = np.asarray(Y, dtype=float).ravel()
    raw_w = np.asarray(raw_w, dtype=float)
    raw_z = np.asarray(raw_z, dtype=float)

    splitter = KFold(n_splits=owner.cv, shuffle=True, random_state=owner.random_state)
    x_final = None
    y_res = np.empty_like(y, dtype=float)
    a_res = np.empty_like(y, dtype=float)

    for train_idx, test_idx in splitter.split(x):
        nuisance = owner._make_nuisance()
        nuisance.train(
            False,
            None,
            y[train_idx],
            a[train_idx],
            X=x[train_idx],
            W=None if nuisance_w is None else nuisance_w[train_idx],
            Z=None if nuisance_z is None else nuisance_z[train_idx],
        )
        y_res_fold, a_res_fold = nuisance.predict(
            y[test_idx],
            a[test_idx],
            X=x[test_idx],
            W=None if nuisance_w is None else nuisance_w[test_idx],
            Z=None if nuisance_z is None else nuisance_z[test_idx],
        )
        bridge = nuisance.predict_bridge_outputs(
            X=x[test_idx],
            W=None if nuisance_w is None else nuisance_w[test_idx],
            Z=None if nuisance_z is None else nuisance_z[test_idx],
        )
        x_final_fold = _build_oldc3_ablation_features(
            x[test_idx],
            raw_w[test_idx],
            raw_z[test_idx],
            bridge,
            include_raw_proxy=owner.include_raw_proxy,
            include_bridge_stats=owner.include_bridge_stats,
            extra_bridge_features=owner.extra_bridge_features,
        )
        if x_final is None:
            x_final = np.empty((len(y), x_final_fold.shape[1]), dtype=float)
        x_final[test_idx] = x_final_fold
        y_res[test_idx] = np.asarray(y_res_fold, dtype=float).ravel()
        a_res[test_idx] = np.asarray(a_res_fold, dtype=float).ravel()

    return x_final, y_res, a_res


def _fit_and_predict_rcf(x_final, y, a, y_res, a_res, *, seed: int):
    tempdir, model_path, feature_cols = _train_r_cf_final_forest(
        x_final,
        y,
        a,
        np.asarray(y, dtype=float).ravel() - np.asarray(y_res, dtype=float).ravel(),
        np.asarray(a, dtype=float).ravel() - np.asarray(a_res, dtype=float).ravel(),
        num_trees=200,
        min_node_size=20,
        seed=seed,
    )
    try:
        return _predict_r_cf_final_forest(model_path, x_final, feature_cols=feature_cols)
    finally:
        tempdir.cleanup()


def _run_r_cf_baseline(obs_df: pd.DataFrame, feature_cols: list[str], *, num_trees: int, seed: int) -> tuple[np.ndarray, float]:
    with tempfile.TemporaryDirectory(prefix="r_cf_variant_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        input_path = tmp_dir_path / "input.csv"
        output_path = tmp_dir_path / "predictions.csv"
        obs_df.to_csv(input_path, index=False)
        cmd = [
            resolve_rscript(),
            str(R_CF_SCRIPT),
            str(input_path),
            ",".join(feature_cols),
            str(int(num_trees)),
            str(output_path),
            str(int(seed)),
        ]
        start = time.time()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        elapsed = time.time() - start
        if proc.returncode != 0:
            raise RuntimeError(
                "R causal_forest baseline failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        preds = pd.read_csv(output_path)["pred"].to_numpy(dtype=float)
    return preds, elapsed


def _evaluate_baseline(case):
    x_full = np.hstack([case["X"], case["W"], case["Z"]])
    start = time.time()
    model = BaselineCausalForestDML(n_estimators=200, min_samples_leaf=20, cv=5, random_state=42)
    model.fit_baseline(x_full, case["A"], case["Y"])
    preds = model.effect(x_full).ravel()
    return _metric_row("EconML Baseline", preds, case["true_cate"], time.time() - start)


def _evaluate_r_cf_baseline(case):
    feature_cols = [*case["x_cols"], *case["w_cols"], *case["z_cols"]]
    obs_df = case["obs_df"].loc[:, [*feature_cols, "A"]].copy()
    obs_df["Y"] = case["Y"]
    preds, elapsed = _run_r_cf_baseline(
        obs_df,
        feature_cols,
        num_trees=200,
        seed=int(case["cfg"].seed),
    )
    return _metric_row("R-CF Baseline", preds, case["true_cate"], elapsed)


def _evaluate_original_c3(case):
    x = np.asarray(case["X"], dtype=float)
    w = np.asarray(case["W"], dtype=float)
    z = np.asarray(case["Z"], dtype=float)
    model = SinglePassBridgeFeatureNCCausalForestDML(
        **_make_original_final_model_kwargs(
            x_core_dim=x.shape[1],
            seed=case["cfg"].seed,
            oracle=False,
            use_true_q=False,
            use_true_h=False,
            q_true_fn=partial(nc_q_from_proxy, dgp=case["dgp"], cfg=case["cfg"]),
            h_true_fn=partial(nc_h_from_proxy, cfg=case["cfg"], dgp=case["dgp"]),
        )
    )
    model._raw_w_for_final = w
    model._raw_z_for_final = z
    start = time.time()
    model.fit_nc(x, case["A"], case["Y"], z, w)
    preds = model.effect_on_final_features(model.training_x_final()).ravel()
    return _metric_row("C3  Final Model (all estimated q/h)", preds, case["true_cate"], time.time() - start)


def _evaluate_oracle_variant(case, variant):
    x = np.asarray(case["X"], dtype=float)
    y = np.asarray(case["Y"], dtype=float)
    a = np.asarray(case["A"], dtype=float)
    u = np.asarray(case["U"], dtype=float).reshape(-1, 1)
    x_oracle = np.hstack([x, u])
    raw_w = np.zeros((len(x), 0), dtype=float)
    raw_z = np.zeros((len(x), 0), dtype=float)
    owner = _build_owner(
        x_core_dim=x.shape[1],
        seed=case["cfg"].seed,
        oracle=True,
        use_true_q=variant["use_true_q"],
        use_true_h=variant["use_true_h"],
        q_true_fn=partial(oracle_q_from_proxy, dgp=case["dgp"], cfg=case["cfg"]),
        h_true_fn=partial(oracle_h_from_proxy, cfg=case["cfg"], dgp=case["dgp"]),
    )
    start = time.time()
    x_final, y_res, a_res = _crossfit_rcf_variant(
        owner,
        x_oracle,
        a,
        y,
        raw_w=raw_w,
        raw_z=raw_z,
        nuisance_w=u,
        nuisance_z=None,
    )
    preds = _fit_and_predict_rcf(x_final, y, a, y_res, a_res, seed=int(case["cfg"].seed))
    return _metric_row(variant["name"], preds, case["true_cate"], time.time() - start)


def _evaluate_proxy_variant(case, variant):
    x = np.asarray(case["X"], dtype=float)
    w = _ensure_2d(np.asarray(case["W"], dtype=float))
    z = _ensure_2d(np.asarray(case["Z"], dtype=float))
    y = np.asarray(case["Y"], dtype=float)
    a = np.asarray(case["A"], dtype=float)
    owner = _build_owner(
        x_core_dim=x.shape[1],
        seed=case["cfg"].seed,
        oracle=False,
        use_true_q=variant["use_true_q"],
        use_true_h=variant["use_true_h"],
        q_true_fn=partial(nc_q_from_proxy, dgp=case["dgp"], cfg=case["cfg"]),
        h_true_fn=partial(nc_h_from_proxy, cfg=case["cfg"], dgp=case["dgp"]),
    )
    start = time.time()
    x_final, y_res, a_res = _crossfit_rcf_variant(
        owner,
        x,
        a,
        y,
        raw_w=w,
        raw_z=z,
        nuisance_w=w,
        nuisance_z=z,
    )
    preds = _fit_and_predict_rcf(x_final, y, a, y_res, a_res, seed=int(case["cfg"].seed))
    return _metric_row(variant["name"], preds, case["true_cate"], time.time() - start)


def _evaluate_variant(case, variant):
    if variant["kind"] == "baseline":
        return _evaluate_baseline(case)
    if variant["kind"] == "baseline_r":
        return _evaluate_r_cf_baseline(case)
    if variant["kind"] == "original_c3":
        return _evaluate_original_c3(case)
    if variant["kind"] == "oracle":
        return _evaluate_oracle_variant(case, variant)
    if variant["kind"] == "proxy":
        return _evaluate_proxy_variant(case, variant)
    raise ValueError(f"Unsupported variant kind: {variant['kind']}")


def _run_basic12(case_specs: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_frames = []
    for case_spec in case_specs:
        cfg = _make_cfg(case_spec)
        case = _build_case(cfg, case_spec)
        case_title = _format_case_title(case_spec, cfg)

        rows = [_evaluate_variant(case, variant) for variant in VARIANT_SPECS]
        case_df = pd.DataFrame(rows)
        case_df.insert(0, "case_id", case_spec["case_id"])
        case_df.insert(1, "case_slug", case_spec["slug"])
        case_df.insert(2, "case_title", case_title)
        case_frames.append(case_df)

        case_csv = output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.csv"
        case_png = output_dir / f"case_{case_spec['case_id']:02d}_{case_spec['slug']}.png"
        case_df.to_csv(case_csv, index=False)
        _render_table_png(
            case_title,
            case_df.to_dict("records"),
            case_png,
            TABLE_COLUMNS,
            TABLE_KEYS,
            meta="RCF Final Model variant benchmark",
        )

    combined_df = pd.concat(case_frames, ignore_index=True)
    summary_df, top5_df = _summarize(combined_df)

    combined_df.to_csv(output_dir / "all_12case_final_model_rcf_variant_results.csv", index=False)
    summary_df.to_csv(output_dir / "all_12case_final_model_rcf_variant_summary.csv", index=False)
    top5_df.to_csv(output_dir / "all_12case_final_model_rcf_variant_top5.csv", index=False)

    _render_table_png(
        "Non-censored RCF Final Model variant benchmark average summary",
        summary_df.to_dict("records"),
        output_dir / "all_12case_final_model_rcf_variant_summary.png",
        SUMMARY_COLUMNS,
        SUMMARY_KEYS,
        dark=True,
        meta="A1/A3/C1/C3 with R grf final forest + original C3 + baselines",
        col_widths=[90, 520, 150, 150, 120, 140, 140, 120, 150, 120],
    )
    _render_table_png(
        "Non-censored RCF Final Model variant benchmark top 5",
        top5_df.to_dict("records"),
        output_dir / "all_12case_final_model_rcf_variant_top5.png",
        TOP5_COLUMNS,
        TOP5_KEYS,
        dark=True,
        meta="Top 5 by avg RMSE",
        col_widths=[90, 560, 140, 140, 140, 160, 120],
    )


def _consolidate_structured_outputs(output_dir: Path) -> None:
    frames: list[pd.DataFrame] = []
    for setting in SETTINGS:
        path = output_dir / _setting_slug(setting) / "results.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    summary_df, top5_df = _summarize(combined)
    combined.to_csv(output_dir / "all_settings_results.csv", index=False)
    summary_df.to_csv(output_dir / "all_settings_summary.csv", index=False)
    top5_df.to_csv(output_dir / "all_settings_top5.csv", index=False)


def _run_structured14(
    settings: list[dict[str, int | str]],
    *,
    case_ids: set[int] | None,
    output_dir: Path,
    skip_existing: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for setting in settings:
        setting_dir = output_dir / _setting_slug(setting)
        results_path = setting_dir / "results.csv"
        summary_path = setting_dir / "summary.csv"
        if skip_existing and results_path.exists() and summary_path.exists():
            continue

        setting_dir.mkdir(parents=True, exist_ok=True)
        case_frames = []
        for raw_case_spec in CASE_SPECS:
            if case_ids is not None and int(raw_case_spec["case_id"]) not in case_ids:
                continue
            case_spec = _case_with_overrides(
                raw_case_spec,
                n=setting["n"],
                p_x=setting["p_x"],
                p_w=setting["p_w"],
                p_z=setting["p_z"],
            )
            cfg = _make_cfg(case_spec)
            case = _build_case(cfg, case_spec)
            case_title = _format_case_title(case_spec, cfg)

            rows = [_evaluate_variant(case, variant) for variant in VARIANT_SPECS]
            case_df = pd.DataFrame(rows)
            case_df.insert(0, "setting_id", str(setting["setting_id"]))
            case_df.insert(1, "setting_slug_full", _setting_slug(setting))
            case_df.insert(2, "case_id", case_spec["case_id"])
            case_df.insert(3, "case_slug", case_spec["slug"])
            case_df.insert(4, "case_title", case_title)
            case_df["n"] = cfg.n
            case_df["p_x"] = cfg.p_x
            case_df["p_w"] = cfg.p_w
            case_df["p_z"] = cfg.p_z
            case_frames.append(case_df)

        results = pd.concat(case_frames, ignore_index=True)
        summary_df, top5_df = _summarize(results)
        results.to_csv(results_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        top5_df.to_csv(setting_dir / "top5.csv", index=False)
        _consolidate_structured_outputs(output_dir)

    _consolidate_structured_outputs(output_dir)


def main() -> int:
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    selected_case_ids = set(args.case_ids) if args.case_ids else None

    if args.suite == "basic12":
        case_specs = [
            _case_with_overrides(case, n=args.n, p_x=args.p_x, p_w=args.p_w, p_z=args.p_z)
            for case in CASE_SPECS
            if selected_case_ids is None or int(case["case_id"]) in selected_case_ids
        ]
        _run_basic12(case_specs, output_dir)
        return 0

    selected_setting_ids = set(args.setting_ids) if args.setting_ids else None
    settings = [s for s in SETTINGS if selected_setting_ids is None or s["setting_id"] in selected_setting_ids]
    _run_structured14(settings, case_ids=selected_case_ids, output_dir=output_dir, skip_existing=args.skip_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
