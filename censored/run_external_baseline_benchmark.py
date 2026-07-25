#!/usr/bin/env python3
"""Run external HTE baselines on the paper's canonical censored simulation grid.

The output schema matches ``censored/run_5model_benchmark.py`` closely enough to append
the result rows to its ``results_full.csv``.  The runner evaluates subject-level treatment
effects against the DGP truth, so population-only AIPW/TMLE estimates intentionally do not
appear here; use ``python -m benchmarks.runner`` for the separate ATE/ATT comparison.

Examples
--------
Smoke test:
    python -m censored.run_external_baseline_benchmark \
        --case-ids 9 --setting-ids S02 --target both --replicates 1

Default external HTE suite:
    python -m censored.run_external_baseline_benchmark \
        --output-dir outputs/reruns/external_hte --replicates 20

BART on a small cell (use an interpreter containing pymc-bart):
    python -m censored.run_external_baseline_benchmark \
        --case-ids 9 --setting-ids S02 --methods bart \
        --bart-python /path/to/bart-env/bin/python --bart-subsample 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
MODEL_DIR = THIS_FILE.parent
PROJECT_ROOT = MODEL_DIR.parent
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
for path in (PROJECT_ROOT, PYTHON_PACKAGE_ROOT, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

METHOD_LABELS = {
    "rsf_t": "RSF T-Learner",
    "rsf_s": "RSF S-Learner",
    "r_csf": "Standard Causal Survival Forest",
    "bart": "Survival BART S-Learner",
}
DEFAULT_METHODS = ("rsf_t", "rsf_s", "r_csf")
RESULTS_FILE = "results_external_hte.csv"
BOOTSTRAP_FILE = "bootstrap_external_hte.csv"
FAILURES_FILE = "failures_external_hte.csv"
METADATA_FILE = "run_metadata_external_hte.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reruns" / "external_hte",
    )
    parser.add_argument("--case-ids", nargs="*", type=int, default=None)
    parser.add_argument("--case-slugs", nargs="*", default=None)
    parser.add_argument("--setting-ids", nargs="*", default=None)
    parser.add_argument(
        "--target",
        choices=["RMST", "survival.probability", "both"],
        default="both",
    )
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_LABELS), default=list(DEFAULT_METHODS))
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=50,
        help="Stratified bootstrap refits per cell (default: 50; BART uses posterior uncertainty).",
    )
    parser.add_argument("--rmst-horizon", type=float, default=3.0)
    parser.add_argument("--horizon-quantile", type=float, default=0.90)
    parser.add_argument(
        "--adjust",
        choices=["xwz", "x"],
        default="xwz",
        help="Use proxies as ordinary covariates (xwz, primary) or omit them (x, sensitivity).",
    )
    parser.add_argument("--num-trees", type=int, default=500, help="Trees for RSF-T and RSF-S.")
    parser.add_argument("--grf-num-trees", type=int, default=500)
    parser.add_argument("--grf-min-node-size", type=int, default=20)
    parser.add_argument("--bart-python", default=None)
    parser.add_argument("--bart-draws", type=int, default=500)
    parser.add_argument("--bart-tune", type=int, default=500)
    parser.add_argument("--bart-trees", type=int, default=50)
    parser.add_argument("--bart-max-bins", type=int, default=16)
    parser.add_argument(
        "--bart-subsample",
        type=int,
        default=0,
        help="0 evaluates the full cohort; a positive cap is faster but is labelled in output.",
    )
    parser.add_argument("--bart-timeout", type=int, default=21600)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-settings", action="store_true")
    return parser.parse_args()


def _cohort_frame(case) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    x = np.asarray(case.X, float)
    w = np.asarray(case.W, float)
    z = np.asarray(case.Z, float)
    if x.ndim == 1:
        x = x[:, None]
    if w.ndim == 1:
        w = w[:, None]
    if z.ndim == 1:
        z = z[:, None]
    cols = {
        "x": [f"X{j}" for j in range(x.shape[1])],
        "w": [f"W{j}" for j in range(w.shape[1])],
        "z": [f"Z{j}" for j in range(z.shape[1])],
    }
    df = pd.DataFrame(
        np.column_stack([x, w, z]),
        columns=cols["x"] + cols["w"] + cols["z"],
    )
    df["A"] = np.asarray(case.A, float)
    df["time"] = np.asarray(case.Y, float)
    df["event"] = np.asarray(case.delta, float)
    return df, cols


def _rsf_predictions(
    method: str,
    df: pd.DataFrame,
    cols: dict[str, list[str]],
    target: str,
    horizon: float,
    adjust: str,
    seed: int,
    num_trees: int,
    predict_df: pd.DataFrame | None = None,
) -> np.ndarray:
    # Lazy import keeps --help and the R-CSF-only path usable when scikit-survival is absent.
    from benchmarks import suite

    old_rsf = suite._rsf

    def configured_rsf(feat, y, model_seed, max_features="sqrt", n_estimators=500):
        return old_rsf(
            feat,
            y,
            model_seed,
            max_features=max_features,
            n_estimators=num_trees,
        )

    suite._rsf = configured_rsf
    try:
        fn = suite.predict_rsf_t_cate if method == "rsf_t" else suite.predict_rsf_s_cate
        unit = fn(df, cols, horizon, adjust=adjust, seed=seed, predict_df=predict_df)
    finally:
        suite._rsf = old_rsf
    return np.asarray(unit[target], float).ravel()


def _r_csf_predictions(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    cols: dict[str, list[str]],
    target: str,
    horizon: float,
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    model = GRFCensoredBaseline(
        target=target,
        horizon=float(horizon),
        random_state=seed,
        n_estimators=args.grf_num_trees,
        min_samples_leaf=args.grf_min_node_size,
    )
    x = train_df[cols["x"]].to_numpy(float)
    if args.adjust == "x":
        w = np.zeros((len(train_df), 0), dtype=float)
        z = np.zeros((len(train_df), 0), dtype=float)
    else:
        w = train_df[cols["w"]].to_numpy(float)
        z = train_df[cols["z"]].to_numpy(float)
    model.fit_components(
        x,
        train_df["A"].to_numpy(float),
        train_df["time"].to_numpy(float),
        train_df["event"].to_numpy(float),
        z,
        w,
    )
    xp = predict_df[cols["x"]].to_numpy(float)
    if args.adjust == "x":
        wp = np.zeros((len(predict_df), 0), dtype=float)
        zp = np.zeros((len(predict_df), 0), dtype=float)
    else:
        wp = predict_df[cols["w"]].to_numpy(float)
        zp = predict_df[cols["z"]].to_numpy(float)
    return np.asarray(model.effect_from_components(xp, wp, zp), float).ravel()


def _stratified_resample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Resample within the treatment-by-event cells, preserving all four cell sizes."""
    rng = np.random.default_rng(seed)
    treatment = df["A"].to_numpy(int)
    event = df["event"].to_numpy(int)
    pieces = []
    for a_value in (0, 1):
        for e_value in (0, 1):
            cell = np.flatnonzero((treatment == a_value) & (event == e_value))
            if cell.size:
                pieces.append(rng.choice(cell, size=cell.size, replace=True))
    if not pieces:
        index = rng.integers(0, len(df), size=len(df))
    else:
        index = np.concatenate(pieces)
        rng.shuffle(index)
    return df.iloc[index].reset_index(drop=True)


def _bootstrap_seed(
    seed: int,
    case_id: int,
    setting_id: str,
    target: str,
    bootstrap_id: int,
) -> int:
    """Stable seed shared by all methods in a cell, enabling paired comparisons."""
    setting_number = int(str(setting_id).lstrip("Ss"))
    target_offset = 0 if target == "RMST" else 50
    return int(seed * 1_000_000 + case_id * 10_000 + setting_number * 100 + target_offset + bootstrap_id)


def _attach_bootstrap_summary(
    row: dict[str, object],
    bootstrap_rows: list[dict[str, object]],
    n_bootstrap: int,
) -> dict[str, object]:
    out = dict(row)
    metrics = ("rmse", "mae", "bias", "abs_bias", "pearson_correlation", "sign_precision")
    for metric in metrics:
        values = np.asarray(
            [r.get(metric, np.nan) for r in bootstrap_rows],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        out[f"{metric}_boot_n"] = int(values.size)
        out[f"{metric}_boot_mean"] = float(np.mean(values)) if values.size else np.nan
        out[f"{metric}_se"] = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
        out[f"{metric}_ci_lo"] = float(np.percentile(values, 2.5)) if values.size else np.nan
        out[f"{metric}_ci_hi"] = float(np.percentile(values, 97.5)) if values.size else np.nan
    out["n_bootstrap_requested"] = int(n_bootstrap)
    out["uq"] = "stratified bootstrap"
    return out


def _bart_predictions(
    df: pd.DataFrame,
    cols: dict[str, list[str]],
    target: str,
    horizon: float,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    exe = args.bart_python
    if not exe:
        raise RuntimeError("--bart-python is required when --methods includes bart")
    bart_script = PROJECT_ROOT / "benchmarks" / "bart.py"
    with tempfile.TemporaryDirectory(prefix="nccsf_bart_") as tmp:
        tmp = Path(tmp)
        cohort_path = tmp / "cohort.csv"
        spec_path = tmp / "spec.json"
        result_path = tmp / "result.json"
        df.to_csv(cohort_path, index=False)
        spec_path.write_text(
            json.dumps(
                {
                    "x_cols": cols["x"],
                    "w_cols": cols["w"],
                    "z_cols": cols["z"],
                    "horizon": float(horizon),
                    "adjust": args.adjust,
                }
            ),
            encoding="utf-8",
        )
        cmd = [
            str(exe),
            "-u",
            str(bart_script),
            "--cohort",
            str(cohort_path),
            "--spec",
            str(spec_path),
            "--out",
            str(result_path),
            "--draws",
            str(args.bart_draws),
            "--tune",
            str(args.bart_tune),
            "--trees",
            str(args.bart_trees),
            "--max-bins",
            str(args.bart_max_bins),
            "--subsample",
            str(args.bart_subsample),
            "--seed",
            str(seed),
        ]
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=args.bart_timeout,
        )
        if proc.returncode != 0 or not result_path.exists():
            detail = (proc.stderr or proc.stdout or "")[-1000:]
            raise RuntimeError(f"BART subprocess failed (rc={proc.returncode}): {detail}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    pred = np.asarray(payload["unit_effects"][target], float)
    row_index = np.asarray(payload["row_index"], int)
    return pred, row_index


def _metadata_row(
    metrics: dict[str, object],
    *,
    method: str,
    case_spec: dict[str, object],
    setting: dict[str, object],
    target: str,
    replicate: int,
    seed: int,
    case,
    args: argparse.Namespace,
    eval_n: int,
) -> dict[str, object]:
    row = dict(metrics)
    if "pearson" in row:
        row["pearson_correlation"] = float(row.pop("pearson"))
    row.update(
        {
            "method": method,
            "name": METHOD_LABELS[method],
            "backend": str(row.get("backend", METHOD_LABELS[method])),
            "case_id": int(case_spec["case_id"]),
            "case_slug": str(case_spec["slug"]),
            "case_title": str(case_spec["title"]),
            "setting_id": str(setting["setting_id"]),
            "target": target,
            "replicate": int(replicate),
            "seed": int(seed),
            "n": int(setting["n"]),
            "eval_n": int(eval_n),
            "p_x": int(setting["p_x"]),
            "p_w": int(setting["p_w"]),
            "p_z": int(setting["p_z"]),
            "horizon": float(case.horizon),
            "adjust": args.adjust,
        }
    )
    return row


def _save(rows: list[dict[str, object]], path: Path, keys: list[str]) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=keys, keep="last")
        sort_cols = [c for c in ("case_id", "setting_id", "target", "replicate", "method") if c in frame]
        frame = frame.sort_values(sort_cols).reset_index(drop=True)
    frame.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    # Delay the paper stack until after argparse, so --help works before server dependencies
    # such as econml, scikit-survival, and R/grf have been installed.
    global CASE_SPECS, SETTINGS, prepare_case, _evaluate_predictions, GRFCensoredBaseline
    global _case_spec_with_setting, _resolve_targets, _selected_case_specs, _selected_settings
    from grf.benchmarks.econml_8variant import _evaluate_predictions, prepare_case
    from censored.grf_censored_baseline import GRFCensoredBaseline
    from censored.run_5model_benchmark import (
        CASE_SPECS,
        SETTINGS,
        _case_spec_with_setting,
        _resolve_targets,
        _selected_case_specs,
        _selected_settings,
    )

    if args.list_cases:
        for spec in CASE_SPECS:
            print(f"{int(spec['case_id']):2d}  {spec['slug']}")
        return 0
    if args.list_settings:
        for setting in SETTINGS:
            print(
                f"{setting['setting_id']}: n={setting['n']} p_x={setting['p_x']} "
                f"p_w={setting['p_w']} p_z={setting['p_z']}"
            )
        return 0
    if args.replicates < 1:
        raise ValueError("--replicates must be at least 1")
    if args.n_bootstrap < 0:
        raise ValueError("--n-bootstrap cannot be negative")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_FILE
    bootstrap_path = output_dir / BOOTSTRAP_FILE
    failures_path = output_dir / FAILURES_FILE
    metadata_path = output_dir / METADATA_FILE

    case_specs = _selected_case_specs(args.case_ids, args.case_slugs)
    settings = _selected_settings(args.setting_ids)
    targets = _resolve_targets(args.target)
    methods = list(dict.fromkeys(args.methods))

    key_cols = ["case_id", "setting_id", "target", "replicate", "method", "adjust"]
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    bootstrap_rows = pd.read_csv(bootstrap_path).to_dict("records") if bootstrap_path.exists() else []
    failures = pd.read_csv(failures_path).to_dict("records") if failures_path.exists() else []
    done = {
        tuple(row.get(c) for c in key_cols)
        for row in rows
    }
    bootstrap_key_cols = key_cols + ["bootstrap_id"]
    bootstrap_done = {
        tuple(row.get(c) for c in bootstrap_key_cols)
        for row in bootstrap_rows
    }
    metadata_path.write_text(
        json.dumps(
            {
                "methods": methods,
                "case_ids": [int(s["case_id"]) for s in case_specs],
                "setting_ids": [str(s["setting_id"]) for s in settings],
                "targets": targets,
                "replicates": args.replicates,
                "seed_start": args.seed_start,
                "n_bootstrap": args.n_bootstrap,
                "bootstrap": "stratified by (A,event); refit on resample; predict fixed original cohort",
                "adjust": args.adjust,
                "rmst_horizon": args.rmst_horizon,
                "horizon_quantile": args.horizon_quantile,
                "note": "AIPW/TMLE are population-only and intentionally excluded from HTE metrics.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total = len(case_specs) * len(settings) * len(targets) * args.replicates * len(methods)
    step = 0
    for replicate in range(args.replicates):
        seed = args.seed_start + replicate
        for case_spec in case_specs:
            for setting in settings:
                configured = _case_spec_with_setting(case_spec, setting)
                for target in targets:
                    if target == "RMST":
                        case = prepare_case(configured, target=target, horizon=args.rmst_horizon)
                    else:
                        case = prepare_case(
                            configured,
                            target=target,
                            horizon_quantile=args.horizon_quantile,
                        )
                    df, cols = _cohort_frame(case)
                    truth_full = np.asarray(case.true_cate, float).ravel()
                    for method in methods:
                        step += 1
                        key = (
                            int(case_spec["case_id"]),
                            str(setting["setting_id"]),
                            target,
                            replicate,
                            method,
                            args.adjust,
                        )
                        try:
                            point_row = next(
                                (
                                    row
                                    for row in rows
                                    if tuple(row.get(c) for c in key_cols) == key
                                ),
                                None,
                            )
                            if point_row is None:
                                started = time.time()
                                if method in {"rsf_t", "rsf_s"}:
                                    pred = _rsf_predictions(
                                        method,
                                        df,
                                        cols,
                                        target,
                                        float(case.horizon),
                                        args.adjust,
                                        seed,
                                        args.num_trees,
                                    )
                                    truth = truth_full
                                elif method == "r_csf":
                                    pred = _r_csf_predictions(
                                        df,
                                        df,
                                        cols,
                                        target,
                                        float(case.horizon),
                                        seed,
                                        args,
                                    )
                                    truth = truth_full
                                else:
                                    pred, row_index = _bart_predictions(
                                        df,
                                        cols,
                                        target,
                                        float(case.horizon),
                                        seed,
                                        args,
                                    )
                                    truth = truth_full[row_index]
                                metrics = _evaluate_predictions(
                                    METHOD_LABELS[method],
                                    pred,
                                    truth,
                                    time.time() - started,
                                    backend=METHOD_LABELS[method],
                                )
                                point_row = _metadata_row(
                                    metrics,
                                    method=method,
                                    case_spec=case_spec,
                                    setting=setting,
                                    target=target,
                                    replicate=replicate,
                                    seed=seed,
                                    case=case,
                                    args=args,
                                    eval_n=len(pred),
                                )
                                if method == "bart":
                                    point_row["uq"] = "posterior"
                                    point_row["n_bootstrap_requested"] = 0
                                rows.append(point_row)
                                done.add(key)
                                _save(rows, results_path, key_cols)
                            else:
                                print(f"[{step}/{total}] resume point {key}", flush=True)

                            cell_bootstrap_rows = [
                                row
                                for row in bootstrap_rows
                                if tuple(row.get(c) for c in key_cols) == key
                            ]
                            if method != "bart" and args.n_bootstrap:
                                for bootstrap_id in range(args.n_bootstrap):
                                    bootstrap_key = key + (bootstrap_id,)
                                    if bootstrap_key in bootstrap_done:
                                        continue
                                    resample_seed = _bootstrap_seed(
                                        seed,
                                        int(case_spec["case_id"]),
                                        str(setting["setting_id"]),
                                        target,
                                        bootstrap_id,
                                    )
                                    bootstrap_df = _stratified_resample(df, resample_seed)
                                    boot_started = time.time()
                                    if method in {"rsf_t", "rsf_s"}:
                                        boot_pred = _rsf_predictions(
                                            method,
                                            bootstrap_df,
                                            cols,
                                            target,
                                            float(case.horizon),
                                            args.adjust,
                                            seed + bootstrap_id + 1,
                                            args.num_trees,
                                            predict_df=df,
                                        )
                                    else:
                                        boot_pred = _r_csf_predictions(
                                            bootstrap_df,
                                            df,
                                            cols,
                                            target,
                                            float(case.horizon),
                                            seed + bootstrap_id + 1,
                                            args,
                                        )
                                    boot_metrics = _evaluate_predictions(
                                        METHOD_LABELS[method],
                                        boot_pred,
                                        truth_full,
                                        time.time() - boot_started,
                                        backend=METHOD_LABELS[method],
                                    )
                                    boot_row = _metadata_row(
                                        boot_metrics,
                                        method=method,
                                        case_spec=case_spec,
                                        setting=setting,
                                        target=target,
                                        replicate=replicate,
                                        seed=seed,
                                        case=case,
                                        args=args,
                                        eval_n=len(boot_pred),
                                    )
                                    boot_row["bootstrap_id"] = bootstrap_id
                                    boot_row["bootstrap_seed"] = resample_seed
                                    bootstrap_rows.append(boot_row)
                                    cell_bootstrap_rows.append(boot_row)
                                    bootstrap_done.add(bootstrap_key)
                                    _save(bootstrap_rows, bootstrap_path, bootstrap_key_cols)
                                    print(
                                        f"  bootstrap {bootstrap_id + 1}/{args.n_bootstrap} "
                                        f"method={METHOD_LABELS[method]} rmse={float(boot_row['rmse']):.4f}",
                                        flush=True,
                                    )

                                updated = _attach_bootstrap_summary(
                                    point_row,
                                    cell_bootstrap_rows,
                                    args.n_bootstrap,
                                )
                                rows = [
                                    updated if tuple(row.get(c) for c in key_cols) == key else row
                                    for row in rows
                                ]
                                point_row = updated
                                _save(rows, results_path, key_cols)

                            print(
                                f"[{step}/{total}] case={int(case_spec['case_id']):02d} "
                                f"setting={setting['setting_id']} target={target} rep={replicate} "
                                f"method={METHOD_LABELS[method]} rmse={float(point_row['rmse']):.4f} "
                                f"corr={float(point_row['pearson_correlation']):.4f} "
                                f"bootstrap={int(point_row.get('rmse_boot_n', 0))}",
                                flush=True,
                            )
                        except Exception as exc:
                            failure = {
                                "case_id": int(case_spec["case_id"]),
                                "case_slug": str(case_spec["slug"]),
                                "setting_id": str(setting["setting_id"]),
                                "target": target,
                                "replicate": replicate,
                                "seed": seed,
                                "method": method,
                                "adjust": args.adjust,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                            failures.append(failure)
                            _save(failures, failures_path, key_cols)
                            print(f"[{step}/{total}] FAILED {key}: {type(exc).__name__}: {exc}", flush=True)
                            if args.fail_fast:
                                raise

    if rows:
        result_df = pd.DataFrame(rows)
        group_cols = ["target", "method", "name"]
        summary = (
            result_df.groupby(group_cols, as_index=False)
            .agg(
                avg_rmse=("rmse", "mean"),
                sd_rmse=("rmse", "std"),
                avg_mae=("mae", "mean"),
                avg_abs_bias=("abs_bias", "mean"),
                avg_correlation=("pearson_correlation", "mean"),
                avg_sign_precision=("sign_precision", "mean"),
                cells=("rmse", "size"),
            )
            .sort_values(["target", "avg_rmse"])
        )
        summary.to_csv(output_dir / "summary_external_hte.csv", index=False)
        print(f"\nWrote {results_path}", flush=True)
        print(summary.to_string(index=False), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
