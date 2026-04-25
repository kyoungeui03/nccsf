#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))
ROOT_SCRIPTS = PROJECT_ROOT / "scripts"
if str(ROOT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _direct_grf_imports import import_grf_module  # noqa: E402
from paper_csf_dgp import (  # noqa: E402
    SETTINGS,
    _classification_mask,
    _make_obs_df,
    _sample_dataset,
    _target_horizon,
    _true_cate,
)

_censored_methods = import_grf_module(PROJECT_ROOT, "grf.methods.econml_oldc3_ablation_survival")
_r_runtime = import_grf_module(PROJECT_ROOT, "grf.r_runtime")
from single_file_censored_models.final_censored_model_conditional import (  # noqa: E402
    FinalModelConditionalCensoredSurvivalForest,
)

FinalModelNoPCICensoredSurvivalForest = _censored_methods.FinalModelNoPCICensoredSurvivalForest
FinalModelRawCensoredSurvivalForest = _censored_methods.FinalModelRawCensoredSurvivalForest
resolve_rscript = _r_runtime.resolve_rscript

R_CSF_SCRIPT = PROJECT_ROOT / "Archive" / "run_paper_grf_csf_direct.R"

MODEL_SPECS = {
    "final": {"name": "Final Conditional Model", "kind": "python", "cls": FinalModelConditionalCensoredSurvivalForest},
    "no-pci": {"name": "Final Model (No PCI)", "kind": "python", "cls": FinalModelNoPCICensoredSurvivalForest},
    "raw": {"name": "Final Model (Raw)", "kind": "python", "cls": FinalModelRawCensoredSurvivalForest},
    "r-csf": {"name": "R-CSF Baseline", "kind": "r_grf", "cls": None},
    "compare-final-rcsf": {"name": "Final vs R-CSF", "kind": "comparison", "cls": None},
}

TARGET_LABELS = {
    "RMST": "RMST",
    "survival.probability": "Survival Probability",
}

PROXY_LAYOUTS = {
    "zero": "X=observed covariates, W=0, Z=0",
    "split-5-5-5": "X=5, W=5, Z=5",
}

PROTOCOLS = {
    "archive-aligned": "Archive-aligned horizons (survival.probability uses RMST horizons).",
    "paper-exact": "Paper-exact setup (p=15 and original survival-probability thresholds).",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the censored Final Model on the paper-style CSF simulation settings and "
            "report test-set CATE MSE."
        )
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="final")
    parser.add_argument("--settings", nargs="*", type=int, choices=sorted(SETTINGS), default=sorted(SETTINGS))
    parser.add_argument("--target", choices=["RMST", "survival.probability", "both"], default="both")
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), default="paper-exact")
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proxy-dim-w", type=int, default=1)
    parser.add_argument("--proxy-dim-z", type=int, default=1)
    parser.add_argument("--proxy-layout", choices=sorted(PROXY_LAYOUTS), default="zero")
    parser.add_argument("--num-trees-r", type=int, default=None)
    parser.add_argument("--final-num-trees", type=int, default=None)
    parser.add_argument("--final-min-node-size", type=int, default=None)
    parser.add_argument("--h-trees", type=int, default=None)
    parser.add_argument("--h-min-node-size", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10, help="Save partial results every N completed replications.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print a progress update every N completed replications.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from an existing replication_results.csv in the output directory if present.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore any existing replication_results.csv and start the run from scratch.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "paper_final_model_simulation",
    )
    return parser.parse_args()


def _make_zero_proxy(n: int, p: int) -> np.ndarray:
    return np.zeros((int(n), int(p)), dtype=float)


def _prepare_model_inputs(
    x_full: np.ndarray,
    *,
    proxy_layout: str,
    proxy_dim_w: int,
    proxy_dim_z: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build model inputs for the archive experiment.

    zero:
        Keep all observed covariates in X and provide zero-valued proxy slots.
    split-5-5-5:
        Manually partition the 15 observed covariates into X/W/Z blocks. This is
        a structural ablation, not a true proximal proxy construction.
    """

    x_full = np.asarray(x_full, dtype=float)
    n, p = x_full.shape

    if proxy_layout == "zero":
        return x_full, _make_zero_proxy(n, proxy_dim_w), _make_zero_proxy(n, proxy_dim_z)

    if proxy_layout == "split-5-5-5":
        if p != 15:
            raise ValueError(
                "proxy-layout='split-5-5-5' requires p=15 so the observed covariates can "
                "be partitioned exactly into X=5, W=5, Z=5."
            )
        x_main = x_full[:, 0:5]
        w = x_full[:, 5:10]
        z = x_full[:, 10:15]
        return x_main, w, z

    raise ValueError(f"Unsupported proxy layout: {proxy_layout}")


def _evaluate_replication(
    *,
    model_name: str,
    model_kind: str,
    model_cls,
    setting_id: int,
    target: str,
    n_train: int,
    n_test: int,
    p: int,
    seed: int,
    proxy_dim_w: int,
    proxy_dim_z: int,
    proxy_layout: str,
    protocol: str,
    num_trees_r: int,
    final_num_trees: int,
    final_min_node_size: int,
    h_trees: int,
    h_min_node_size: int,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    x_train_full, a_train, y_train, d_train = _sample_dataset(setting_id, n_train, p, rng, protocol=protocol)
    x_test_full, _, _, _ = _sample_dataset(setting_id, n_test, p, rng, protocol=protocol)

    horizon = _target_horizon(setting_id, target, protocol=protocol)

    true_cate = _true_cate(setting_id, x_test_full, target=target, horizon=horizon)
    x_train, w_train, z_train = _prepare_model_inputs(
        x_train_full,
        proxy_layout=proxy_layout,
        proxy_dim_w=proxy_dim_w,
        proxy_dim_z=proxy_dim_z,
    )
    x_test, w_test, z_test = _prepare_model_inputs(
        x_test_full,
        proxy_layout=proxy_layout,
        proxy_dim_w=proxy_dim_w,
        proxy_dim_z=proxy_dim_z,
    )

    t0 = time.time()
    if model_kind == "r_grf":
        preds = _predict_r_csf_baseline(
            x_train_full,
            a_train,
            y_train,
            d_train,
            x_test=x_test_full,
            horizon=horizon,
            target=target,
            num_trees=num_trees_r,
            seed=seed,
        )
    else:
        model_kwargs = dict(target=target, horizon=horizon, random_state=seed)
        if model_cls is FinalModelConditionalCensoredSurvivalForest:
            # Be explicit here so the archive experiment always uses the current
            # single-file Final Conditional model with the full learned summary
            # representation at the final stage.
            model_kwargs["surv_scalar_mode"] = "full"
        model_kwargs["n_estimators"] = int(final_num_trees)
        model_kwargs["min_samples_leaf"] = int(final_min_node_size)
        model_kwargs["h_n_estimators"] = int(h_trees)
        model_kwargs["h_min_samples_leaf"] = int(h_min_node_size)
        model = model_cls(**model_kwargs)
        model.fit_components(x_train, a_train, y_train, d_train, z_train, w_train)
        preds = np.asarray(model.effect_from_components(x_test, w_test, z_test), dtype=float).ravel()
    fit_time = time.time() - t0

    mask = _classification_mask(setting_id, x_test_full)
    sign_acc = float(np.mean(np.sign(preds[mask]) == np.sign(true_cate[mask])))
    mse = float(np.mean(np.square(preds - true_cate)))
    return {
        "model": model_name,
        "proxy_layout": proxy_layout,
        "proxy_layout_label": PROXY_LAYOUTS[proxy_layout],
        "protocol": protocol,
        "setting_id": int(setting_id),
        "target": target,
        "rep_seed": int(seed),
        "horizon": float(horizon),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "proxy_dim_w": int(proxy_dim_w),
        "proxy_dim_z": int(proxy_dim_z),
        "mse": mse,
        "mse_x100": float(100.0 * mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(preds - true_cate))),
        "sign_acc": sign_acc,
        "class_error": float(1.0 - sign_acc),
        "mean_pred": float(np.mean(preds)),
        "mean_true": float(np.mean(true_cate)),
        "bias": float(np.mean(preds - true_cate)),
        "pearson": float(np.corrcoef(preds, true_cate)[0, 1]),
        "fit_time_sec": float(fit_time),
    }


def _predict_r_csf_baseline(
    x_train: np.ndarray,
    a_train: np.ndarray,
    y_train: np.ndarray,
    d_train: np.ndarray,
    *,
    x_test: np.ndarray,
    horizon: float,
    target: str,
    num_trees: int,
    seed: int,
) -> np.ndarray:
    """Call installed R grf::causal_survival_forest directly from the archive experiment."""

    train_obs_df, feature_cols = _make_obs_df(x_train, a_train, y_train, d_train)
    test_df = pd.DataFrame(np.asarray(x_test, dtype=float), columns=feature_cols)
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs", prefix="archive_r_csf_surv_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        train_path = tmp_dir / "train.csv"
        test_path = tmp_dir / "test.csv"
        preds_path = tmp_dir / "predictions.csv"
        train_obs_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        cmd = [
            resolve_rscript(),
            str(R_CSF_SCRIPT),
            str(train_path),
            str(test_path),
            ",".join(feature_cols),
            str(target),
            str(float(horizon)),
            str(int(num_trees)),
            str(preds_path),
            str(int(seed)),
        ]
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                "Installed R grf baseline failed in archive experiment.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        preds = pd.read_csv(preds_path)["prediction"].to_numpy(dtype=float)
    return preds


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["model", "proxy_layout", "proxy_layout_label", "protocol", "setting_id", "target"], as_index=False)
        .agg(
            reps=("rep_seed", "count"),
            mean_horizon=("horizon", "mean"),
            mse=("mse", "mean"),
            mse_sd=("mse", "std"),
            mse_x100=("mse_x100", "mean"),
            mse_x100_sd=("mse_x100", "std"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            class_error=("class_error", "mean"),
            sign_acc=("sign_acc", "mean"),
            bias=("bias", "mean"),
            pearson=("pearson", "mean"),
            fit_time_sec=("fit_time_sec", "mean"),
            excess_mse=("excess_mse_rep", "mean"),
        )
        .sort_values(["target", "setting_id"])
        .reset_index(drop=True)
    )


def _attach_repwise_excess_mse(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    best_rep_mse = results.groupby(
        ["protocol", "proxy_layout", "setting_id", "target", "rep_seed"]
    )["mse_x100"].transform("min")
    results["excess_mse_rep"] = results["mse_x100"] / best_rep_mse
    return results


def _build_paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    mse = summary.pivot(index="setting_id", columns="target", values="mse_x100")
    class_error = summary.pivot(index="setting_id", columns="target", values="class_error")
    pearson = summary.pivot(index="setting_id", columns="target", values="pearson")
    table = pd.DataFrame({"Setting": sorted(summary["setting_id"].unique())})
    table["RMST MSE x100"] = table["Setting"].map(mse.get("RMST", pd.Series(dtype=float)))
    table["Survival Probability MSE x100"] = table["Setting"].map(mse.get("survival.probability", pd.Series(dtype=float)))
    table["RMST Class Error"] = table["Setting"].map(class_error.get("RMST", pd.Series(dtype=float)))
    table["Survival Probability Class Error"] = table["Setting"].map(class_error.get("survival.probability", pd.Series(dtype=float)))
    table["RMST Pearson"] = table["Setting"].map(pearson.get("RMST", pd.Series(dtype=float)))
    table["Survival Probability Pearson"] = table["Setting"].map(pearson.get("survival.probability", pd.Series(dtype=float)))
    return table


def _build_compare_table1(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    targets = ["RMST", "survival.probability"]
    models = ["Final Conditional Model", "R-CSF Baseline"]
    labels = {"Final Conditional Model": "Final", "R-CSF Baseline": "CSF"}
    for target in targets:
        panel = "Panel A: RMST" if target == "RMST" else "Panel B: Survival probability"
        sub = summary[summary["target"] == target]
        for setting_id in sorted(sub["setting_id"].unique()):
            row_mse: dict[str, float | int | str] = {"Panel": panel, "Setting": int(setting_id), "Metric": "MSE"}
            row_excess: dict[str, float | int | str] = {"Panel": "", "Setting": "", "Metric": "Excess MSE"}
            for model in models:
                model_rows = sub[(sub["setting_id"] == setting_id) & (sub["model"] == model)]
                if model_rows.empty:
                    row_mse[labels[model]] = np.nan
                    row_excess[labels[model]] = np.nan
                else:
                    row_mse[labels[model]] = float(model_rows["mse_x100"].iloc[0])
                    row_excess[labels[model]] = float(model_rows["excess_mse"].iloc[0])
            rows.extend([row_mse, row_excess])
    return pd.DataFrame(rows)


def _build_compare_table2(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    targets = ["RMST", "survival.probability"]
    models = ["Final Conditional Model", "R-CSF Baseline"]
    labels = {"Final Conditional Model": "Final", "R-CSF Baseline": "CSF"}
    for target in targets:
        panel = "Panel A: RMST" if target == "RMST" else "Panel B: Survival probability"
        sub = summary[summary["target"] == target]
        for setting_id in sorted(sub["setting_id"].unique()):
            row: dict[str, float | int | str] = {"Panel": panel, "Setting": int(setting_id)}
            for model in models:
                model_rows = sub[(sub["setting_id"] == setting_id) & (sub["model"] == model)]
                row[labels[model]] = float(model_rows["class_error"].iloc[0]) if not model_rows.empty else np.nan
            rows.append(row)
            panel = ""
    return pd.DataFrame(rows)


def _render_summary_png(summary: pd.DataFrame, path: Path) -> None:
    display = summary.copy()
    for col in ["mean_horizon", "mse_x100", "mse_x100_sd", "rmse", "mae", "class_error", "sign_acc", "bias", "pearson"]:
        display[col] = display[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
    display["fit_time_sec"] = display["fit_time_sec"].map(lambda v: f"{float(v):.1f}")

    fig_h = max(3.2, 1.0 + 0.45 * len(display))
    fig, ax = plt.subplots(figsize=(15, fig_h), dpi=180)
    ax.axis("off")
    ax.set_title("Paper-Style Final Model Simulation Summary", fontsize=16, fontweight="bold", pad=12)
    table = ax.table(
        cellText=display.loc[
            :,
            ["model", "proxy_layout", "setting_id", "target", "reps", "mean_horizon", "mse_x100", "class_error", "sign_acc", "pearson", "fit_time_sec"],
        ].values,
        colLabels=["Model", "Proxy Layout", "Setting", "Target", "Reps", "Horizon", "MSE x100", "Class Error", "Sign Acc", "Pearson", "Fit Time (s)"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _render_paper_table_png(table_df: pd.DataFrame, path: Path, *, title: str) -> None:
    display = table_df.copy()
    for col in display.columns:
        if col not in {"Panel", "Setting", "Metric"}:
            display[col] = display[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
    fig_h = max(3.2, 1.2 + 0.55 * len(display))
    fig, ax = plt.subplots(figsize=(16, fig_h), dpi=180)
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    table = ax.table(
        cellText=display.values,
        colLabels=list(display.columns),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _resolve_protocol_settings(args: argparse.Namespace) -> dict[str, int | str]:
    if args.protocol == "paper-exact":
        return {
            "reps": int(250 if args.reps is None else args.reps),
            "n_train": int(2000 if args.n_train is None else args.n_train),
            "n_test": int(2000 if args.n_test is None else args.n_test),
            "p": int(15 if args.p is None else args.p),
            "proxy_layout": args.proxy_layout,
            "num_trees_r": int(2000 if args.num_trees_r is None else args.num_trees_r),
            "final_num_trees": int(2000 if args.final_num_trees is None else args.final_num_trees),
            "final_min_node_size": int(5 if args.final_min_node_size is None else args.final_min_node_size),
            "h_trees": int(500 if args.h_trees is None else args.h_trees),
            "h_min_node_size": int(15 if args.h_min_node_size is None else args.h_min_node_size),
        }
    return {
        "reps": int(10 if args.reps is None else args.reps),
        "n_train": int(2000 if args.n_train is None else args.n_train),
        "n_test": int(2000 if args.n_test is None else args.n_test),
        "p": int(15 if args.p is None else args.p),
        "proxy_layout": args.proxy_layout,
        "num_trees_r": int(200 if args.num_trees_r is None else args.num_trees_r),
        "final_num_trees": int(200 if args.final_num_trees is None else args.final_num_trees),
        "final_min_node_size": int(20 if args.final_min_node_size is None else args.final_min_node_size),
        "h_trees": int(600 if args.h_trees is None else args.h_trees),
        "h_min_node_size": int(5 if args.h_min_node_size is None else args.h_min_node_size),
    }


RESULT_KEY_COLS = ["model", "proxy_layout", "protocol", "setting_id", "target", "rep_seed"]
RESUME_COMPAT_KEYS = [
    "model",
    "selected_models",
    "protocol",
    "proxy_layout",
    "n_train",
    "n_test",
    "p",
    "seed",
    "proxy_dim_w",
    "proxy_dim_z",
    "final_num_trees",
    "final_min_node_size",
    "h_trees",
    "h_min_node_size",
    "num_trees_r",
]


def _empty_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model",
            "proxy_layout",
            "proxy_layout_label",
            "protocol",
            "setting_id",
            "target",
            "rep_seed",
            "horizon",
            "n_train",
            "n_test",
            "proxy_dim_w",
            "proxy_dim_z",
            "mse",
            "mse_x100",
            "rmse",
            "mae",
            "sign_acc",
            "class_error",
            "mean_pred",
            "mean_true",
            "bias",
            "pearson",
            "fit_time_sec",
        ]
    )


def _load_existing_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_results_frame()
    results = pd.read_csv(path)
    if results.empty:
        return _empty_results_frame()
    results = results.drop_duplicates(subset=RESULT_KEY_COLS, keep="last").reset_index(drop=True)
    return results


def _load_existing_metadata(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_resume_compatible(existing_metadata: dict[str, object] | None, current_metadata: dict[str, object]) -> None:
    if not existing_metadata:
        return

    mismatches: list[str] = []
    for key in RESUME_COMPAT_KEYS:
        existing_value = existing_metadata.get(key)
        current_value = current_metadata.get(key)
        if existing_value != current_value:
            mismatches.append(f"{key}: existing={existing_value!r}, current={current_value!r}")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Resume requested, but the existing output directory was created with different run settings. "
            f"Use a new --output-dir or pass --no-resume. Differences: {mismatch_text}"
        )


def _filter_results_to_requested_tasks(results: pd.DataFrame, requested_keys: set[tuple[str, str, str, int, str, int]]) -> pd.DataFrame:
    if results.empty:
        return _empty_results_frame()

    key_frame = results.loc[:, RESULT_KEY_COLS].copy()
    key_frame["setting_id"] = key_frame["setting_id"].astype(int)
    key_frame["rep_seed"] = key_frame["rep_seed"].astype(int)
    mask = [
        (
            str(row["model"]),
            str(row["proxy_layout"]),
            str(row["protocol"]),
            int(row["setting_id"]),
            str(row["target"]),
            int(row["rep_seed"]),
        )
        in requested_keys
        for _, row in key_frame.iterrows()
    ]
    filtered = results.loc[mask].copy()
    if filtered.empty:
        return _empty_results_frame()
    return filtered.reset_index(drop=True)


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _save_progress_outputs(
    *,
    results: pd.DataFrame,
    output_dir: Path,
    selected_models: list[str],
    metadata_base: dict[str, object],
    elapsed_sec: float,
    total_tasks: int,
    completed_tasks: int,
    status: str,
    last_completed: dict[str, object] | None = None,
    render_png: bool = False,
) -> None:
    results_path = output_dir / "replication_results.csv"
    summary_path = output_dir / "summary.csv"
    png_path = output_dir / "summary.png"
    metadata_path = output_dir / "metadata.json"
    progress_path = output_dir / "progress.json"

    if results.empty:
        results = _empty_results_frame()
        summary = pd.DataFrame()
    else:
        results = results.drop_duplicates(subset=RESULT_KEY_COLS, keep="last").reset_index(drop=True)
        results = _attach_repwise_excess_mse(results)
        summary = _summarize(results)

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    if render_png and not summary.empty:
        _render_summary_png(summary, png_path)

    extra_outputs: list[str] = []
    if not summary.empty:
        if len(selected_models) > 1:
            table1 = _build_compare_table1(summary)
            table2 = _build_compare_table2(summary)
            table1_path = output_dir / "table1_mse_excess.csv"
            table1_png_path = output_dir / "table1_mse_excess.png"
            table2_path = output_dir / "table2_classification.csv"
            table2_png_path = output_dir / "table2_classification.png"
            table1.to_csv(table1_path, index=False)
            table2.to_csv(table2_path, index=False)
            if render_png:
                _render_paper_table_png(table1, table1_png_path, title="Table 1. MSE and Excess MSE")
                _render_paper_table_png(table2, table2_png_path, title="Table 2. Classification Error")
            extra_outputs.extend([str(table1_path), str(table2_path)])
            if render_png:
                extra_outputs.extend([str(table1_png_path), str(table2_png_path)])
        else:
            paper_table = _build_paper_table(summary)
            paper_table_path = output_dir / "paper_table.csv"
            paper_table_png_path = output_dir / "paper_table.png"
            paper_table.to_csv(paper_table_path, index=False)
            if render_png:
                _render_paper_table_png(
                    paper_table,
                    paper_table_png_path,
                    title=f"Paper-Style Final Model Table: {selected_models[0]}",
                )
            extra_outputs.append(str(paper_table_path))
            if render_png:
                extra_outputs.append(str(paper_table_png_path))

    metadata = dict(metadata_base)
    metadata.update(
        {
            "elapsed_sec": float(elapsed_sec),
            "completed_tasks": int(completed_tasks),
            "total_tasks": int(total_tasks),
            "remaining_tasks": int(total_tasks - completed_tasks),
            "status": status,
            "generated_outputs": extra_outputs,
        }
    )
    if last_completed is not None:
        metadata["last_completed"] = last_completed
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    progress = {
        "status": status,
        "completed_tasks": int(completed_tasks),
        "total_tasks": int(total_tasks),
        "remaining_tasks": int(total_tasks - completed_tasks),
        "percent_complete": float(100.0 * completed_tasks / max(total_tasks, 1)),
        "elapsed_sec": float(elapsed_sec),
        "elapsed_pretty": _format_seconds(elapsed_sec),
        "last_completed": last_completed,
    }
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    resolved = _resolve_protocol_settings(args)
    reps = int(resolved["reps"])
    n_train = int(resolved["n_train"])
    n_test = int(resolved["n_test"])
    p = int(resolved["p"])
    proxy_layout = str(resolved["proxy_layout"])
    final_num_trees = int(resolved["final_num_trees"])
    final_min_node_size = int(resolved["final_min_node_size"])
    h_trees = int(resolved["h_trees"])
    h_min_node_size = int(resolved["h_min_node_size"])

    if p < 3:
        raise ValueError("p must be at least 3 because the paper settings use X1, X2, and X3.")
    if args.proxy_dim_w < 1 or args.proxy_dim_z < 1:
        raise ValueError("proxy-dim-w and proxy-dim-z must both be at least 1.")
    if proxy_layout == "split-5-5-5" and p != 15:
        raise ValueError("proxy-layout='split-5-5-5' currently requires --p 15.")

    selected_models = ["final", "r-csf"] if args.model == "compare-final-rcsf" else [args.model]
    output_dir = args.output_dir.resolve() / args.model / args.protocol / proxy_layout
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "replication_results.csv"

    targets = ["RMST", "survival.probability"] if args.target == "both" else [args.target]
    started = time.time()
    metadata_base = {
        "script": str(Path(__file__).resolve()),
        "model": args.model,
        "selected_models": selected_models,
        "protocol": args.protocol,
        "proxy_layout": proxy_layout,
        "settings": args.settings,
        "target": args.target,
        "reps": reps,
        "n_train": n_train,
        "n_test": n_test,
        "p": p,
        "seed": int(args.seed),
        "proxy_dim_w": int(args.proxy_dim_w),
        "proxy_dim_z": int(args.proxy_dim_z),
        "final_num_trees": final_num_trees,
        "final_min_node_size": final_min_node_size,
        "h_trees": h_trees,
        "h_min_node_size": h_min_node_size,
        "num_trees_r": int(resolved["num_trees_r"]),
        "notes": [
            "This script uses the paper-style CSF DGP defined locally in Archive/paper_csf_dgp.py, built on grf.synthetic.grf type1-type4.",
            f"Protocol for this run: {PROTOCOLS[args.protocol]}",
            f"Proxy layout for this run: {PROXY_LAYOUTS[proxy_layout]}.",
            "The Final path is pinned to the current single-file Final Conditional model with surv_scalar_mode='full'.",
            "For paper-exact runs, we align the final-stage forest scale to the paper where possible: 2,000 trees and minimum node size 5, and we set the h-forest scale to 500 trees with minimum node size 15.",
            "The R-CSF path is pinned directly to Archive/run_paper_grf_csf_direct.R, which now sets num.trees=2,000, min.node.size=5, sample.fraction=0.5, and mtry=min(p, ceil(sqrt(p))+20) for paper-exact runs.",
            "When proxy_layout='zero', this is the closest paper-style replication because all observed covariates are passed through X and W/Z are zero placeholders.",
            "When proxy_layout='split-5-5-5', the DGP, horizons, sample sizes, and evaluation protocol remain paper-matched, but the X/W/Z split should be interpreted as our proxy extension rather than an exact paper replication.",
            "Excess MSE is computed using the paper definition: for each replication, divide by the best method's MSE within that setting/target, then average those ratios across replications.",
            "Reported MSE is computed against the known test-set true CATE from the closed-form DGP.",
        ],
    }

    tasks: list[dict[str, object]] = []
    for model_key in selected_models:
        model_spec = MODEL_SPECS[model_key]
        model_name = model_spec["name"]
        model_kind = model_spec["kind"]
        model_cls = model_spec["cls"]
        for target in targets:
            for setting_id in args.settings:
                for rep in range(reps):
                    seed = int(args.seed + 1000 * setting_id + 10000 * (0 if target == "RMST" else 1) + rep)
                    tasks.append(
                        {
                            "model_name": model_name,
                            "model_kind": model_kind,
                            "model_cls": model_cls,
                            "setting_id": int(setting_id),
                            "target": target,
                            "seed": seed,
                        }
                    )

    total_tasks = len(tasks)
    requested_keys = {
        (
            str(task["model_name"]),
            proxy_layout,
            args.protocol,
            int(task["setting_id"]),
            str(task["target"]),
            int(task["seed"]),
        )
        for task in tasks
    }
    metadata_path = output_dir / "metadata.json"
    if args.resume:
        _assert_resume_compatible(_load_existing_metadata(metadata_path), metadata_base)
        existing_results = _filter_results_to_requested_tasks(_load_existing_results(results_path), requested_keys)
    else:
        existing_results = _empty_results_frame()
    completed_keys = {
        (
            str(row["model"]),
            str(row["proxy_layout"]),
            str(row["protocol"]),
            int(row["setting_id"]),
            str(row["target"]),
            int(row["rep_seed"]),
        )
        for _, row in existing_results.iterrows()
    }

    if existing_results.empty:
        print(f"Starting fresh run with {total_tasks} replications.", flush=True)
    else:
        print(
            f"Resuming from {len(completed_keys)} completed replications out of {total_tasks}.",
            flush=True,
        )

    pending_tasks = [
        task
        for task in tasks
        if (
            str(task["model_name"]),
            proxy_layout,
            args.protocol,
            int(task["setting_id"]),
            str(task["target"]),
            int(task["seed"]),
        )
        not in completed_keys
    ]
    print(f"Pending replications: {len(pending_tasks)}", flush=True)

    completed_tasks = len(completed_keys)
    last_completed: dict[str, object] | None = None
    results = existing_results.copy()

    _save_progress_outputs(
        results=results,
        output_dir=output_dir,
        selected_models=selected_models,
        metadata_base=metadata_base,
        elapsed_sec=time.time() - started,
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        status="running",
        last_completed=None,
        render_png=False,
    )

    try:
        for index, task in enumerate(pending_tasks, start=1):
            task_started = time.time()
            row = _evaluate_replication(
                model_name=str(task["model_name"]),
                model_kind=str(task["model_kind"]),
                model_cls=task["model_cls"],
                setting_id=int(task["setting_id"]),
                target=str(task["target"]),
                n_train=n_train,
                n_test=n_test,
                p=p,
                seed=int(task["seed"]),
                proxy_dim_w=args.proxy_dim_w,
                proxy_dim_z=args.proxy_dim_z,
                proxy_layout=proxy_layout,
                protocol=args.protocol,
                num_trees_r=int(resolved["num_trees_r"]),
                final_num_trees=final_num_trees,
                final_min_node_size=final_min_node_size,
                h_trees=h_trees,
                h_min_node_size=h_min_node_size,
            )
            row_frame = pd.DataFrame([row])
            if results.empty:
                results = row_frame.copy()
            else:
                results = pd.concat([results, row_frame], ignore_index=True)
            completed_tasks += 1
            last_completed = {
                "model": row["model"],
                "target": row["target"],
                "setting_id": int(row["setting_id"]),
                "rep_seed": int(row["rep_seed"]),
                "fit_time_sec": float(row["fit_time_sec"]),
                "task_elapsed_sec": float(time.time() - task_started),
            }

            if completed_tasks % max(args.progress_every, 1) == 0 or completed_tasks == total_tasks:
                elapsed = time.time() - started
                avg_time = elapsed / max(completed_tasks - len(completed_keys), 1)
                remaining = total_tasks - completed_tasks
                eta = avg_time * remaining
                print(
                    (
                        f"[progress] {completed_tasks}/{total_tasks} "
                        f"({100.0 * completed_tasks / max(total_tasks, 1):.1f}%) "
                        f"completed | last={row['model']} {row['target']} "
                        f"setting={row['setting_id']} seed={row['rep_seed']} "
                        f"| elapsed={_format_seconds(elapsed)} ETA={_format_seconds(eta)}"
                    ),
                    flush=True,
                )

            if completed_tasks % max(args.save_every, 1) == 0 or completed_tasks == total_tasks:
                _save_progress_outputs(
                    results=results,
                    output_dir=output_dir,
                    selected_models=selected_models,
                    metadata_base=metadata_base,
                    elapsed_sec=time.time() - started,
                    total_tasks=total_tasks,
                    completed_tasks=completed_tasks,
                    status="running",
                    last_completed=last_completed,
                    render_png=False,
                )
                print(
                    f"[checkpoint] Saved partial results after {completed_tasks} / {total_tasks} replications.",
                    flush=True,
                )
    except KeyboardInterrupt:
        _save_progress_outputs(
            results=results,
            output_dir=output_dir,
            selected_models=selected_models,
            metadata_base=metadata_base,
            elapsed_sec=time.time() - started,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            status="interrupted",
            last_completed=last_completed,
            render_png=False,
        )
        print("[interrupted] Partial results saved before exit.", flush=True)
        raise
    except Exception:
        _save_progress_outputs(
            results=results,
            output_dir=output_dir,
            selected_models=selected_models,
            metadata_base=metadata_base,
            elapsed_sec=time.time() - started,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            status="failed",
            last_completed=last_completed,
            render_png=False,
        )
        print("[failed] Partial results saved before propagating the error.", flush=True)
        raise

    _save_progress_outputs(
        results=results,
        output_dir=output_dir,
        selected_models=selected_models,
        metadata_base=metadata_base,
        elapsed_sec=time.time() - started,
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        status="completed",
        last_completed=last_completed,
        render_png=True,
    )

    final_results = _attach_repwise_excess_mse(results.drop_duplicates(subset=RESULT_KEY_COLS, keep="last"))
    final_summary = _summarize(final_results)
    metadata_path = output_dir / "metadata.json"
    summary_path = output_dir / "summary.csv"
    png_path = output_dir / "summary.png"

    print(f"Saved replication results: {results_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Saved summary PNG: {png_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)
    print(final_summary.to_string(index=False), flush=True)
    if len(selected_models) > 1:
        print("\nTable 1 / Table 2 outputs saved.", flush=True)
    else:
        paper_table = _build_paper_table(final_summary)
        print("\nPaper-style table", flush=True)
        print(paper_table.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
