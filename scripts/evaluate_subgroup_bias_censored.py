#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
MODEL_DIR = Path(__file__).resolve().parent

if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from grf.benchmarks.econml_8variant import prepare_case  # noqa: E402

try:  # pragma: no cover
    from .final_censored_model_conditional import FinalModelConditionalCensoredSurvivalForest  # noqa: E402
    from .revised_censored_baseline import RevisedBaselineCensoredSurvivalForest  # noqa: E402
    from .run_5model_benchmark import (  # noqa: E402
        CASE_SPECS,
        SETTINGS,
        SingleFileFinalOracleCensoredSurvivalForest,
    )
except ImportError:  # pragma: no cover
    from single_file_censored_models.final_censored_model_conditional import FinalModelConditionalCensoredSurvivalForest  # type: ignore  # noqa: E402
    from single_file_censored_models.revised_censored_baseline import RevisedBaselineCensoredSurvivalForest  # type: ignore  # noqa: E402
    from single_file_censored_models.run_5model_benchmark import (  # type: ignore  # noqa: E402
        CASE_SPECS,
        SETTINGS,
        SingleFileFinalOracleCensoredSurvivalForest,
    )


SUPPORTED_MODELS = {
    "Final Conditional Oracle",
    "Final Conditional",
    "Revised Conditional",
}


def _log(message: str) -> None:
    print(message, flush=True)


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate subgroup-level bias for censored synthetic settings using an "
            "independent large test dataset."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "subgroup_bias" / "censored",
        help="Directory to write subgroup bias outputs.",
    )
    parser.add_argument(
        "--setting-ids",
        nargs="*",
        default=None,
        help="Optional subset of setting IDs (e.g., S01 S02).",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of case IDs to evaluate (default: all).",
    )
    parser.add_argument(
        "--target",
        choices=["RMST", "survival.probability", "both"],
        default="both",
        help="Target to evaluate.",
    )
    parser.add_argument(
        "--rmst-horizon",
        type=float,
        default=3.0,
        help="Fixed RMST horizon used when target=RMST.",
    )
    parser.add_argument(
        "--horizon-quantile",
        type=float,
        default=0.90,
        help="Survival-probability horizon quantile used by prepare_case.",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=42,
        help="Random seed for training data generation.",
    )
    parser.add_argument(
        "--test-seed-base",
        type=int,
        default=20270000,
        help="Base seed used to generate independent test datasets.",
    )
    parser.add_argument(
        "--test-n",
        type=int,
        default=20000,
        help="Number of rows in the independent test dataset per setting-case-target.",
    )
    parser.add_argument(
        "--min-subgroup-size",
        type=int,
        default=200,
        help="Minimum subgroup size required to report subgroup metrics.",
    )
    parser.add_argument(
        "--x-quantiles",
        type=int,
        default=3,
        help="Number of quantile bins for each selected X covariate subgroup family.",
    )
    parser.add_argument(
        "--tau-quantiles",
        type=int,
        default=5,
        help="Number of quantile bins for true-tau subgroup family.",
    )
    parser.add_argument(
        "--x-subgroup-cols",
        nargs="*",
        type=int,
        default=[0, 1],
        help="Indices of X columns used for covariate subgrouping (e.g., 0 1).",
    )
    parser.add_argument(
        "--skip-cross",
        action="store_true",
        help="Disable cross groups between the first two X subgroup columns.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "Final Conditional Oracle",
            "Final Conditional",
            "Revised Conditional",
        ],
        help=(
            "Models to evaluate. Supported: 'Final Conditional Oracle', "
            "'Final Conditional', 'Revised Conditional'."
        ),
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=42,
        help="Random state passed to model constructors.",
    )
    parser.add_argument(
        "--num-trees",
        type=int,
        default=200,
        help="Number of trees for the final forest in each model.",
    )
    return parser.parse_args()


def _resolve_targets(target_arg: str) -> list[str]:
    if target_arg == "both":
        return ["RMST", "survival.probability"]
    return [target_arg]


def _prepare_case_for_target(case_spec: dict[str, object], *, target: str, rmst_horizon: float, horizon_quantile: float):
    if target == "RMST":
        return prepare_case(case_spec, target=target, horizon=float(rmst_horizon))
    return prepare_case(case_spec, target=target, horizon_quantile=float(horizon_quantile))


def _setting_sort_key(setting_id: str) -> int:
    text = str(setting_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return int(digits)
    return 10**9


def _selected_settings(setting_ids: list[str] | None) -> list[dict[str, int | str]]:
    if not setting_ids:
        selected = list(SETTINGS)
    else:
        wanted = {str(s).upper() for s in setting_ids}
        selected = [s for s in SETTINGS if str(s["setting_id"]).upper() in wanted]
    if not selected:
        raise ValueError("No settings selected.")
    selected = sorted(selected, key=lambda s: _setting_sort_key(str(s["setting_id"])))
    return selected


def _selected_case_specs(case_ids: list[int] | None) -> list[dict[str, object]]:
    if not case_ids:
        selected = list(CASE_SPECS)
    else:
        wanted = {int(c) for c in case_ids}
        selected = [spec for spec in CASE_SPECS if int(spec["case_id"]) in wanted]
    if not selected:
        raise ValueError("No case specs selected.")
    return selected


def _build_case_spec(
    case_spec: dict[str, object],
    setting: dict[str, int | str],
    *,
    seed: int,
) -> dict[str, object]:
    case_copy = dict(case_spec)
    cfg = dict(case_spec["cfg"])
    cfg.update(
        {
            "n": int(setting["n"]),
            "p_x": int(setting["p_x"]),
            "p_w": int(setting["p_w"]),
            "p_z": int(setting["p_z"]),
            "seed": int(seed),
            # Explicitly enforce censored setup target as requested.
            "target_censor_rate": 0.35,
            # Keep censoring calibration in play for target-censoring generation.
            "lam_c": None,
        }
    )
    case_copy["cfg"] = cfg
    return case_copy


def _safe_qcut(series: pd.Series, q: int, prefix: str) -> pd.Series:
    if q <= 1:
        return pd.Series([f"{prefix}_Q1"] * len(series), index=series.index, dtype="object")

    cats = pd.qcut(series, q=q, labels=False, duplicates="drop")
    if cats.isna().all():
        return pd.Series([f"{prefix}_Q1"] * len(series), index=series.index, dtype="object")

    numeric = cats.fillna(-1).astype(int)
    if (numeric < 0).any():
        valid = numeric[numeric >= 0]
        if valid.empty:
            return pd.Series([f"{prefix}_Q1"] * len(series), index=series.index, dtype="object")
        fill_val = int(valid.mode().iloc[0])
        numeric = numeric.where(numeric >= 0, fill_val)
    return numeric.map(lambda k: f"{prefix}_Q{k + 1}")


def _make_subgroup_frame(
    x_test: np.ndarray,
    true_cate: np.ndarray,
    x_subgroup_cols: list[int],
    x_quantiles: int,
    tau_quantiles: int,
    include_cross: bool,
) -> pd.DataFrame:
    n = x_test.shape[0]
    frame = pd.DataFrame(index=np.arange(n))

    active_cols = [col for col in x_subgroup_cols if 0 <= col < x_test.shape[1]]

    x_labels: dict[int, pd.Series] = {}
    for col_idx in active_cols:
        labels = _safe_qcut(pd.Series(x_test[:, col_idx]), q=x_quantiles, prefix=f"X{col_idx}")
        frame[f"group_x{col_idx}"] = labels
        x_labels[col_idx] = labels

    frame["group_tau"] = _safe_qcut(pd.Series(true_cate), q=tau_quantiles, prefix="tau")

    if include_cross and len(active_cols) >= 2:
        c0, c1 = active_cols[0], active_cols[1]
        frame[f"group_x{c0}x{c1}"] = x_labels[c0].astype(str) + "&" + x_labels[c1].astype(str)

    return frame


def _fit_model(
    name: str,
    train_case,
    *,
    target: str,
    model_seed: int,
    num_trees: int,
):
    x = np.asarray(train_case.X, dtype=float)
    w = np.asarray(train_case.W, dtype=float)
    z = np.asarray(train_case.Z, dtype=float)
    a = np.asarray(train_case.A, dtype=float)
    time_obs = np.asarray(train_case.Y, dtype=float)
    event = np.asarray(train_case.delta, dtype=float)
    u = _ensure_2d(np.asarray(train_case.U, dtype=float))
    horizon = float(train_case.horizon)

    if name == "Final Conditional":
        model = FinalModelConditionalCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=model_seed,
            surv_scalar_mode="full",
            n_estimators=num_trees,
            censoring_estimator="cox",
        )
        t0 = time.time()
        model.fit_components(x, a, time_obs, event, z, w)
        return model, time.time() - t0

    if name == "Final Conditional Oracle":
        model = SingleFileFinalOracleCensoredSurvivalForest(
            cfg=train_case.cfg,
            dgp=train_case.dgp,
            p_x=x.shape[1],
            target=target,
            horizon=horizon,
            random_state=model_seed,
            q_kind="logit",
            h_kind="extra",
            h_n_estimators=600,
            h_min_samples_leaf=5,
            q_clip=0.03,
            y_tilde_clip_quantile=0.98,
            y_res_clip_percentiles=(2.0, 98.0),
            n_estimators=num_trees,
            min_samples_leaf=20,
            cv=5,
            censoring_estimator="cox",
            event_survival_estimator="cox",
            m_pred_mode="bridge",
            nuisance_feature_mode="broad_dup",
        )
        x_oracle = np.column_stack([x, u])
        t0 = time.time()
        model.fit_oracle(x_oracle, a, time_obs, event, u)
        return model, time.time() - t0

    if name == "Revised Conditional":
        model = RevisedBaselineCensoredSurvivalForest(
            target=target,
            horizon=horizon,
            random_state=model_seed,
            n_estimators=num_trees,
            censoring_estimator="cox",
        )
        t0 = time.time()
        model.fit_components(x, a, time_obs, event, z, w)
        return model, time.time() - t0

    raise ValueError(f"Unsupported model name: {name}")


def _predict_effect(name: str, model, test_case) -> np.ndarray:
    x = np.asarray(test_case.X, dtype=float)
    w = np.asarray(test_case.W, dtype=float)
    z = np.asarray(test_case.Z, dtype=float)

    if name == "Final Conditional Oracle":
        u = _ensure_2d(np.asarray(test_case.U, dtype=float))
        return np.asarray(model.effect_oracle(x, u), dtype=float).ravel()

    return np.asarray(model.effect_from_components(x, w, z), dtype=float).ravel()


def _evaluate_subgroups(
    subgroup_frame: pd.DataFrame,
    preds: np.ndarray,
    true_cate: np.ndarray,
    min_subgroup_size: int,
    model_name: str,
    target: str,
    setting_id: str,
    case_id: int,
    case_slug: str,
) -> list[dict[str, object]]:
    preds = np.asarray(preds, dtype=float).ravel()
    true_cate = np.asarray(true_cate, dtype=float).ravel()

    rows: list[dict[str, object]] = []
    subgroup_cols = [col for col in subgroup_frame.columns if col.startswith("group_")]

    for subgroup_col in subgroup_cols:
        family = subgroup_col.replace("group_", "")
        labels = subgroup_frame[subgroup_col].astype(str)
        for subgroup_name, idx in labels.groupby(labels).groups.items():
            mask = np.zeros(len(labels), dtype=bool)
            mask[np.asarray(list(idx), dtype=int)] = True
            n_g = int(mask.sum())
            if n_g < min_subgroup_size:
                continue

            true_ate = float(np.mean(true_cate[mask]))
            pred_ate = float(np.mean(preds[mask]))
            bias = pred_ate - true_ate
            rows.append(
                {
                    "target": target,
                    "setting_id": setting_id,
                    "case_id": int(case_id),
                    "case_slug": case_slug,
                    "model": model_name,
                    "subgroup_family": family,
                    "subgroup": subgroup_name,
                    "n_subgroup": n_g,
                    "true_subgroup_ate": true_ate,
                    "pred_subgroup_ate": pred_ate,
                    "subgroup_bias": float(bias),
                    "subgroup_abs_bias": float(abs(bias)),
                }
            )
    return rows


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = np.asarray(weights, dtype=float)
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return float("nan")
    s = float(np.sum(w))
    if s <= 0:
        return float(np.mean(x))
    return float(np.sum(w * x) / s)


def _build_summary_tables(detailed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_summary_rows: list[dict[str, object]] = []
    family_summary_rows: list[dict[str, object]] = []

    for (target, model_name), mdf in detailed.groupby(["target", "model"]):
        model_summary_rows.append(
            {
                "target": target,
                "model": model_name,
                "num_rows": int(len(mdf)),
                "weighted_signed_bias": _weighted_mean(mdf["subgroup_bias"], mdf["n_subgroup"]),
                "weighted_abs_bias": _weighted_mean(mdf["subgroup_abs_bias"], mdf["n_subgroup"]),
                "worst_abs_bias": float(mdf["subgroup_abs_bias"].max()),
                "median_abs_bias": float(mdf["subgroup_abs_bias"].median()),
            }
        )

        for family, fdf in mdf.groupby("subgroup_family"):
            family_summary_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "subgroup_family": family,
                    "num_rows": int(len(fdf)),
                    "weighted_signed_bias": _weighted_mean(fdf["subgroup_bias"], fdf["n_subgroup"]),
                    "weighted_abs_bias": _weighted_mean(fdf["subgroup_abs_bias"], fdf["n_subgroup"]),
                    "worst_abs_bias": float(fdf["subgroup_abs_bias"].max()),
                }
            )

    model_summary = (
        pd.DataFrame(model_summary_rows)
        .sort_values(["target", "weighted_abs_bias"])
        .reset_index(drop=True)
    )
    family_summary = (
        pd.DataFrame(family_summary_rows)
        .sort_values(["target", "subgroup_family", "weighted_abs_bias"])
        .reset_index(drop=True)
    )
    return model_summary, family_summary


def main() -> int:
    run_start = time.time()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested_models = [m.strip() for m in args.models]
    bad = [m for m in requested_models if m not in SUPPORTED_MODELS]
    if bad:
        raise ValueError(f"Unsupported models requested: {bad}. Supported: {sorted(SUPPORTED_MODELS)}")

    settings = _selected_settings(args.setting_ids)
    case_specs = _selected_case_specs(args.case_ids)
    targets = _resolve_targets(args.target)

    total_settings = int(len(settings))
    total_cases = int(len(case_specs))
    total_models = int(len(requested_models))
    total_targets = int(len(targets))
    total_fits = total_settings * total_cases * total_targets * total_models
    fit_counter = 0

    _log(
        (
            "[start] subgroup-bias evaluation (censored) "
            f"| settings={total_settings} cases={total_cases} targets={total_targets} "
            f"models={total_models} total_fits={total_fits} | test_n={args.test_n}"
        )
    )
    _log("[start] censoring target rate is explicitly fixed at 0.35 for all generated runs")

    detailed_rows: list[dict[str, object]] = []
    train_time_rows: list[dict[str, object]] = []

    for s_idx, setting in enumerate(settings):
        setting_id = str(setting["setting_id"])
        _log(f"[setting {s_idx + 1}/{total_settings}] {setting_id} started")

        for case_spec in case_specs:
            case_id = int(case_spec["case_id"])
            case_slug = str(case_spec["slug"])
            _log(
                (
                    f"  [case {case_id:02d}/{total_cases:02d}] {case_slug} "
                    f"| train_n={int(setting['n'])} p_x={int(setting['p_x'])} "
                    f"p_w={int(setting['p_w'])} p_z={int(setting['p_z'])}"
                )
            )

            for t_idx, target in enumerate(targets):
                train_seed = int(args.train_seed)
                test_seed = int(args.test_seed_base + 100000 * s_idx + 1000 * case_id + 10 * t_idx)

                train_spec = _build_case_spec(case_spec, setting, seed=train_seed)
                test_spec = _build_case_spec(case_spec, setting, seed=test_seed)
                test_spec_cfg = dict(test_spec["cfg"])
                test_spec_cfg["n"] = int(args.test_n)
                test_spec["cfg"] = test_spec_cfg

                train_case = _prepare_case_for_target(
                    train_spec,
                    target=target,
                    rmst_horizon=float(args.rmst_horizon),
                    horizon_quantile=float(args.horizon_quantile),
                )
                test_case = _prepare_case_for_target(
                    test_spec,
                    target=target,
                    rmst_horizon=float(args.rmst_horizon),
                    horizon_quantile=float(args.horizon_quantile),
                )

                subgroup_frame = _make_subgroup_frame(
                    x_test=np.asarray(test_case.X, dtype=float),
                    true_cate=np.asarray(test_case.true_cate, dtype=float),
                    x_subgroup_cols=list(args.x_subgroup_cols),
                    x_quantiles=int(args.x_quantiles),
                    tau_quantiles=int(args.tau_quantiles),
                    include_cross=not args.skip_cross,
                )

                for model_name in requested_models:
                    _log(f"    [fit {fit_counter + 1}/{total_fits}] target={target} training {model_name}")

                    model, elapsed = _fit_model(
                        model_name,
                        train_case,
                        target=target,
                        model_seed=int(args.model_seed),
                        num_trees=int(args.num_trees),
                    )

                    preds = _predict_effect(model_name, model, test_case)

                    detailed_rows.extend(
                        _evaluate_subgroups(
                            subgroup_frame=subgroup_frame,
                            preds=preds,
                            true_cate=np.asarray(test_case.true_cate, dtype=float),
                            min_subgroup_size=int(args.min_subgroup_size),
                            model_name=model_name,
                            target=target,
                            setting_id=setting_id,
                            case_id=case_id,
                            case_slug=case_slug,
                        )
                    )

                    train_time_rows.append(
                        {
                            "target": target,
                            "setting_id": setting_id,
                            "case_id": case_id,
                            "case_slug": case_slug,
                            "model": model_name,
                            "train_seconds": float(elapsed),
                            "train_n": int(train_spec["cfg"]["n"]),
                            "test_n": int(test_spec["cfg"]["n"]),
                        }
                    )
                    fit_counter += 1
                    _log(f"    [fit {fit_counter}/{total_fits}] done {model_name} | train={elapsed:.2f}s")

        _log(f"[setting {s_idx + 1}/{total_settings}] {setting_id} completed")

    if not detailed_rows:
        raise RuntimeError("No subgroup rows were produced. Consider lowering --min-subgroup-size.")

    detailed = pd.DataFrame(detailed_rows)
    model_summary, family_summary = _build_summary_tables(detailed)
    train_times = pd.DataFrame(train_time_rows)

    detailed_path = args.output_dir / "subgroup_bias_detailed.csv"
    model_summary_path = args.output_dir / "subgroup_bias_model_summary.csv"
    family_summary_path = args.output_dir / "subgroup_bias_family_summary.csv"
    train_times_path = args.output_dir / "subgroup_bias_train_times.csv"

    detailed.to_csv(detailed_path, index=False)
    model_summary.to_csv(model_summary_path, index=False)
    family_summary.to_csv(family_summary_path, index=False)
    train_times.to_csv(train_times_path, index=False)

    run_config = {
        "output_dir": str(args.output_dir.resolve()),
        "setting_ids": args.setting_ids,
        "case_ids": args.case_ids,
        "target": args.target,
        "rmst_horizon": args.rmst_horizon,
        "horizon_quantile": args.horizon_quantile,
        "train_seed": args.train_seed,
        "test_seed_base": args.test_seed_base,
        "test_n": args.test_n,
        "min_subgroup_size": args.min_subgroup_size,
        "x_quantiles": args.x_quantiles,
        "tau_quantiles": args.tau_quantiles,
        "x_subgroup_cols": args.x_subgroup_cols,
        "skip_cross": args.skip_cross,
        "models": requested_models,
        "model_seed": args.model_seed,
        "num_trees": args.num_trees,
        "target_censor_rate": 0.35,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"Saved: {detailed_path}")
    print(f"Saved: {model_summary_path}")
    print(f"Saved: {family_summary_path}")
    print(f"Saved: {train_times_path}")
    print(f"Rows: {len(detailed)}")
    print(f"Settings: {len(settings)} | Cases: {len(case_specs)} | Targets: {len(targets)}")
    _log(f"[done] total elapsed: {time.time() - run_start:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
