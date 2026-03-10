from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..methods import causal_survival_forest


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _format_reference_precision(values: np.ndarray) -> list[str]:
    return [format(float(value), ".16g") for value in values]


def _format_metric_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 1e-3):
            return f"{value:.6e}"
        return f"{value:.6f}"
    return str(value)


def render_comparison_report(metrics: dict[str, object], output_path: Path) -> None:
    rows = [
        ("Numeric Match %", metrics["numeric_match_pct"]),
        ("Exact Float Match %", metrics["exact_float_match_pct"]),
        ("Exact Text Match %", metrics["exact_text_match_pct"]),
        ("Pearson Correlation", metrics["pearson_correlation"]),
        ("RMSE", metrics["rmse"]),
        ("Max Abs Diff", metrics["max_abs_diff"]),
        ("Strict Float Identical", metrics["strict_float_identical"]),
        ("Strict Text Identical", metrics["strict_text_identical"]),
        ("Passed Threshold", metrics["passed"]),
    ]
    display = pd.DataFrame(
        {
            "Metric": [row[0] for row in rows],
            "Value": [_format_metric_value(row[1]) for row in rows],
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.6), dpi=200)
    fig.patch.set_facecolor("#efefef")
    ax.axis("off")

    fig.suptitle("Baseline CSF vs R GRF Reference", fontsize=20, fontweight="semibold", y=0.97)
    fig.text(0.5, 0.90, "Bundled reference prediction comparison", ha="center", va="center", fontsize=12, color="#56606e")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        colLoc="center",
        cellLoc="center",
        colWidths=[0.62, 0.30],
        bbox=[0.05, 0.17, 0.90, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c7cbd3")
        if row == 0:
            cell.set_facecolor("#1c2a3f")
            cell.set_text_props(color="white", weight="bold")
            cell.set_linewidth(1.6)
            cell.set_height(0.11)
        else:
            cell.set_facecolor("#f7f7f8" if row % 2 == 1 else "#ececef")
            cell.set_height(0.10)
            if col == 0:
                cell.set_text_props(ha="left")

    footer = (
        f"Dataset: {metrics['dataset']}\n"
        f"Reference CSV: {metrics['reference_predictions']}\n"
        f"Python CSV: {metrics['python_csv_path']}"
    )
    fig.text(0.05, 0.08, footer, ha="left", va="bottom", fontsize=9, color="#4a5563")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_reference_comparison(project_root: Path) -> dict[str, object]:
    data_path = project_root / "data" / "reference_input.csv"
    reference_path = project_root / "data" / "reference_grf_master_predictions.csv"
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(data_path)
    reference_frame = pd.read_csv(reference_path)
    reference_predictions = reference_frame["prediction"].to_numpy(dtype=float)
    reference_text_lines = reference_path.read_text(encoding="utf-8").splitlines()
    reference_value_text = reference_text_lines[1:]
    feature_cols = sorted([column for column in frame.columns if column.startswith("X")], key=lambda col: int(col[1:]))

    horizon = float(frame["time"].max())
    num_trees = 200
    num_threads = 1
    seed = 42

    candidate = causal_survival_forest(
        frame[feature_cols].to_numpy(),
        frame["time"].to_numpy(),
        frame["A"].to_numpy(),
        frame["event"].to_numpy(),
        target="RMST",
        horizon=horizon,
        num_trees=num_trees,
        num_threads=num_threads,
        seed=seed,
        feature_names=feature_cols,
    )

    try:
        candidate_predictions = candidate.predict()["predictions"].to_numpy(dtype=float)
    finally:
        candidate.cleanup()

    pd.DataFrame({"prediction": reference_predictions}).to_csv(outputs_dir / "reference_predictions.csv", index=False)
    pd.DataFrame({"prediction": candidate_predictions}).to_csv(outputs_dir / "python_predictions.csv", index=False)

    candidate_precision_lines = ["prediction", *_format_reference_precision(candidate_predictions)]
    candidate_precision_path = outputs_dir / "python_predictions_reference_precision.csv"
    candidate_precision_path.write_text("\n".join(candidate_precision_lines) + "\n", encoding="utf-8")

    is_close = np.isclose(reference_predictions, candidate_predictions, rtol=1e-6, atol=1e-8)
    exact_float_equal = reference_predictions == candidate_predictions
    exact_text_equal = np.array(reference_value_text) == np.array(candidate_precision_lines[1:])
    agreement_pct = 100.0 * float(np.mean(is_close))
    exact_float_match_pct = 100.0 * float(np.mean(exact_float_equal))
    exact_text_match_pct = 100.0 * float(np.mean(exact_text_equal))
    correlation = _pearson_correlation(reference_predictions, candidate_predictions)
    rmse = float(np.sqrt(np.mean((reference_predictions - candidate_predictions) ** 2)))
    max_abs_diff = float(np.max(np.abs(reference_predictions - candidate_predictions)))

    comparison_rows = pd.DataFrame(
        {
            "index": np.arange(reference_predictions.shape[0]),
            "reference_prediction": reference_predictions,
            "python_prediction": candidate_predictions,
            "abs_diff": np.abs(reference_predictions - candidate_predictions),
            "exact_float_equal": exact_float_equal,
            "within_tolerance": is_close,
            "reference_text": reference_value_text,
            "python_text_reference_precision": candidate_precision_lines[1:],
            "exact_text_equal": exact_text_equal,
        }
    )
    comparison_rows.to_csv(outputs_dir / "reference_comparison_rows.csv", index=False)

    metrics = {
        "dataset": str(data_path),
        "reference_predictions": str(reference_path),
        "horizon": horizon,
        "num_trees": num_trees,
        "num_threads": num_threads,
        "seed": seed,
        "numeric_match_pct": agreement_pct,
        "exact_float_match_pct": exact_float_match_pct,
        "exact_text_match_pct": exact_text_match_pct,
        "pearson_correlation": correlation,
        "rmse": rmse,
        "max_abs_diff": max_abs_diff,
        "reference_csv_path": str(reference_path),
        "python_csv_path": str(outputs_dir / "python_predictions.csv"),
        "python_reference_precision_csv_path": str(candidate_precision_path),
        "row_report_path": str(outputs_dir / "reference_comparison_rows.csv"),
        "strict_float_identical": bool(np.all(exact_float_equal)),
        "strict_text_identical": bool(np.all(exact_text_equal)),
        "threshold": 90.0,
        "passed": agreement_pct >= 90.0,
    }
    (outputs_dir / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    render_comparison_report(metrics, outputs_dir / "reference_comparison_report.png")
    render_comparison_report(metrics, project_root / "reference_comparison_report.png")
    return metrics
