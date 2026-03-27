#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path("/Users/kyoungeuihong/Desktop/csf_grf_new")
DESKTOP = Path("/Users/kyoungeuihong/Desktop")

SETTINGS = {
    "S01": {"n": 1000, "p_x": 5, "p_w": 1, "p_z": 1},
    "S02": {"n": 2000, "p_x": 5, "p_w": 1, "p_z": 1},
    "S03": {"n": 4000, "p_x": 5, "p_w": 1, "p_z": 1},
    "S04": {"n": 8000, "p_x": 5, "p_w": 1, "p_z": 1},
    "S05": {"n": 2000, "p_x": 5, "p_w": 3, "p_z": 3},
    "S06": {"n": 2000, "p_x": 5, "p_w": 5, "p_z": 5},
    "S07": {"n": 2000, "p_x": 5, "p_w": 10, "p_z": 10},
    "S08": {"n": 2000, "p_x": 10, "p_w": 1, "p_z": 1},
    "S09": {"n": 2000, "p_x": 20, "p_w": 1, "p_z": 1},
    "S10": {"n": 2000, "p_x": 10, "p_w": 5, "p_z": 5},
    "S11": {"n": 2000, "p_x": 20, "p_w": 5, "p_z": 5},
    "S12": {"n": 1000, "p_x": 10, "p_w": 5, "p_z": 5},
    "S13": {"n": 2000, "p_x": 10, "p_w": 10, "p_z": 10},
    "S14": {"n": 4000, "p_x": 20, "p_w": 10, "p_z": 10},
}

NC_CANDIDATE_RESULTS = ROOT / "non_censored" / "outputs" / "benchmark_structured_14settings_dml_candidates" / "all_settings_results.csv"
SURV_CANDIDATE_RESULTS = ROOT / "outputs" / "benchmark_structured_14settings_dml_candidates" / "all_settings_results.csv"

NC_REF_DIRS = [
    ROOT / "non_censored" / "outputs" / "benchmark_structured_14settings_extended_full",
    ROOT / "non_censored" / "outputs" / "benchmark_structured_priority_S05_S07",
    ROOT / "non_censored" / "outputs" / "benchmark_structured_14settings_extended",
]
SURV_REF_DIRS = [
    ROOT / "outputs" / "benchmark_structured_14settings_extended_full",
    ROOT / "outputs" / "benchmark_structured_priority_S05_S07",
    ROOT / "outputs" / "benchmark_structured_14settings_extended",
]

OUT_PDF = DESKTOP / "structured_setting_trends_rmse.pdf"


def _setting_from_tuple(n: int, p_x: int, p_w: int, p_z: int) -> str:
    for sid, spec in SETTINGS.items():
        if spec["n"] == n and spec["p_x"] == p_x and spec["p_w"] == p_w and spec["p_z"] == p_z:
            return sid
    raise KeyError((n, p_x, p_w, p_z))


def _load_candidate_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["setting_id"] = df.apply(
        lambda r: _setting_from_tuple(int(r["n"]), int(r["p_x"]), int(r["p_w"]), int(r["p_z"])),
        axis=1,
    )
    return df


def _load_ref_results(roots: list[Path], ref_names: list[str]) -> pd.DataFrame:
    frames = []
    for sid in SETTINGS:
        for root in roots:
            matches = sorted(root.glob(f"{sid}_*/results.csv"))
            if matches:
                frames.append(pd.read_csv(matches[0]))
                break
    if not frames:
        raise RuntimeError("No reference setting results found.")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["name"].isin(ref_names)].copy()
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["setting_id", "name"], as_index=False)
        .agg(avg_rmse=("rmse", "mean"))
        .reset_index(drop=True)
    )


def _render_axis_chart(
    *,
    title: str,
    subtitle: str,
    output_path: Path,
    settings: list[str],
    x_values: list[float] | None,
    x_labels: list[str],
    xlabel: str,
    model_order: list[str],
    label_map: dict[str, str],
    color_map: dict[str, str],
    style_map: dict[str, str],
    marker_map: dict[str, str],
    width_map: dict[str, float],
    alpha_map: dict[str, float],
    agg: pd.DataFrame,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=220)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    x_plot = list(range(len(settings))) if x_values is None else x_values

    for name in model_order:
        sub = agg[(agg["name"] == name) & (agg["setting_id"].isin(settings))].copy()
        if sub.empty:
            continue
        sub["setting_order"] = sub["setting_id"].map({sid: i for i, sid in enumerate(settings)})
        sub = sub.sort_values("setting_order")
        xs = [x_plot[i] for i in sub["setting_order"].tolist()]
        ys = sub["avg_rmse"].astype(float).tolist()
        ax.plot(
            xs,
            ys,
            label=label_map[name],
            color=color_map[name],
            linestyle=style_map[name],
            linewidth=width_map[name],
            alpha=alpha_map[name],
            marker=marker_map[name],
            markersize=7,
        )

    ax.set_title(title, fontsize=18.5, fontweight="bold", pad=18)
    ax.text(
        0.0,
        1.01,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color="#667085",
    )
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylabel("Average RMSE", fontsize=13, fontweight="bold")
    ax.set_xticks(x_plot)
    ax.set_xticklabels(x_labels)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.9, alpha=0.9)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=10.2,
        ncol=2,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
    )
    for text in legend.get_texts():
        text.set_color("#111827")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig


def main() -> int:
    nc_cand = _aggregate(_load_candidate_results(NC_CANDIDATE_RESULTS))
    surv_cand = _aggregate(_load_candidate_results(SURV_CANDIDATE_RESULTS))

    nc_ref = _aggregate(
        _load_ref_results(
            NC_REF_DIRS,
            ["EconML Baseline", "Old C3  Legacy NC-CSF"],
        )
    )
    surv_ref = _aggregate(
        _load_ref_results(
            SURV_REF_DIRS,
            ["R-CSF Baseline", "Old C3  Legacy NC-CSF"],
        )
    )

    nc_agg = pd.concat([nc_cand, nc_ref], ignore_index=True)
    surv_agg = pd.concat([surv_cand, surv_ref], ignore_index=True)

    nc_models = [
        "EconML Baseline",
        "Old C3  Legacy NC-CSF",
        "Summary-DML (PCI)",
        "Augmented-DML (PCI)",
        "Summary-DML (no PCI)",
        "Augmented-DML (no PCI)",
    ]
    surv_models = [
        "R-CSF Baseline",
        "Old C3  Legacy NC-CSF",
        "SummarySurv-DML (PCI)",
        "AugmentedSurv-DML (PCI)",
        "SummarySurv-DML (no PCI)",
        "AugmentedSurv-DML (no PCI)",
    ]

    common_colors = {
        "B2": "#9ca3af",
        "Old C3": "#111827",
        "Summary PCI": "#2563eb",
        "Augmented PCI": "#059669",
        "Summary noPCI": "#93c5fd",
        "Augmented noPCI": "#86efac",
    }

    nc_labels = {
        "EconML Baseline": "EconML Baseline",
        "Old C3  Legacy NC-CSF": "Old C3",
        "Summary-DML (PCI)": "Summary-DML (PCI)",
        "Augmented-DML (PCI)": "Augmented-DML (PCI)",
        "Summary-DML (no PCI)": "Summary-DML (no PCI)",
        "Augmented-DML (no PCI)": "Augmented-DML (no PCI)",
    }
    surv_labels = {
        "R-CSF Baseline": "R-CSF Baseline",
        "Old C3  Legacy NC-CSF": "Old C3",
        "SummarySurv-DML (PCI)": "SummarySurv-DML (PCI)",
        "AugmentedSurv-DML (PCI)": "AugmentedSurv-DML (PCI)",
        "SummarySurv-DML (no PCI)": "SummarySurv-DML (no PCI)",
        "AugmentedSurv-DML (no PCI)": "AugmentedSurv-DML (no PCI)",
    }

    def style_bundle(model_names: list[str], labels: dict[str, str]):
        color_map = {}
        style_map = {}
        marker_map = {}
        width_map = {}
        alpha_map = {}
        for name in model_names:
            label = labels[name]
            if label == "B2":
                color_map[name] = common_colors["B2"]
                style_map[name] = "-"
                marker_map[name] = "o"
                width_map[name] = 2.4
                alpha_map[name] = 0.95
            elif label == "Old C3":
                color_map[name] = common_colors["Old C3"]
                style_map[name] = "-"
                marker_map[name] = "s"
                width_map[name] = 2.8
                alpha_map[name] = 0.95
            elif "(PCI)" in label and "Summary" in label:
                color_map[name] = common_colors["Summary PCI"]
                style_map[name] = "-"
                marker_map[name] = "o"
                width_map[name] = 3.0
                alpha_map[name] = 1.0
            elif "(PCI)" in label and "Augmented" in label:
                color_map[name] = common_colors["Augmented PCI"]
                style_map[name] = "-"
                marker_map[name] = "D"
                width_map[name] = 3.0
                alpha_map[name] = 1.0
            elif "(no PCI)" in label and "Summary" in label:
                color_map[name] = common_colors["Summary noPCI"]
                style_map[name] = "--"
                marker_map[name] = "o"
                width_map[name] = 2.2
                alpha_map[name] = 0.9
            else:
                color_map[name] = common_colors["Augmented noPCI"]
                style_map[name] = "--"
                marker_map[name] = "D"
                width_map[name] = 2.2
                alpha_map[name] = 0.9
        return color_map, style_map, marker_map, width_map, alpha_map

    nc_color_map, nc_style_map, nc_marker_map, nc_width_map, nc_alpha_map = style_bundle(nc_models, nc_labels)
    surv_color_map, surv_style_map, surv_marker_map, surv_width_map, surv_alpha_map = style_bundle(surv_models, surv_labels)

    chart_specs = [
        (
            "nc_sample_size_scaling_rmse.png",
            "Non-Censored Sample-Size Scaling (S01-S04)",
            "Fixed p_x = 5 and p_w = p_z = 1. Solid lines: PCI models. Dashed lines: no-PCI matched baselines.",
            ["S01", "S02", "S03", "S04"],
            [1000, 2000, 4000, 8000],
            ["1000", "2000", "4000", "8000"],
            "Sample size n",
            nc_models,
            nc_labels,
            nc_color_map,
            nc_style_map,
            nc_marker_map,
            nc_width_map,
            nc_alpha_map,
            nc_agg,
        ),
        (
            "nc_proxy_scaling_rmse.png",
            "Non-Censored Proxy-Dimensional Scaling (S02, S05, S06, S07)",
            "Fixed n = 2000 and p_x = 5. The x-axis is the shared proxy dimension p_w = p_z.",
            ["S02", "S05", "S06", "S07"],
            [1, 3, 5, 10],
            ["1", "3", "5", "10"],
            "Proxy dimension p_w = p_z",
            nc_models,
            nc_labels,
            nc_color_map,
            nc_style_map,
            nc_marker_map,
            nc_width_map,
            nc_alpha_map,
            nc_agg,
        ),
        (
            "nc_x_scaling_rmse.png",
            "Non-Censored Covariate-Dimensional Scaling (S02, S08, S09)",
            "Fixed n = 2000 and p_w = p_z = 1. The x-axis is the main covariate dimension p_x.",
            ["S02", "S08", "S09"],
            [5, 10, 20],
            ["5", "10", "20"],
            "Covariate dimension p_x",
            nc_models,
            nc_labels,
            nc_color_map,
            nc_style_map,
            nc_marker_map,
            nc_width_map,
            nc_alpha_map,
            nc_agg,
        ),
        (
            "nc_joint_stress_scaling_rmse.png",
            "Non-Censored Joint Complexity Scaling (S10-S14)",
            "Joint stress regime where X and proxy complexity increase together. S12 additionally lowers n.",
            ["S10", "S11", "S12", "S13", "S14"],
            None,
            ["S10", "S11", "S12", "S13", "S14"],
            "Joint-stress setting",
            nc_models,
            nc_labels,
            nc_color_map,
            nc_style_map,
            nc_marker_map,
            nc_width_map,
            nc_alpha_map,
            nc_agg,
        ),
        (
            "surv_sample_size_scaling_rmse.png",
            "Censored Sample-Size Scaling (S01-S04)",
            "Fixed p_x = 5 and p_w = p_z = 1. Solid lines: PCI models. Dashed lines: no-PCI matched baselines.",
            ["S01", "S02", "S03", "S04"],
            [1000, 2000, 4000, 8000],
            ["1000", "2000", "4000", "8000"],
            "Sample size n",
            surv_models,
            surv_labels,
            surv_color_map,
            surv_style_map,
            surv_marker_map,
            surv_width_map,
            surv_alpha_map,
            surv_agg,
        ),
        (
            "surv_proxy_scaling_rmse.png",
            "Censored Proxy-Dimensional Scaling (S02, S05, S06, S07)",
            "Fixed n = 2000 and p_x = 5. The x-axis is the shared proxy dimension p_w = p_z.",
            ["S02", "S05", "S06", "S07"],
            [1, 3, 5, 10],
            ["1", "3", "5", "10"],
            "Proxy dimension p_w = p_z",
            surv_models,
            surv_labels,
            surv_color_map,
            surv_style_map,
            surv_marker_map,
            surv_width_map,
            surv_alpha_map,
            surv_agg,
        ),
        (
            "surv_x_scaling_rmse.png",
            "Censored Covariate-Dimensional Scaling (S02, S08, S09)",
            "Fixed n = 2000 and p_w = p_z = 1. The x-axis is the main covariate dimension p_x.",
            ["S02", "S08", "S09"],
            [5, 10, 20],
            ["5", "10", "20"],
            "Covariate dimension p_x",
            surv_models,
            surv_labels,
            surv_color_map,
            surv_style_map,
            surv_marker_map,
            surv_width_map,
            surv_alpha_map,
            surv_agg,
        ),
        (
            "surv_joint_stress_scaling_rmse.png",
            "Censored Joint Complexity Scaling (S10-S14)",
            "Joint stress regime where X and proxy complexity increase together. Old C3/B2 anchors are unavailable at S14, so their lines stop at S13.",
            ["S10", "S11", "S12", "S13", "S14"],
            None,
            ["S10", "S11", "S12", "S13", "S14"],
            "Joint-stress setting",
            surv_models,
            surv_labels,
            surv_color_map,
            surv_style_map,
            surv_marker_map,
            surv_width_map,
            surv_alpha_map,
            surv_agg,
        ),
    ]

    figures = []
    for (
        filename,
        title,
        subtitle,
        settings,
        x_values,
        x_labels,
        xlabel,
        model_order,
        label_map,
        color_map,
        style_map,
        marker_map,
        width_map,
        alpha_map,
        agg,
    ) in chart_specs:
        figures.append(
            _render_axis_chart(
                title=title,
                subtitle=subtitle,
                output_path=DESKTOP / filename,
                settings=settings,
                x_values=x_values,
                x_labels=x_labels,
                xlabel=xlabel,
                model_order=model_order,
                label_map=label_map,
                color_map=color_map,
                style_map=style_map,
                marker_map=marker_map,
                width_map=width_map,
                alpha_map=alpha_map,
                agg=agg,
            )
        )

    with PdfPages(OUT_PDF) as pdf:
        for fig in figures:
            pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
