#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/Users/kyoungeuihong/Desktop/csf_grf_new")
DESKTOP = Path("/Users/kyoungeuihong/Desktop")

CANDIDATE_RESULTS = ROOT / "non_censored" / "outputs" / "benchmark_structured_14settings_dml_candidates" / "all_settings_results.csv"
REF_DIR = ROOT / "non_censored" / "outputs" / "benchmark_structured_14settings_extended_full"
OUT_PNG = DESKTOP / "non_censored_sample_size_scaling_rmse.png"

SETTINGS = ["S01", "S02", "S03", "S04"]
N_ORDER = [1000, 2000, 4000, 8000]


def _aggregate_candidates() -> pd.DataFrame:
    df = pd.read_csv(CANDIDATE_RESULTS)
    df = df[
        (df["p_x"] == 5)
        & (df["p_w"] == 1)
        & (df["p_z"] == 1)
        & (df["n"].isin(N_ORDER))
    ].copy()
    setting_map = {1000: "S01", 2000: "S02", 4000: "S03", 8000: "S04"}
    df["setting_id"] = df["n"].map(setting_map)
    return (
        df.groupby(["setting_id", "n", "name"], as_index=False)
        .agg(avg_rmse=("rmse", "mean"))
        .sort_values(["n", "name"])
        .reset_index(drop=True)
    )


def _aggregate_refs() -> pd.DataFrame:
    frames = []
    for sid in SETTINGS:
        matches = sorted(REF_DIR.glob(f"{sid}_*/results.csv"))
        if not matches:
            continue
        frames.append(pd.read_csv(matches[0]))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["name"].isin(["EconML Baseline", "Old C3  Legacy NC-CSF"])]
    return (
        df.groupby(["setting_id", "n", "name"], as_index=False)
        .agg(avg_rmse=("rmse", "mean"))
        .sort_values(["n", "name"])
        .reset_index(drop=True)
    )


def _series_map(df: pd.DataFrame) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name, group in df.groupby("name"):
        ordered = group.sort_values("n")
        out[name] = [float(v) for v in ordered["avg_rmse"]]
    return out


def main() -> int:
    cand = _aggregate_candidates()
    ref = _aggregate_refs()
    data = _series_map(pd.concat([cand, ref], ignore_index=True))

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=220)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    colors = {
        "EconML Baseline": "#9ca3af",
        "Old C3  Legacy NC-CSF": "#111827",
        "Summary-DML (PCI)": "#2563eb",
        "Augmented-DML (PCI)": "#059669",
        "Summary-DML (no PCI)": "#93c5fd",
        "Augmented-DML (no PCI)": "#86efac",
    }
    labels = {
        "EconML Baseline": "EconML Baseline",
        "Old C3  Legacy NC-CSF": "Old C3",
        "Summary-DML (PCI)": "Summary-DML (PCI)",
        "Augmented-DML (PCI)": "Augmented-DML (PCI)",
        "Summary-DML (no PCI)": "Summary-DML (no PCI)",
        "Augmented-DML (no PCI)": "Augmented-DML (no PCI)",
    }
    styles = {
        "EconML Baseline": "-",
        "Old C3  Legacy NC-CSF": "-",
        "Summary-DML (PCI)": "-",
        "Augmented-DML (PCI)": "-",
        "Summary-DML (no PCI)": "--",
        "Augmented-DML (no PCI)": "--",
    }
    widths = {
        "EconML Baseline": 2.4,
        "Old C3  Legacy NC-CSF": 2.8,
        "Summary-DML (PCI)": 3.0,
        "Augmented-DML (PCI)": 3.0,
        "Summary-DML (no PCI)": 2.2,
        "Augmented-DML (no PCI)": 2.2,
    }
    alphas = {
        "EconML Baseline": 0.95,
        "Old C3  Legacy NC-CSF": 0.95,
        "Summary-DML (PCI)": 1.0,
        "Augmented-DML (PCI)": 1.0,
        "Summary-DML (no PCI)": 0.9,
        "Augmented-DML (no PCI)": 0.9,
    }
    markers = {
        "EconML Baseline": "o",
        "Old C3  Legacy NC-CSF": "s",
        "Summary-DML (PCI)": "o",
        "Augmented-DML (PCI)": "D",
        "Summary-DML (no PCI)": "o",
        "Augmented-DML (no PCI)": "D",
    }

    plot_order = [
        "EconML Baseline",
        "Old C3  Legacy NC-CSF",
        "Summary-DML (PCI)",
        "Augmented-DML (PCI)",
        "Summary-DML (no PCI)",
        "Augmented-DML (no PCI)",
    ]

    for name in plot_order:
        ax.plot(
            N_ORDER,
            data[name],
            label=labels[name],
            color=colors[name],
            linestyle=styles[name],
            linewidth=widths[name],
            alpha=alphas[name],
            marker=markers[name],
            markersize=7,
        )

    ax.set_title("Non-Censored Sample-Size Scaling (S01-S04)\nRMSE across the 12 basic synthetic cases", fontsize=19, fontweight="bold", pad=18)
    ax.text(
        0.0,
        1.01,
        "Fixed p_x = 5 and p_w = p_z = 1. Solid lines: PCI models. Dashed lines: no-PCI matched baselines.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color="#667085",
    )
    ax.set_xlabel("Sample size n", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average RMSE", fontsize=13, fontweight="bold")
    ax.set_xticks(N_ORDER)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.9, alpha=0.9)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=10.5,
        ncol=2,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
    )
    for text in legend.get_texts():
        text.set_color("#111827")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
