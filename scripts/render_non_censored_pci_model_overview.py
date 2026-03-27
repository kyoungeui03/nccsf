#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from textwrap import fill
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DESKTOP = Path("/Users/kyoungeuihong/Desktop")
SUMMARY_CSV = (
    Path("/Users/kyoungeuihong/Desktop/csf_grf_new")
    / "non_censored/outputs/benchmark_structured_14settings_pci_contenders/all_settings_summary.csv"
)
OUTPUT_PNG = DESKTOP / "non_censored_pci_model_overview.png"


MODEL_INFO = {
    "BroadAugSP (PCI)": {
        "structure": "Single-pass DML\nBroad nuisance base [X,W,Z]\nq=logit, h=extra(1200,3)",
        "final_input": "[X, W, Z, q, h1, h0, m]",
        "notes": "Best overall RMSE and MAE. Uses econml internal folds to build bridge summaries and fit the final forest in one path.",
    },
    "BroadAugDML (PCI)": {
        "structure": "Two-stage DML\nBroad nuisance base [X,W,Z]\nq=logit, h=extra(1200,3)",
        "final_input": "[X, W, Z, q, h1, h0, m]",
        "notes": "Broad-input version of augmented DML. Strong overall, but slower than the single-pass broad variant with no accuracy gain.",
    },
    "B2Sum (PCI)": {
        "structure": "Baseline CF + summary block\nq=logit, h=extra(1200,3)",
        "final_input": "[X, W, Z, q, h1, h0, m]",
        "notes": "Simplest summary-augmented baseline. Keeps the PureB2 learner and only appends PCI bridge summaries to the raw feature block.",
    },
    "AugDML (PCI)": {
        "structure": "Two-stage DML\nNarrow nuisance split\nq=logit, h=extra(1200,3)",
        "final_input": "[X, W, Z, q, h1, h0, m]",
        "notes": "Two-stage augmented DML with task-specific nuisance inputs q:[X,Z] and h:[X,W]. Strong but not competitive enough against the broad versions.",
    },
    "AugSP (PCI)": {
        "structure": "Single-pass DML\nNarrow nuisance split\nq=logit, h=extra(1200,3)",
        "final_input": "[X, W, Z, q, h1, h0, m]",
        "notes": "Cleaner single-pass version of AugDML. More robust case-wise than B2Sum, but slightly behind the best broad single-pass model.",
    },
    "OldC3Raw (PCI)": {
        "structure": "Legacy DML\nRaw final-stage input\nq=logit, h=extra(1200,3)",
        "final_input": "[X, W, Z]",
        "notes": "PCI enters only through nuisance residualization. The final forest still splits on raw proxies, with no explicit summary block.",
    },
    "PureB2": {
        "structure": "Baseline CF\nNo summary stage",
        "final_input": "[X, W, Z]",
        "notes": "Raw baseline anchor. No bridge summary block and no external nuisance-summary construction.",
    },
}


def _wrap(text: str, width: int) -> str:
    return fill(str(text), width=width, break_long_words=False, break_on_hyphens=False)


def _format_time(value: str) -> str:
    return f"{float(value):.2f}s"


def _result_block(row: dict[str, str]) -> str:
    return (
        f"RMSE {float(row['avg_rmse']):.5f}\n"
        f"MAE {float(row['avg_mae']):.5f}\n"
        f"Pearson {float(row['avg_pearson']):.5f}\n"
        f"Time {_format_time(row['avg_time'])}"
    )


def _load_rows() -> list[list[str]]:
    with SUMMARY_CSV.open(newline="") as f:
        summary_rows = list(csv.DictReader(f))

    rows: list[list[str]] = []
    for row in summary_rows:
        name = row["name"]
        info = MODEL_INFO[name]
        rows.append(
            [
                row["rank"],
                name,
                info["structure"],
                info["final_input"],
                info["notes"],
                _result_block(row),
            ]
        )
    return rows


def _wrap_rows(rows: list[list[str]], widths: list[int]) -> tuple[list[list[str]], list[int]]:
    wrapped_rows: list[list[str]] = []
    line_counts: list[int] = []
    for row in rows:
        wrapped = [_wrap(cell, width) for cell, width in zip(row, widths)]
        wrapped_rows.append(wrapped)
        line_counts.append(max(cell.count("\n") + 1 for cell in wrapped))
    return wrapped_rows, line_counts


def main() -> int:
    rows = _load_rows()
    wrapped_rows, line_counts = _wrap_rows(rows, [4, 18, 24, 22, 42, 16])

    fig, ax = plt.subplots(figsize=(24, 11.5), dpi=220)
    bg = "#ffffff"
    header_bg = "#243042"
    edge = "#cbd5e1"
    text = "#222222"
    muted = "#667085"
    alt_bg = "#ffffff"
    cell_bg = "#f3f4f6"
    best_bg = "#eaf7f0"
    second_bg = "#f5fbff"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")

    fig.text(
        0.015,
        0.975,
        "Non-Censored PCI Contenders: Model Overview",
        fontsize=21,
        fontweight="bold",
        color=text,
        ha="left",
        va="top",
    )
    fig.text(
        0.015,
        0.946,
        "Structured 14-setting benchmark. Table summarizes architecture, final-stage representation, and average results.",
        fontsize=10.8,
        color=muted,
        ha="left",
        va="top",
    )

    columns = ["Rank", "Model", "Structure", "Final Input", "Description", "14-Setting Result"]
    col_widths = [0.05, 0.14, 0.18, 0.14, 0.31, 0.18]
    table = ax.table(
        cellText=wrapped_rows,
        colLabels=columns,
        colLoc="center",
        cellLoc="left",
        bbox=[0.0, 0.02, 1.0, 0.90],
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)
    table.scale(1, 1.0)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(edge)
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.get_text().set_color("#ffffff")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("center")
            cell.get_text().set_va("center")
        else:
            if r == 1:
                face = best_bg
            elif r == 2:
                face = second_bg
            else:
                face = alt_bg if r % 2 == 1 else cell_bg
            cell.set_facecolor(face)
            cell.get_text().set_color(text)
            if c in (0, 1):
                cell.get_text().set_fontweight("bold")
            if c == 0:
                cell.get_text().set_ha("center")
            else:
                cell.get_text().set_ha("left")
            cell.get_text().set_va("center")
            cell.get_text().set_wrap(True)
            cell.PAD = 0.038

    header_height = 0.065
    total_body_units = sum(line + 0.9 for line in line_counts)
    available_body_height = 0.90 - header_height
    unit_height = available_body_height / total_body_units

    for c in range(len(columns)):
        table[(0, c)].set_height(header_height)

    for idx, lines in enumerate(line_counts, start=1):
        row_height = unit_height * (lines + 0.9)
        for c in range(len(columns)):
            table[(idx, c)].set_height(row_height)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PNG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
