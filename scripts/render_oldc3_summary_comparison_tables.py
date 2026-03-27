#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DESKTOP = Path("/Users/kyoungeuihong/Desktop")
NON_CENSORED_PNG = DESKTOP / "old_c3_vs_summary_dml_comparison.png"
CENSORED_PNG = DESKTOP / "old_c3_vs_summarysurv_dml_comparison.png"


def _wrap_cell(text: str, width: int) -> str:
    return fill(str(text), width=width, break_long_words=False, break_on_hyphens=False)


def _wrap_rows(rows: list[list[str]], widths: list[int]) -> tuple[list[list[str]], list[int]]:
    wrapped_rows: list[list[str]] = []
    line_counts: list[int] = []
    for row in rows:
        wrapped = [_wrap_cell(cell, width) for cell, width in zip(row, widths)]
        wrapped_rows.append(wrapped)
        line_counts.append(max(cell.count("\n") + 1 for cell in wrapped))
    return wrapped_rows, line_counts


def _render_table(*, title: str, subtitle: str, rows: list[list[str]], output_path: Path) -> None:
    wrapped_rows, line_counts = _wrap_rows(rows, [24, 42, 42, 46])

    fig, ax = plt.subplots(figsize=(20, 11.5), dpi=220)
    bg = "#ffffff"
    header_bg = "#243042"
    edge = "#cbd5e1"
    text = "#222222"
    muted = "#667085"
    alt_bg = "#ffffff"
    cell_bg = "#f3f4f6"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")

    fig.text(0.015, 0.972, title, fontsize=20, fontweight="bold", color=text, ha="left", va="top")
    fig.text(0.015, 0.942, subtitle, fontsize=10.5, color=muted, ha="left", va="top")

    columns = ["Axis", "Old C3", "Summary Model", "Why It Matters / Notes"]
    col_widths = [0.18, 0.27, 0.27, 0.28]
    table = ax.table(
        cellText=wrapped_rows,
        colLabels=columns,
        colLoc="center",
        cellLoc="left",
        bbox=[0.0, 0.02, 1.0, 0.90],
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.2)
    table.scale(1, 1.0)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(edge)
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.get_text().set_color("#ffffff")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("center")
        else:
            cell.set_facecolor(alt_bg if r % 2 == 1 else cell_bg)
            cell.get_text().set_color(text)
            if c == 0:
                cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("left")
            cell.get_text().set_va("center")
            cell.get_text().set_wrap(True)
            cell.PAD = 0.045

    header_height = 0.064
    total_body_units = sum(line + 0.8 for line in line_counts)
    available_body_height = 0.90 - header_height
    unit_height = available_body_height / total_body_units

    for c in range(len(columns)):
        table[(0, c)].set_height(header_height)

    for idx, lines in enumerate(line_counts, start=1):
        row_height = unit_height * (lines + 0.8)
        for c in range(len(columns)):
            table[(idx, c)].set_height(row_height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    non_censored_rows = [
        [
            "Overall philosophy",
            "Uses PCI mainly for residualization.",
            "Uses the same PCI nuisance layer, but passes nuisance summaries into the final-stage predictor.",
            "The main change is not the nuisance family, but what the final DML forest sees.",
        ],
        [
            "Treatment nuisance q",
            "Legacy logistic nuisance with duplicated-proxy handling (dup).",
            "Same q family and same old C3 hyperparameters.",
            "This keeps the comparison clean. Summary-DML does not switch to the richer New C3 nuisance.",
        ],
        [
            "Outcome nuisance h",
            "RandomForestRegressor, 300 trees, min leaf 20.",
            "Same h family and same old C3 hyperparameters.",
            "Performance gains are not coming from a stronger nuisance learner.",
        ],
        [
            "Final learner",
            "CausalForestDML.",
            "CausalForestDML.",
            "The learner family stays matched. The change is the final-stage feature representation.",
        ],
        [
            "Final heterogeneity input",
            "Raw [X, W, Z].",
            "Summary X_final = [X, q_pred, h1_pred, h0_pred, m_pred].",
            "Old C3 splits directly on raw proxies. Summary-DML splits on nuisance-derived summaries.",
        ],
        [
            "Nuisance feature mode",
            "dup",
            "dup",
            "Both models use the same old C3 nuisance expansion. Summary-DML is not using interact.",
        ],
        [
            "Raw proxy access at final stage",
            "Yes. The final forest directly sees raw W and Z.",
            "No. The final forest does not directly see raw W and Z.",
            "This is the key conceptual change.",
        ],
        [
            "Pipeline structure",
            "Single legacy DML pipeline.",
            "Outer cross-fitted nuisance stage builds q and h summaries, then a matched DML model is fit on X_final.",
            "The extra outer stage exists only to construct the summary representation.",
        ],
        [
            "No-PCI comparator",
            "Old E2: same legacy DML family, nuisance proxies removed.",
            "Summary-DML (no PCI): same summary pipeline, but nuisance proxies removed when building q and h summaries.",
            "This makes the PCI effect visible inside the same summary-based architecture.",
        ],
        [
            "Core claim",
            "PCI is used, but not directly reflected in the final-stage representation.",
            "PCI-generated summaries directly shape the final-stage representation.",
            "This is why Summary-DML is the strongest simplified non-censored candidate.",
        ],
    ]

    censored_rows = [
        [
            "Overall philosophy",
            "Uses PCI mainly for residualization in the legacy censored DML pipeline.",
            "Uses the same PCI nuisance layer, but passes bridge and survival summaries into the final-stage predictor.",
            "The main change is again the final-stage representation, not the nuisance family.",
        ],
        [
            "Treatment nuisance q",
            "Legacy logistic nuisance with duplicated-proxy handling (dup).",
            "Same q family and same old C3 hyperparameters.",
            "This is a matched comparison, not a switch to the New C3 nuisance design.",
        ],
        [
            "Outcome nuisance h",
            "ExtraTreesRegressor, 800 trees, min leaf 5.",
            "Same h family and same old C3 hyperparameters.",
            "The censored summary model keeps the same old C3 bridge learner.",
        ],
        [
            "Censoring nuisance",
            "Kaplan-Meier by default.",
            "Same Kaplan-Meier censoring estimator.",
            "The censored comparison keeps the old C3 censoring pipeline matched.",
        ],
        [
            "Final learner",
            "CausalForestDML.",
            "CausalForestDML.",
            "The learner family stays matched. Only the final-stage representation is changed.",
        ],
        [
            "Final heterogeneity input",
            "Raw [X, W, Z].",
            "Summary X_final = [X, q_pred, h1_pred, h0_pred, m_pred, surv1_pred, surv0_pred, surv_diff_pred].",
            "Old C3 splits on raw proxies. SummarySurv-DML splits on bridge and survival summaries.",
        ],
        [
            "Raw proxy access at final stage",
            "Yes. The final forest directly sees raw W and Z.",
            "No. The final forest does not directly see raw W and Z.",
            "This is the cleanest censored version of the proxy-to-summary transition.",
        ],
        [
            "Pipeline structure",
            "Single legacy censored DML pipeline.",
            "Outer cross-fitted nuisance stage builds q, h, and survival summaries, then a matched DML model is fit on X_final.",
            "The extra outer stage exists only to construct summary features before the final DML fit.",
        ],
        [
            "No-PCI comparator",
            "Old E2: same legacy censored DML family, nuisance proxies removed.",
            "SummarySurv-DML (no PCI): same summary pipeline, but nuisance proxies removed when building q, h, and survival summaries.",
            "This isolates the PCI contribution inside the same summary-based censored architecture.",
        ],
        [
            "Core claim",
            "PCI is used, but not directly reflected in the final-stage representation.",
            "PCI-generated bridge and survival summaries directly shape the final-stage representation.",
            "This is why SummarySurv-DML is the strongest simplified censored candidate.",
        ],
    ]

    _render_table(
        title="Old C3 vs Summary-DML (PCI): Code-Level Comparison",
        subtitle="Slide-ready summary of matched nuisance design and final-stage representation (non-censored).",
        rows=non_censored_rows,
        output_path=NON_CENSORED_PNG,
    )
    _render_table(
        title="Old C3 vs SummarySurv-DML (PCI): Code-Level Comparison",
        subtitle="Slide-ready summary of matched nuisance design and final-stage representation (censored).",
        rows=censored_rows,
        output_path=CENSORED_PNG,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
