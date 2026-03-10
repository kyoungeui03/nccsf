#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))

from grf.synthetic import (  # noqa: E402
    LegacyComparisonConfig,
    SynthConfig,
    add_eq8_eq9_columns,
    add_ground_truth_cate,
    generate_legacy_comparison_nc_cox,
    generate_synthetic_nc_cox,
    standardized_synthetic_scenarios,
)


def main() -> int:
    output_root = PROJECT_ROOT / "data" / "synthetic_scenarios"
    output_root.mkdir(parents=True, exist_ok=True)

    catalog_rows: list[dict[str, object]] = []
    for scenario in standardized_synthetic_scenarios():
        scenario_dir = output_root / scenario.slug
        scenario_dir.mkdir(parents=True, exist_ok=True)

        if scenario.family == "survival":
            config = SynthConfig(**scenario.config)
            observed_df, truth_df, params = generate_synthetic_nc_cox(config)
            observed_df, truth_df = add_ground_truth_cate(observed_df, truth_df, config, params)
        elif scenario.family == "legacy_comparison":
            config = LegacyComparisonConfig(**scenario.config)
            observed_df, truth_df, params = generate_legacy_comparison_nc_cox(config)
            observed_df, truth_df = add_eq8_eq9_columns(observed_df, truth_df, config, params)
        else:
            raise ValueError(f"Unsupported scenario family: {scenario.family}")

        observed_path = scenario_dir / "observed.csv"
        truth_path = scenario_dir / "truth.csv"
        metadata_path = scenario_dir / "metadata.json"

        observed_df.to_csv(observed_path, index=False)
        truth_df.to_csv(truth_path, index=False)
        metadata = {
            "slug": scenario.slug,
            "family": scenario.family,
            "source": scenario.source,
            "title": scenario.title,
            "notes": scenario.notes,
            "config": scenario.config,
            "observed_csv": str(observed_path),
            "truth_csv": str(truth_path),
            "num_rows": int(observed_df.shape[0]),
            "num_features": int(len([column for column in observed_df.columns if column.startswith("X")])),
            "actual_censor_rate": float(1.0 - observed_df["event"].mean()),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        catalog_rows.append(
            {
                "slug": scenario.slug,
                "family": scenario.family,
                "title": scenario.title,
                "source": scenario.source,
                "notes": scenario.notes,
                "num_rows": metadata["num_rows"],
                "num_features": metadata["num_features"],
                "actual_censor_rate": metadata["actual_censor_rate"],
                "observed_csv": str(observed_path),
                "truth_csv": str(truth_path),
            }
        )

    catalog_df = pd.DataFrame(catalog_rows)
    catalog_csv = output_root / "catalog.csv"
    catalog_json = output_root / "catalog.json"
    catalog_df.to_csv(catalog_csv, index=False)
    catalog_json.write_text(json.dumps(catalog_rows, indent=2), encoding="utf-8")

    print(catalog_df[["slug", "title", "actual_censor_rate"]].to_string(index=False))
    print(f"\nSaved synthetic catalog to {catalog_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
