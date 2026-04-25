#!/usr/bin/env python3
"""Run the ACTG175 HIV analysis for the R-CSF baseline only.

This wrapper preserves the exact data construction, bootstrap logic, age-profile
evaluation, Table-4 subject sampling, and output generation used by
`run_hiv_actg175_final_conditional.py`, but restricts the run to the
`r_csf_x14` model and makes the paper-style bootstrap setup explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
MODEL_DIR = THIS_FILE.parent
PROJECT_ROOT = MODEL_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

try:  # pragma: no cover
    from .run_hiv_actg175_final_conditional import main as _base_main
except ImportError:  # pragma: no cover
    from single_file_censored_models.run_hiv_actg175_final_conditional import (  # type: ignore
        main as _base_main,
    )


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hiv_actg175_r_csf_split_comparison_10k_bootstrap"
DEFAULT_CACHE_CSV = PROJECT_ROOT / "outputs" / "hiv_actg175_final_conditional" / "actg175_ucimlrepo_cache.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ACTG175 HIV analysis for the installed R "
            "grf::causal_survival_forest baseline only, using the same bootstrap-style "
            "evaluation protocol as the split-comparison 10k bootstrap run."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where runner outputs will be written.",
    )
    parser.add_argument(
        "--cache-csv",
        type=Path,
        default=DEFAULT_CACHE_CSV,
        help="Optional cache path for the fetched UCI dataset.",
    )
    parser.add_argument(
        "--refresh-dataset",
        action="store_true",
        help="Refetch the dataset from ucimlrepo instead of reusing the cached CSV.",
    )
    parser.add_argument(
        "--grf-num-trees",
        type=int,
        default=2000,
        help="Trees in the R-CSF baseline.",
    )
    parser.add_argument(
        "--grf-min-node-size",
        type=int,
        default=5,
        help="Minimum node size for the R-CSF baseline.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=200,
        help="Nonparametric bootstrap repetitions for Figure 4 bands and Table 4 SEs.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
        help="Seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--bootstrap-alpha",
        type=float,
        default=0.05,
        help="Bootstrap central interval level, e.g. 0.05 for 95%% bands.",
    )
    parser.add_argument(
        "--save-every-bootstrap",
        type=int,
        default=10,
        help="Save partial bootstrap draws every N completed bootstrap replications.",
    )
    parser.add_argument(
        "--progress-every-bootstrap",
        type=int,
        default=5,
        help="Print bootstrap progress every N completed bootstrap replications.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from existing per-model outputs and bootstrap checkpoints when available.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing partial outputs and rerun from scratch.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Model random seed.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for the Table-4-style random 10 sample.",
    )
    return parser


def _forward_args(args: argparse.Namespace) -> list[str]:
    forwarded = [
        "--models",
        "r_csf_x14",
        "--output-dir",
        str(args.output_dir),
        "--cache-csv",
        str(args.cache_csv),
        "--grf-num-trees",
        str(args.grf_num_trees),
        "--grf-min-node-size",
        str(args.grf_min_node_size),
        "--bootstrap-reps",
        str(args.bootstrap_reps),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
        "--bootstrap-alpha",
        str(args.bootstrap_alpha),
        "--save-every-bootstrap",
        str(args.save_every_bootstrap),
        "--progress-every-bootstrap",
        str(args.progress_every_bootstrap),
        "--random-state",
        str(args.random_state),
        "--sample-seed",
        str(args.sample_seed),
    ]
    if args.resume:
        forwarded.append("--resume")
    else:
        forwarded.append("--no-resume")
    if args.refresh_dataset:
        forwarded.append("--refresh-dataset")
    return forwarded


def _write_wrapper_config(args: argparse.Namespace, forwarded: list[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "wrapper_script": str(THIS_FILE),
        "base_runner": str(MODEL_DIR / "run_hiv_actg175_final_conditional.py"),
        "fixed_model": "r_csf_x14",
        "goal": "Match the split_comparison_10k_bootstrap HIV setup while running only the R-CSF baseline.",
        "forwarded_args": forwarded,
        "assumptions": [
            "Dataset subset, age-profile construction, Table-4 subject sampling, bootstrap logic, BLP path, and output tables are all delegated to the base HIV runner.",
            "Only the selected model is changed to r_csf_x14; all other experiment logic is inherited from the base runner.",
        ],
    }
    with open(args.output_dir / "wrapper_run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.cache_csv = args.cache_csv.resolve()
    forwarded = _forward_args(args)
    _write_wrapper_config(args, forwarded)
    return int(_base_main(forwarded))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
