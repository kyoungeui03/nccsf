#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path("/Users/kyoungeuihong/Desktop/csf_grf_new")
NC_RUNNER = ROOT / "scripts" / "run_14setting_non_censored_unified_b2sum_family.py"
C_RUNNER = ROOT / "scripts" / "run_14setting_censored_unified_b2sum_family.py"
NC_OUT = ROOT / "non_censored" / "outputs" / "benchmark_structured_14settings_unified_b2sum_final5"
C_OUT = ROOT / "outputs" / "benchmark_structured_14settings_unified_b2sum_final5"
LOGDIR = ROOT / "outputs" / "logs_unified_b2sum_final5_14setting"

Q_A = ["S14", "S07", "S10", "S03", "S12", "S01"]
Q_B = ["S11", "S13", "S04", "S09", "S06", "S08", "S05", "S02"]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "RCPP_PARALLEL_NUM_THREADS": "1",
        }
    )
    return env


def _spawn(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    with log_path.open("wb") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    return proc.pid


def main() -> int:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    NC_OUT.mkdir(parents=True, exist_ok=True)
    C_OUT.mkdir(parents=True, exist_ok=True)

    env = _env()
    jobs = {
        "nc_q1": ["python3", str(NC_RUNNER), "--output-dir", str(NC_OUT), "--setting-ids", *Q_A],
        "nc_q2": ["python3", str(NC_RUNNER), "--output-dir", str(NC_OUT), "--setting-ids", *Q_B],
        "c_q1": ["python3", str(C_RUNNER), "--output-dir", str(C_OUT), "--setting-ids", *Q_A],
        "c_q2": ["python3", str(C_RUNNER), "--output-dir", str(C_OUT), "--setting-ids", *Q_B],
    }

    pid_lines: list[str] = []
    for name, cmd in jobs.items():
        pid = _spawn(cmd, LOGDIR / f"{name}.log", env)
        pid_lines.append(f"{name} {pid}")

    (LOGDIR / "pids.txt").write_text("\n".join(pid_lines) + "\n")
    print("Started unified B2Sum final5 14-setting run.")
    print(f"NC output: {NC_OUT}")
    print(f"C output:  {C_OUT}")
    print(f"Logs:      {LOGDIR}")
    print(f"PIDs:      {LOGDIR / 'pids.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
