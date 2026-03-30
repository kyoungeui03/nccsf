from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_rscript() -> str:
    """Return a usable Rscript executable path.

    Resolution order:
    1. RSCRIPT environment variable
    2. PATH lookup for Rscript / Rscript.exe
    3. R_HOME/bin/Rscript(.exe)
    4. Common macOS/Linux install locations
    """

    candidates: list[str | None] = [
        os.environ.get("RSCRIPT"),
        shutil.which("Rscript"),
        shutil.which("Rscript.exe"),
    ]

    r_home = os.environ.get("R_HOME")
    if r_home:
        candidates.extend(
            [
                str(Path(r_home) / "bin" / "Rscript"),
                str(Path(r_home) / "bin" / "Rscript.exe"),
            ]
        )

    candidates.extend(
        [
            "/usr/local/bin/Rscript",
            "/opt/homebrew/bin/Rscript",
            "/Library/Frameworks/R.framework/Resources/bin/Rscript",
        ]
    )

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Rscript executable not found. Install R so that `Rscript` is available on PATH, "
        "or set RSCRIPT=/absolute/path/to/Rscript before running the benchmark."
    )
