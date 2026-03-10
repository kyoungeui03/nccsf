from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = PROJECT_ROOT / "core" / "src"
CORE_THIRD_PARTY = PROJECT_ROOT / "core" / "third_party"
NATIVE_SRC = PROJECT_ROOT / "native" / "src"
NATIVE_LIB = PROJECT_ROOT / "native" / "lib"


def _library_name() -> str:
    if sys.platform == "darwin":
        return "libcsfgrf.dylib"
    if sys.platform.startswith("linux"):
        return "libcsfgrf.so"
    raise RuntimeError(f"Unsupported platform: {sys.platform}")


def _linker_flags() -> list[str]:
    if sys.platform == "darwin":
        return ["-dynamiclib"]
    return ["-shared"]


def build_native(force: bool = False) -> Path:
    output_path = NATIVE_LIB / _library_name()
    NATIVE_LIB.mkdir(parents=True, exist_ok=True)

    source_files = sorted(str(path) for path in CORE_SRC.rglob("*.cpp"))
    source_files.extend(sorted(str(path) for path in NATIVE_SRC.rglob("*.cpp")))
    if not source_files:
        raise RuntimeError("No C++ source files found for the native build.")

    if not force and output_path.exists():
        newest_source_mtime = max(Path(path).stat().st_mtime for path in source_files)
        if output_path.stat().st_mtime >= newest_source_mtime:
            return output_path

    command = [
        "c++",
        "-std=c++17",
        "-O3",
        "-fPIC",
        *(_linker_flags()),
        "-I",
        str(CORE_THIRD_PARTY),
        "-I",
        str(CORE_SRC),
        *source_files,
        "-o",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local csf_grf native library.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the output library is up to date.")
    args = parser.parse_args()

    output_path = build_native(force=args.force)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
