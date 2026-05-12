"""Validate the unified local + Colab project structure."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FOLDERS = (
    "configs",
    "docs",
    "notebooks/local",
    "notebooks/colab",
    "notebooks/archive",
    "prompts",
    "scripts",
    "src",
    "src/utils",
    "src/data",
    "src/training",
    "src/evaluation",
    "src/visualization",
    "tests",
)

REQUIRED_FILES = (
    "configs/paths.example.yaml",
    "configs/datasets.yaml",
    "configs/models.yaml",
    "src/utils/paths.py",
)


def _check_exists(relative_paths: tuple[str, ...], *, expect_dir: bool) -> list[str]:
    missing: list[str] = []
    for relative_path in relative_paths:
        path = REPO_ROOT / relative_path
        ok = path.is_dir() if expect_dir else path.is_file()
        marker = "OK" if ok else "MISSING"
        print(f"[{marker}] {relative_path}")
        if not ok:
            missing.append(relative_path)
    return missing


def main() -> int:
    print(f"Current working directory: {Path.cwd()}")
    print(f"Repository root: {REPO_ROOT}")

    missing_folders = _check_exists(REQUIRED_FOLDERS, expect_dir=True)
    missing_files = _check_exists(REQUIRED_FILES, expect_dir=False)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    try:
        paths_module = importlib.import_module("src.utils.paths")
        print("[OK] Imported src.utils.paths")
    except Exception as exc:
        print(f"[MISSING] Could not import src.utils.paths: {exc}")
        return 1

    try:
        path_config = paths_module.get_path_config()
        print("Resolved paths:")
        for key, value in path_config.items():
            print(f"  {key}: {value}")
    except Exception as exc:
        print(f"[MISSING] Could not load path config: {exc}")
        return 1

    try:
        test_artifact_dir = paths_module.phase_artifact_dir("phase0", "test")
        print(f"[OK] Created test artifact directory: {test_artifact_dir}")
    except Exception as exc:
        print(f"[MISSING] Could not create test artifact directory: {exc}")
        return 1

    if missing_folders or missing_files:
        print("Critical project structure checks failed.")
        return 1

    print("Project structure check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
