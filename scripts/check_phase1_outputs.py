"""Check required Phase 1 generated outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import get_path_config  # noqa: E402


REQUIRED_DATASET_FILES = (
    ("processed", "periad", "train.csv"),
    ("processed", "periad", "test.csv"),
    ("processed", "periad", "train.jsonl"),
    ("processed", "periad", "test.jsonl"),
    ("processed", "periad", "label_map.json"),
    ("processed", "periad", "dataset_card.json"),
    ("processed", "veaa", "metadata_summary.json"),
)

REQUIRED_TABLES = (
    "periad_split_summary.csv",
    "periad_author_distribution.csv",
    "periad_length_statistics.csv",
    "veaa_summary.csv",
    "tokenization_stats.csv",
    "tokenization_stats.json",
)

REQUIRED_REPORTS = (
    "periad_validation_report.json",
    "periad_validation_report.md",
    "phase1_dataset_report.md",
)

REQUIRED_PLOTS = (
    "periad_author_distribution.png",
    "periad_text_length_histogram.png",
    "periad_split_distribution.png",
)


def check_path(path: Path) -> bool:
    ok = path.exists() and path.stat().st_size > 0
    marker = "OK" if ok else "MISSING"
    print(f"[{marker}] {path}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1/dataset_pipeline.yaml")
    parser.parse_args()

    paths = get_path_config()
    datasets_root = paths["datasets_root"]
    artifacts_root = paths["artifacts_root"]
    missing = 0
    for parts in REQUIRED_DATASET_FILES:
        if not check_path(datasets_root.joinpath(*parts)):
            missing += 1
    table_dir = artifacts_root / "phase1" / "tables"
    report_dir = artifacts_root / "phase1" / "reports"
    plot_dir = artifacts_root / "phase1" / "plots"
    for filename in REQUIRED_TABLES:
        if not check_path(table_dir / filename):
            missing += 1
    for filename in REQUIRED_REPORTS:
        if not check_path(report_dir / filename):
            missing += 1
    for filename in REQUIRED_PLOTS:
        if not check_path(plot_dir / filename):
            missing += 1

    if missing:
        print(f"Phase 1 output check failed: {missing} required files missing or empty.")
        return 1
    print("Phase 1 output check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
