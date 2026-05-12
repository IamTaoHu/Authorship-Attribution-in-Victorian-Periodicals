"""Check required Phase 2 generated outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import get_path_config  # noqa: E402


REQUIRED_RUN_FILES = (
    "predictions.csv",
    "eval_metrics.json",
    "classification_report.csv",
    "confusion_matrix.csv",
    "run_config_resolved.yaml",
    "trainer_state.json",
    "training_log.csv",
)
PROBABILITY_COLUMNS = [f"prob_{index}" for index in range(6)]


def check_path(path: Path) -> bool:
    ok = path.exists() and path.stat().st_size > 0
    marker = "OK" if ok else "MISSING"
    print(f"[{marker}] {path}")
    return ok


def check_run(run_dir: Path) -> bool:
    missing = 0
    for filename in REQUIRED_RUN_FILES:
        if not check_path(run_dir / filename):
            missing += 1
    predictions_path = run_dir / "predictions.csv"
    if predictions_path.exists():
        frame = pd.read_csv(predictions_path, nrows=1)
        missing_probability_columns = [column for column in PROBABILITY_COLUMNS if column not in frame.columns]
        if missing_probability_columns:
            print(f"[MISSING] {predictions_path} columns: {', '.join(missing_probability_columns)}")
            missing += 1
        else:
            print(f"[OK] {predictions_path} contains prob_0 through prob_5")
    return missing == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    paths = get_path_config()
    artifacts_root = paths["artifacts_root"]
    phase2_dir = artifacts_root / "phase2"
    runs_dir = phase2_dir / "runs"
    if not runs_dir.exists():
        print(f"Phase 2 output check failed: no runs directory found at {runs_dir}")
        return 1
    completed = [run_dir for run_dir in sorted(runs_dir.iterdir()) if run_dir.is_dir() and check_run(run_dir)]
    table_path = phase2_dir / "tables" / "encoder_results.csv"
    if table_path.exists() and table_path.stat().st_size > 0:
        print(f"[OK] {table_path}")
    else:
        print(f"[WARNING] Aggregate table has not been generated yet: {table_path}")
    if not completed:
        print("Phase 2 output check failed: no completed run found.")
        return 1
    print(f"Phase 2 output check passed: {len(completed)} completed run(s) found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
