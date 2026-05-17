"""Validate Phase 6 ensemble outputs."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.ensemble_phase6 import ENSEMBLE_STRATEGIES  # noqa: E402
from src.prompting.phase4_parser import CANONICAL_AUTHORS  # noqa: E402
from src.utils.paths import get_path_config  # noqa: E402


REQUIRED_COLUMNS = {"sample_id", "true_author", "predicted_author", "strategy"}


def check_path(path: Path, *, nonempty: bool = True) -> bool:
    ok = path.exists() and (not nonempty or path.stat().st_size > 0)
    print(f"[{'OK' if ok else 'MISSING'}] {path}")
    return ok


def phase6_dir() -> Path:
    return get_path_config()["artifacts_root"] / "phase6"


def check_prediction_file(path: Path) -> bool:
    if not check_path(path):
        return False
    frame = pd.read_csv(path)
    ok = True
    if frame.empty:
        print(f"[MISSING] {path} has zero rows")
        ok = False
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        print(f"[MISSING] {path} columns: {', '.join(missing)}")
        ok = False
    valid = set(CANONICAL_AUTHORS)
    for column in ("true_author", "predicted_author"):
        if column in frame.columns:
            bad = sorted(set(frame[column].dropna().astype(str)).difference(valid))
            if bad:
                print(f"[MISMATCH] {path} invalid {column} labels: {', '.join(bad)}")
                ok = False
    return ok


def main() -> int:
    root = phase6_dir()
    ok = True
    for directory in ("predictions", "tables", "plots", "reports"):
        ok = check_path(root / directory, nonempty=False) and ok

    results_path = root / "tables" / "ensemble_results.csv"
    single_path = root / "tables" / "single_model_results.csv"
    ok = check_path(results_path) and ok
    ok = check_path(single_path) and ok
    completed_strategies: set[str] = set()
    if results_path.exists():
        results = pd.read_csv(results_path)
        strategies = set(results.get("strategy", pd.Series(dtype=str)).astype(str))
        missing = sorted(set(ENSEMBLE_STRATEGIES).difference(strategies))
        if missing:
            print(f"[MISSING] ensemble_results.csv strategies: {', '.join(missing)}")
            ok = False
        completed_strategies = set(results.loc[results.get("status", "") == "ok", "strategy"].astype(str))

    for name in completed_strategies:
        ok = check_prediction_file(root / "predictions" / f"{name}.csv") and ok
        ok = check_path(root / "plots" / f"confusion_matrix_{name}.png") and ok

    for name in ("per_author_f1.csv", "hard_author_focus.csv"):
        ok = check_path(root / "tables" / name) and ok
    ok = check_path(root / "reports" / "phase6_ensemble_report.md") and ok
    ok = check_path(root / "plots" / "phase6_accuracy_comparison.png") and ok
    ok = check_path(root / "plots" / "phase6_macro_f1_comparison.png") and ok

    if ok:
        print("Phase 6 output check passed.")
        return 0
    print("Phase 6 output check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
