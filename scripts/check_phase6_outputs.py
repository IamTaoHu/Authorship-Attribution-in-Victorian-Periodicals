"""Validate Phase 6 ensemble outputs."""

from __future__ import annotations

import argparse
import json
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
VOTING_IMPROVEMENT_FILES = (
    "plots/ensemble_macro_f1_gain_over_best_single.png",
    "plots/ensemble_accuracy_gain_over_best_single.png",
    "plots/hard_author_f1_improvement.png",
    "plots/per_author_best_ensemble_vs_best_single_f1_delta.png",
    "metrics/voting_improvement_summary.json",
)
VOTING_SUMMARY_KEYS = {
    "best_single_model_name",
    "best_ensemble_name",
    "macro_f1_gain",
    "accuracy_gain",
    "generated_plot_paths",
}


def check_path(path: Path, *, nonempty: bool = True) -> bool:
    ok = path.exists() and (not nonempty or path.stat().st_size > 0)
    print(f"[{'OK' if ok else 'MISSING'}] {path}")
    return ok


def phase6_dir() -> Path:
    return get_path_config()["artifacts_root"] / "phase6"


def resolve_phase6_dir(raw: str | None) -> Path:
    if raw is None:
        return phase6_dir()
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


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


def check_voting_improvement_outputs(root: Path) -> bool:
    ok = True
    for relative in VOTING_IMPROVEMENT_FILES:
        ok = check_path(root / relative) and ok

    summary_path = root / "metrics" / "voting_improvement_summary.json"
    if not summary_path.exists() or summary_path.stat().st_size <= 0:
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[MISMATCH] {summary_path} is not valid JSON: {exc}")
        return False

    missing = sorted(VOTING_SUMMARY_KEYS.difference(summary))
    if missing:
        print(f"[MISSING] {summary_path} keys: {', '.join(missing)}")
        ok = False

    generated_paths = summary.get("generated_plot_paths")
    if not isinstance(generated_paths, list) or not generated_paths:
        print(f"[MISMATCH] {summary_path} generated_plot_paths must be a non-empty list")
        return False

    for raw_path in generated_paths:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = root / path
        ok = check_path(path) and ok
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase6_dir", default=None, help="Phase 6 artifact directory. Defaults to configured artifacts_root/phase6.")
    parser.add_argument("--require_voting_improvement_plots", action="store_true", help="Require Phase 6 voting improvement plots and summary JSON.")
    args = parser.parse_args()

    root = resolve_phase6_dir(args.phase6_dir)
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
    if args.require_voting_improvement_plots:
        ok = check_voting_improvement_outputs(root) and ok

    if ok:
        print("Phase 6 output check passed.")
        return 0
    print("Phase 6 output check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
