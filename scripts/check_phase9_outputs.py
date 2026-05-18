"""Validate Phase 9 topic-aware classification outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.topic_features_phase9 import CANONICAL_AUTHORS, default_phase9_dir  # noqa: E402


REQUIRED_DIRS = ("configs", "tables", "metrics", "predictions", "plots", "reports", "models", "logs")
REQUIRED_TABLES = (
    "tables/phase9_results_summary.csv",
    "tables/per_author_f1_comparison.csv",
)
REQUIRED_PLOTS = (
    "plots/phase9_macro_f1_comparison.png",
    "plots/phase9_accuracy_comparison.png",
    "plots/per_author_f1_comparison.png",
    "plots/per_author_f1_delta_vs_text_only.png",
    "plots/confusion_matrix_deberta_topic_concat.png",
    "plots/topic_only_vs_transformer_summary.png",
)
TRANSFORMER_TYPES = {"transformer_text_only", "transformer_topic_concat"}


def check_path(path: Path, *, nonempty: bool = True) -> bool:
    ok = path.exists() and (not nonempty or (path.stat().st_size > 0 if path.is_file() else True))
    print(f"[{'OK' if ok else 'MISSING'}] {path}")
    return ok


def check_summary(root: Path) -> tuple[bool, pd.DataFrame]:
    path = root / "tables" / "phase9_results_summary.csv"
    if not path.exists():
        return False, pd.DataFrame()
    frame = pd.read_csv(path)
    ok = True
    required = {"variant", "model_type", "accuracy", "macro_f1", "weighted_f1"}
    missing = required.difference(frame.columns)
    if missing:
        print(f"[MISSING] phase9_results_summary.csv columns: {', '.join(sorted(missing))}")
        ok = False
    for column in ("accuracy", "macro_f1"):
        values = pd.to_numeric(frame.get(column), errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            print(f"[MISMATCH] {column} must be numeric and between 0 and 1")
            ok = False
    if frame.empty:
        print("[MISMATCH] phase9_results_summary.csv has no rows")
        ok = False
    return ok, frame


def check_predictions(root: Path, summary: pd.DataFrame) -> bool:
    ok = True
    expected_count: int | None = None
    for _, row in summary.iterrows():
        variant = str(row["variant"])
        pred_path = root / "predictions" / f"{variant}_predictions.csv"
        metrics_path = root / "metrics" / f"{variant}_metrics.json"
        ok = check_path(pred_path) and ok
        ok = check_path(metrics_path) and ok
        if not pred_path.exists():
            continue
        frame = pd.read_csv(pred_path)
        id_ok = "sample_id" in frame.columns or "source_row_id" in frame.columns or "source_row_index" in frame.columns
        required = {"true_author", "predicted_author", "correct"}
        missing = required.difference(frame.columns)
        if missing or not id_ok:
            print(f"[MISSING] {pred_path.name} required columns; missing={sorted(missing)} id_column_present={id_ok}")
            ok = False
        if expected_count is None:
            expected_count = len(frame)
        elif len(frame) != expected_count:
            print(f"[MISMATCH] {pred_path.name} row count {len(frame)} != expected {expected_count}")
            ok = False
        authors = set(frame.get("true_author", pd.Series(dtype=str)).astype(str))
        missing_authors = [author for author in CANONICAL_AUTHORS if author not in authors]
        if missing_authors:
            print(f"[MISMATCH] {pred_path.name} true_author missing canonical authors: {missing_authors}")
            ok = False
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            for key in ("macro_f1", "accuracy"):
                value = float(metrics[key])
                if value < 0 or value > 1:
                    raise ValueError
        except Exception:
            print(f"[MISMATCH] {metrics_path.name} must contain numeric macro_f1 and accuracy between 0 and 1")
            ok = False
    return ok


def check_models(root: Path, summary: pd.DataFrame) -> bool:
    ok = True
    for _, row in summary.iterrows():
        if str(row.get("model_type", "")) in TRANSFORMER_TYPES:
            ok = check_path(root / "models" / str(row["variant"]), nonempty=False) and ok
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase9_dir", default=None)
    parser.add_argument("--require_models", action="store_true")
    parser.add_argument("--require_plots", action="store_true")
    args = parser.parse_args()
    root = Path(args.phase9_dir).expanduser() if args.phase9_dir else default_phase9_dir()
    ok = True
    for directory in REQUIRED_DIRS:
        ok = check_path(root / directory, nonempty=False) and ok
    ok = check_path(root / "configs" / "resolved_topic_aware_classification.yaml") and ok
    for relative in REQUIRED_TABLES:
        ok = check_path(root / relative) and ok
    ok = check_path(root / "reports" / "phase9_topic_aware_classification.md") and ok
    summary_ok, summary = check_summary(root)
    ok = summary_ok and ok
    if not summary.empty:
        ok = check_predictions(root, summary) and ok
        if args.require_models:
            ok = check_models(root, summary) and ok
    if args.require_plots:
        for relative in REQUIRED_PLOTS:
            ok = check_path(root / relative) and ok
    if ok:
        print("Phase 9 output check passed.")
        return 0
    print("Phase 9 output check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
