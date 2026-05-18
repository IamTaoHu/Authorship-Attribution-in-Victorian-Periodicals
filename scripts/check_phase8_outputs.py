"""Validate Phase 8 BERTopic outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic_modeling.bertopic_phase8 import default_phase8_dir  # noqa: E402
from src.utils.paths import get_path_config  # noqa: E402


REQUIRED_FILES = (
    "embeddings/document_embeddings.npy",
    "assignments/topic_assignments.csv",
    "tables/topic_info.csv",
    "tables/topic_words.csv",
    "tables/per_author_topic_distribution.csv",
    "tables/per_split_topic_distribution.csv",
    "tables/document_topic_features.csv",
    "models/bertopic_model",
    "plots/topic_size_bar.png",
    "plots/per_author_topic_heatmap.png",
    "plots/topic_scatter.png",
    "reports/bertopic_summary.json",
    "reports/bertopic_report.md",
)
POLISHED_FILES = (
    "plots/per_author_top_topics_heatmap.png",
    "plots/topic_scatter_by_author.png",
    "tables/top_topic_words.csv",
    "reports/top_topic_words.md",
)
REQUIRED_ASSIGNMENT_COLUMNS = {"sample_id", "split", "author", "text", "topic", "source_row_index", "source_row_id"}


def check_path(path: Path, *, nonempty: bool = True) -> bool:
    ok = path.exists() and (not nonempty or path.stat().st_size > 0 if path.is_file() else True)
    print(f"[{'OK' if ok else 'MISSING'}] {path}")
    return ok


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def check_outside_repo(root: Path, summary: dict) -> bool:
    ok = True
    candidates = [root]
    output_paths = summary.get("output_paths", {}) if isinstance(summary.get("output_paths", {}), dict) else {}
    for raw_path in output_paths.values():
        if raw_path:
            candidates.append(Path(str(raw_path)))
    for path in candidates:
        if is_relative_to(path, REPO_ROOT):
            print(f"[MISMATCH] Phase 8 output path is inside repo root: {path}")
            ok = False
    return ok


def load_summary(root: Path) -> dict:
    path = root / "reports" / "bertopic_summary.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[MISMATCH] Could not read summary JSON: {exc}")
        return {}


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        print(f"[MISSING] Phase 1 CSV for row-count check: {path}")
        return None
    return int(len(pd.read_csv(path, usecols=["text"])))


def expected_input_total(summary: dict) -> int | None:
    input_paths = summary.get("input_paths", {}) if isinstance(summary.get("input_paths", {}), dict) else {}
    train_path = Path(input_paths.get("train", "")) if input_paths.get("train") else None
    test_path = Path(input_paths.get("test", "")) if input_paths.get("test") else None
    if train_path is None or test_path is None:
        paths = get_path_config()
        processed = paths["datasets_root"] / "processed" / "periad"
        train_path = processed / "train.csv"
        test_path = processed / "test.csv"
    train_count = count_csv_rows(train_path)
    test_count = count_csv_rows(test_path)
    if train_count is None or test_count is None:
        return None
    return train_count + test_count


def check_assignments(root: Path, summary: dict) -> bool:
    path = root / "assignments" / "topic_assignments.csv"
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    ok = True
    missing = sorted(REQUIRED_ASSIGNMENT_COLUMNS.difference(frame.columns))
    if missing:
        print(f"[MISSING] topic_assignments.csv columns: {', '.join(missing)}")
        ok = False
    expected = expected_input_total(summary)
    limit = summary.get("limit")
    expected_rows = int(limit) if limit is not None else expected
    if expected_rows is None:
        ok = False
    elif len(frame) != expected_rows:
        print(f"[MISMATCH] topic_assignments row count {len(frame)} != expected {expected_rows}")
        ok = False
    if frame.get("sample_id", pd.Series(dtype=str)).duplicated().any():
        print("[MISMATCH] topic_assignments.csv contains duplicate sample_id values")
        ok = False
    return ok


def check_topic_info(root: Path) -> bool:
    path = root / "tables" / "topic_info.csv"
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    missing = {"Topic", "Count"}.difference(frame.columns)
    if missing:
        print(f"[MISSING] topic_info.csv columns: {', '.join(sorted(missing))}")
        return False
    if frame.empty:
        print("[MISMATCH] topic_info.csv has zero rows")
        return False
    return True


def check_document_features(root: Path, assignments: pd.DataFrame | None = None) -> bool:
    path = root / "tables" / "document_topic_features.csv"
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    ok = True
    missing = {"sample_id", "split", "author", "topic"}.difference(frame.columns)
    if missing:
        print(f"[MISSING] document_topic_features.csv columns: {', '.join(sorted(missing))}")
        ok = False
    topic_columns = [column for column in frame.columns if column.startswith("topic_")]
    if not topic_columns:
        print("[MISSING] document_topic_features.csv has no topic_* one-hot columns")
        ok = False
    if assignments is None:
        assignment_path = root / "assignments" / "topic_assignments.csv"
        assignments = pd.read_csv(assignment_path) if assignment_path.exists() else None
    if assignments is not None and "topic" in assignments.columns:
        for topic in sorted(assignments["topic"].astype(int).unique().tolist()):
            column = f"topic_{topic}"
            if column not in frame.columns:
                print(f"[MISSING] document_topic_features.csv one-hot column: {column}")
                ok = False
    return ok


def check_probabilities(root: Path, summary: dict, allow_missing: bool) -> bool:
    path = root / "assignments" / "topic_probabilities.npy"
    required = bool(summary.get("calculate_probabilities", False))
    if not path.exists():
        if required and not allow_missing:
            print("[MISSING] topic_probabilities.npy; pass --allow_missing_probabilities only for accepted fallback runs")
            return False
        print("[OK] topic_probabilities.npy missing is allowed for this run")
        return True
    assignments = pd.read_csv(root / "assignments" / "topic_assignments.csv")
    array = np.load(path)
    if array.shape[0] != len(assignments):
        print(f"[MISMATCH] topic_probabilities rows {array.shape[0]} != topic_assignments rows {len(assignments)}")
        return False
    return check_path(path)


def check_interactive(root: Path) -> bool:
    html_files = [path for path in (root / "interactive").glob("*.html") if path.stat().st_size > 0]
    if not html_files:
        print(f"[MISSING] At least one non-empty interactive HTML under {root / 'interactive'}")
        return False
    for path in html_files:
        print(f"[OK] {path}")
    log_path = root / "interactive" / "interactive_export_log.json"
    if log_path.exists():
        print(f"[OK] {log_path}")
    return True


def check_polished_outputs(root: Path) -> bool:
    ok = True
    for relative in POLISHED_FILES:
        ok = check_path(root / relative) and ok
    table_path = root / "tables" / "top_topic_words.csv"
    if table_path.exists():
        frame = pd.read_csv(table_path)
        required = {"topic", "topic_name", "document_count", "rank", "top_words"}
        missing = required.difference(frame.columns)
        if missing:
            print(f"[MISSING] top_topic_words.csv columns: {', '.join(sorted(missing))}")
            ok = False
        if frame.empty:
            print("[MISMATCH] top_topic_words.csv has zero rows")
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase8_dir", default=None)
    parser.add_argument("--allow_missing_probabilities", action="store_true")
    parser.add_argument("--require_polished_visualizations", action="store_true")
    args = parser.parse_args()
    root = Path(args.phase8_dir).expanduser() if args.phase8_dir else default_phase8_dir()

    ok = True
    for directory in ("assignments", "tables", "embeddings", "models", "plots", "interactive", "reports"):
        ok = check_path(root / directory, nonempty=False) and ok
    for relative in REQUIRED_FILES:
        ok = check_path(root / relative, nonempty=not relative.endswith("bertopic_model")) and ok

    summary = load_summary(root)
    ok = check_outside_repo(root, summary) and ok
    ok = check_assignments(root, summary) and ok
    assignment_path = root / "assignments" / "topic_assignments.csv"
    assignments = pd.read_csv(assignment_path) if assignment_path.exists() else None
    ok = check_topic_info(root) and ok
    ok = check_document_features(root, assignments) and ok
    ok = check_probabilities(root, summary, args.allow_missing_probabilities) and ok
    ok = check_interactive(root) and ok
    if args.require_polished_visualizations:
        ok = check_polished_outputs(root) and ok

    if ok:
        print("Phase 8 output check passed.")
        return 0
    print("Phase 8 output check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
