"""Validate Phase 7 LDA outputs."""

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

from src.utils.paths import get_path_config  # noqa: E402


CANONICAL_AUTHORS = (
    "Leslie Stephen",
    "John Morley",
    "Eliza Lynn Linton",
    "George Henry Lewes",
    "Anne Mozley",
    "James Fitzjames Stephen",
)

EXPECTED_TOPIC_COUNTS = {10, 20, 30, 40}
REQUIRED_DIRS = ("tables", "metrics", "models", "plots")
REQUIRED_FILES = (
    "tables/lda_model_selection.csv",
    "tables/best_topics.csv",
    "tables/document_topic_distribution.csv",
    "tables/author_topic_distribution.csv",
    "tables/topic_entropy_by_document.csv",
    "metrics/lda_summary.json",
    "models/best_lda.joblib",
    "models/vectorizer.joblib",
    "plots/model_selection_perplexity.png",
    "plots/model_selection_coherence_or_diversity.png",
    "plots/author_topic_heatmap.png",
    "plots/author_topic_stacked_bar.png",
    "plots/topic_entropy_by_author.png",
)
INTERACTIVE_HTML = "interactive/phase7_lda_pyldavis.html"


def default_phase7_dir() -> Path:
    return get_path_config()["artifacts_root"] / "phase7" / "lda"


def check_path(path: Path, *, nonempty: bool = True) -> bool:
    ok = path.exists() and (not nonempty or path.stat().st_size > 0)
    print(f"[{'OK' if ok else 'MISSING'}] {path}")
    return ok


def topic_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("topic_")]


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        print(f"[MISSING] Phase 1 input CSV for row-count check: {path}")
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


def load_summary(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[MISMATCH] Could not read summary JSON: {exc}")
        return {}


def check_model_selection(root: Path) -> bool:
    path = root / "tables" / "lda_model_selection.csv"
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    ok = True
    missing = EXPECTED_TOPIC_COUNTS.difference(set(frame.get("k", pd.Series(dtype=int)).astype(int)))
    if missing:
        print(f"[MISSING] lda_model_selection.csv k values: {sorted(missing)}")
        ok = False
    if "selected" not in frame.columns or int(frame["selected"].astype(bool).sum()) != 1:
        print("[MISMATCH] lda_model_selection.csv must mark exactly one selected row")
        ok = False
    return ok


def check_document_topics(root: Path, summary: dict) -> bool:
    path = root / "tables" / "document_topic_distribution.csv"
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    ok = True
    expected = expected_input_total(summary)
    summary_total = int(summary.get("input_row_counts", {}).get("total", -1))
    if summary_total <= 0:
        print("[MISMATCH] lda_summary.json missing positive input_row_counts.total")
        ok = False
    if expected is None:
        ok = False
    elif len(frame) != expected:
        print(f"[MISMATCH] document_topic_distribution row count {len(frame)} != expected {expected}")
        ok = False
    elif summary_total != expected:
        print(f"[MISMATCH] lda_summary.json total {summary_total} != Phase 1 CSV total {expected}")
        ok = False
    missing_columns = {"sample_id", "split", "author", "text"}.difference(frame.columns)
    if missing_columns:
        print(f"[MISSING] document_topic_distribution columns: {', '.join(sorted(missing_columns))}")
        ok = False
    columns = topic_columns(frame)
    if not columns:
        print("[MISSING] document_topic_distribution has no topic_* columns")
        return False
    sums = frame[columns].sum(axis=1).to_numpy(dtype=float)
    if not np.allclose(sums, 1.0, atol=1e-5):
        print(f"[MISMATCH] topic probabilities do not sum to 1; max abs error={float(np.max(np.abs(sums - 1.0))):.6g}")
        ok = False
    return ok


def check_author_topics(root: Path) -> bool:
    path = root / "tables" / "author_topic_distribution.csv"
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    ok = True
    if "author" not in frame.columns:
        print("[MISSING] author_topic_distribution.csv column: author")
        return False
    missing = set(CANONICAL_AUTHORS).difference(set(frame["author"].astype(str)))
    if missing:
        print(f"[MISSING] author_topic_distribution.csv canonical authors: {', '.join(sorted(missing))}")
        ok = False
    if not topic_columns(frame):
        print("[MISSING] author_topic_distribution.csv has no topic_* columns")
        ok = False
    return ok


def check_summary(root: Path) -> tuple[bool, dict]:
    path = root / "metrics" / "lda_summary.json"
    if not path.exists():
        return False, {}
    summary = load_summary(path)
    ok = True
    for key in ("selected_k", "random_state", "input_row_counts", "output_paths", "config_values"):
        if key not in summary:
            print(f"[MISSING] lda_summary.json key: {key}")
            ok = False
    if summary.get("random_state") != 42:
        print(f"[MISMATCH] lda_summary.json random_state should be 42, got {summary.get('random_state')}")
        ok = False
    if summary.get("selected_k") not in EXPECTED_TOPIC_COUNTS:
        print(f"[MISMATCH] selected_k should be one of {sorted(EXPECTED_TOPIC_COUNTS)}, got {summary.get('selected_k')}")
        ok = False
    return ok, summary


def check_interactive_html(root: Path) -> bool:
    path = root / INTERACTIVE_HTML
    ok = check_path(path)
    if not ok:
        return False
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        print(f"[MISMATCH] Could not read interactive HTML: {exc}")
        return False
    lowered = content.lower()
    if "pyldavis" not in lowered and "ldavis" not in lowered:
        print("[MISMATCH] interactive HTML does not contain pyLDAvis/ldavis markers")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7_dir", default=None)
    parser.add_argument("--require_interactive", action="store_true")
    args = parser.parse_args()
    root = Path(args.phase7_dir).expanduser() if args.phase7_dir else default_phase7_dir()

    ok = True
    for directory in REQUIRED_DIRS:
        ok = check_path(root / directory, nonempty=False) and ok
    for relative in REQUIRED_FILES:
        ok = check_path(root / relative) and ok

    summary_ok, summary = check_summary(root)
    ok = summary_ok and ok
    ok = check_model_selection(root) and ok
    ok = check_document_topics(root, summary) and ok
    ok = check_author_topics(root) and ok
    if args.require_interactive:
        ok = check_interactive_html(root) and ok

    if ok:
        print("Phase 7 output check passed.")
        return 0
    print("Phase 7 output check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
