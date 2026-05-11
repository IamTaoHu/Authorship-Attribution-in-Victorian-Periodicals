"""Analyze Phase 5 QLoRA prediction errors."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CANONICAL_AUTHORS = [
    "Leslie Stephen",
    "John Morley",
    "Eliza Lynn Linton",
    "George Henry Lewes",
    "Anne Mozley",
    "James Fitzjames Stephen",
]
AUTHOR_LIKE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+){1,3}\b")
TOPIC_TERMS = ("topic", "subject", "theme", "politics", "religion", "review", "essay")
OCR_TERMS = ("\ufffd", "--", "  ", " diffi-", " beauti-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 5 error analysis.")
    parser.add_argument("--eval_dir", type=Path, default=Path("outputs/phase5/eval"))
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/phase5/error_analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [resolve_repo_path(args.run_dir)] if args.run_dir else discover_runs(resolve_repo_path(args.eval_dir))
    output_root = resolve_repo_path(args.output_dir)
    for run_dir in run_dirs:
        analyze_run(run_dir, output_root / run_dir.name)


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def discover_runs(eval_dir: Path) -> list[Path]:
    if not eval_dir.exists():
        return []
    return sorted(path for path in eval_dir.iterdir() if path.is_dir() and (path / "predictions.csv").exists())


def analyze_run(run_dir: Path, output_dir: Path) -> None:
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.exists():
        print(f"Skipping {run_dir.name}: missing predictions.csv")
        return
    df = pd.read_csv(predictions_path)
    df = add_analysis_columns(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    false_positives = df[(~df["correct_bool"]) & (df["predicted_author"].isin(CANONICAL_AUTHORS))].copy()
    false_negatives = df[~df["correct_bool"]].copy()
    invalid_outputs = df[df["is_invalid_output"]].copy()
    hard_pairs = build_hard_pairs(false_negatives)
    false_positives.to_csv(output_dir / "false_positives.csv", index=False)
    false_negatives.to_csv(output_dir / "false_negatives.csv", index=False)
    invalid_outputs.to_csv(output_dir / "invalid_outputs.csv", index=False)
    hard_pairs.to_csv(output_dir / "hard_author_pairs.csv", index=False)
    write_summary(df, hard_pairs, output_dir / "error_summary.md", run_dir.name)
    print(f"Wrote Phase 5 error analysis to {output_dir}")


def add_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    for column in ("predicted_author", "true_author", "parsed_status", "raw_output", "text"):
        if column not in frame.columns:
            frame[column] = ""
    frame["predicted_author"] = frame["predicted_author"].map(normalize)
    frame["true_author"] = frame["true_author"].map(normalize)
    frame["correct_bool"] = frame.apply(lambda row: parse_bool(row.get("correct")) or row["predicted_author"] == row["true_author"], axis=1)
    frame["is_invalid_output"] = ~frame["predicted_author"].isin(CANONICAL_AUTHORS) | ~frame["parsed_status"].fillna("").astype(str).str.startswith("ok")
    frame["is_hallucination"] = frame.apply(is_hallucination, axis=1)
    frame["is_over_generation"] = frame["raw_output"].fillna("").astype(str).str.split().map(len) > 12
    frame["topic_memorization_marker"] = frame["raw_output"].fillna("").astype(str).str.lower().map(lambda text: any(term in text for term in TOPIC_TERMS))
    frame["ocr_sensitivity_marker"] = frame["text"].fillna("").astype(str).map(lambda text: any(term in text for term in OCR_TERMS))
    frame["focus_author_marker"] = frame["true_author"].isin({"James Fitzjames Stephen", "Eliza Lynn Linton"})
    return frame


def normalize(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def is_hallucination(row: pd.Series) -> bool:
    if not bool(row.get("is_invalid_output")):
        return False
    raw = str(row.get("raw_output") or "")
    normalized_valid = {author.lower() for author in CANONICAL_AUTHORS}
    for candidate in AUTHOR_LIKE_RE.findall(raw):
        if candidate.lower() not in normalized_valid and candidate not in {"Final Answer", "The Author"}:
            return True
    return False


def build_hard_pairs(errors: pd.DataFrame) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame(columns=["true_author", "predicted_author", "count", "percentage_of_errors"])
    pairs = errors.groupby(["true_author", "predicted_author"], dropna=False).size().reset_index(name="count")
    pairs["percentage_of_errors"] = pairs["count"] / len(errors) * 100.0
    return pairs.sort_values("count", ascending=False)


def write_summary(df: pd.DataFrame, hard_pairs: pd.DataFrame, output_path: Path, run_name: str) -> None:
    total = len(df)
    accuracy = float(df["correct_bool"].mean()) if total else 0.0
    invalid_rate = float(df["is_invalid_output"].mean()) if total else 0.0
    lines = [
        f"# Phase 5 Error Summary: {run_name}",
        "",
        f"- Samples: {total}",
        f"- Accuracy: {accuracy:.4f}",
        f"- Invalid output rate: {invalid_rate:.4f}",
        f"- Hallucinated author outputs: {int(df['is_hallucination'].sum())}",
        f"- Over-generation outputs: {int(df['is_over_generation'].sum())}",
        f"- Topic memorization markers: {int(df['topic_memorization_marker'].sum())}",
        f"- OCR sensitivity markers: {int(df['ocr_sensitivity_marker'].sum())}",
        f"- James Fitzjames Stephen / Eliza Lynn Linton focused rows: {int(df['focus_author_marker'].sum())}",
        "",
        "## Hard Author Pairs",
        "",
        dataframe_to_markdown(hard_pairs.head(10)),
        "",
        "## Notes",
        "",
        "- Inspect invalid outputs for hallucination and answer-format drift.",
        "- Inspect hard author pairs for confused author style boundaries.",
        "- Compare focus-author rows against Phase 3 and Phase 4 errors before using Phase 6 ensemble outputs.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(format_value(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
