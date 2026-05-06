"""Dataset inspection, label distribution, and text-quality reporting."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict

from src.data.load_datasets import (
    build_label_mapping,
    infer_label_column,
    infer_text_column,
    normalize_author_label,
)


EXPECTED_PERIAD_SPLITS = {"train": 8279, "test": 3549, "total": 11828}


def inspect_dataset(
    dataset: DatasetDict,
    dataset_name: str,
    max_examples: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Inspect all splits in a DatasetDict and return a JSON report plus distribution rows."""
    report: dict[str, Any] = {
        "dataset_name": dataset_name,
        "splits": {},
        "inferred_text_column": infer_text_column(dataset),
        "inferred_label_column": infer_label_column(dataset),
        "text_column_candidates": _column_candidates(dataset, text_like=True),
        "label_column_candidates": _column_candidates(dataset, text_like=False),
    }
    distribution_rows: list[dict[str, Any]] = []

    for split_name, split_dataset in dataset.items():
        text_column = infer_text_column(split_dataset)
        label_column = infer_label_column(split_dataset)
        split_report = inspect_split(
            split_dataset,
            split_name=split_name,
            text_column=text_column,
            label_column=label_column,
            max_examples=max_examples,
        )
        report["splits"][split_name] = split_report
        for label, count in split_report.get("class_distribution", {}).items():
            distribution_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "label": label,
                    "count": count,
                }
            )

    return report, pd.DataFrame(distribution_rows)


def inspect_split(
    dataset: Dataset,
    split_name: str,
    text_column: Optional[str],
    label_column: Optional[str],
    max_examples: int = 3,
) -> dict[str, Any]:
    """Inspect one dataset split for schema, examples, labels, and paragraph lengths."""
    examples = [
        _json_safe_row(dataset[index])
        for index in range(min(max_examples, len(dataset)))
    ]

    report: dict[str, Any] = {
        "split": split_name,
        "columns": list(dataset.column_names),
        "num_rows": len(dataset),
        "examples": examples,
        "text_column": text_column,
        "label_column": label_column,
        "unique_labels": [],
        "class_distribution": {},
        "avg_paragraph_length_chars": None,
        "avg_paragraph_length_words": None,
    }

    if label_column and label_column in dataset.column_names:
        labels = get_label_values(dataset, label_column)
        label_counts = Counter(labels)
        report["unique_labels"] = sorted(label_counts)
        report["class_distribution"] = dict(sorted(label_counts.items()))

    if text_column and text_column in dataset.column_names and len(dataset) > 0:
        texts = [str(value) for value in dataset[text_column]]
        char_lengths = [len(text) for text in texts]
        word_lengths = [len(text.split()) for text in texts]
        report["avg_paragraph_length_chars"] = float(sum(char_lengths) / len(char_lengths))
        report["avg_paragraph_length_words"] = float(sum(word_lengths) / len(word_lengths))

    return report


def get_label_values(dataset: Dataset, label_column: str) -> list[str]:
    """Return normalized display labels from string, numeric, or ClassLabel values."""
    feature = dataset.features.get(label_column)
    values = dataset[label_column]
    if isinstance(feature, ClassLabel):
        labels = [
            normalize_author_label(_class_label_to_str(feature, value)) if value is not None else ""
            for value in values
        ]
    else:
        labels = [normalize_author_label(value) for value in values]
    return labels


def validate_periad_splits(dataset: DatasetDict) -> list[str]:
    """Validate PERIAD split sizes against the paper and return warnings."""
    warnings: list[str] = []
    train_rows = len(dataset["train"]) if "train" in dataset else 0
    test_rows = len(dataset["test"]) if "test" in dataset else 0
    total_rows = sum(len(split) for split in dataset.values())

    observed = {"train": train_rows, "test": test_rows, "total": total_rows}
    for key, expected in EXPECTED_PERIAD_SPLITS.items():
        actual = observed[key]
        if actual != expected:
            warnings.append(
                f"PERIAD {key} row count differs from paper: expected {expected}, observed {actual}."
            )
    return warnings


def analyze_text_quality(
    dataset: DatasetDict,
    text_column: str,
    output_dir: str | Path,
    max_examples: int = 200,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Analyze OCR/text-quality issues and save suspicious examples."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = defaultdict(int)
    suspicious_rows: list[dict[str, Any]] = []
    seen_texts: dict[str, tuple[str, int]] = {}
    char_lengths: list[int] = []
    word_lengths: list[int] = []

    for split_name, split_dataset in dataset.items():
        if text_column not in split_dataset.column_names:
            continue
        for row_index, text_value in enumerate(split_dataset[text_column]):
            text = "" if text_value is None else str(text_value)
            char_lengths.append(len(text))
            word_lengths.append(len(text.split()))
            reasons = _text_quality_reasons(text)
            normalized = " ".join(text.split()).lower()
            if normalized in seen_texts and normalized:
                reasons.append("duplicate_paragraph")
            elif normalized:
                seen_texts[normalized] = (split_name, row_index)

            for reason in reasons:
                counters[reason] += 1

            if reasons and len(suspicious_rows) < max_examples:
                suspicious_rows.append(
                    {
                        "split": split_name,
                        "row_index": row_index,
                        "reasons": ";".join(sorted(set(reasons))),
                        "text_preview": text[:500],
                    }
                )

    report = {
        "text_column": text_column,
        "num_paragraphs": len(char_lengths),
        "issue_counts": dict(sorted(counters.items())),
        "duplicate_paragraphs": counters.get("duplicate_paragraph", 0),
        "short_paragraph_threshold_words": 20,
        "long_paragraph_threshold_words": 1000,
        "avg_chars": _mean(char_lengths),
        "avg_words": _mean(word_lengths),
        "max_chars": max(char_lengths) if char_lengths else 0,
        "max_words": max(word_lengths) if word_lengths else 0,
    }
    suspicious_df = pd.DataFrame(suspicious_rows)
    return report, suspicious_df


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save dictionary data as pretty JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def make_periad_class_distribution(dataset: DatasetDict, label_column: str) -> pd.DataFrame:
    """Create PERIAD class distribution rows for train, test, and all splits."""
    rows: list[dict[str, Any]] = []
    all_counts: Counter[str] = Counter()

    for split_name, split_dataset in dataset.items():
        if label_column not in split_dataset.column_names:
            continue
        split_counts = Counter(get_label_values(split_dataset, label_column))
        all_counts.update(split_counts)
        for label, count in sorted(split_counts.items()):
            rows.append({"split": split_name, "label": label, "count": count})

    for label, count in sorted(all_counts.items()):
        rows.append({"split": "all", "label": label, "count": count})

    return pd.DataFrame(rows)


def label_mapping_report(dataset: DatasetDict, label_column: str) -> dict[str, Any]:
    """Build the serializable PERIAD label mapping report."""
    mapping = build_label_mapping(dataset, label_column)
    return {"label_to_id": mapping, "id_to_label": {str(index): label for label, index in mapping.items()}}


def _text_quality_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    if re.search(r"[^\x00-\x7F]", text):
        reasons.append("non_ascii_characters")
    if "\ufffd" in text:
        reasons.append("encoding_replacement_character")
    if re.search(r"[^\w\s.,;:!?'\-\"()\[\]{}]", text, flags=re.UNICODE):
        reasons.append("strange_symbols")
    if re.search(r" {2,}", text):
        reasons.append("repeated_spaces")
    if len(text.split()) < 20:
        reasons.append("very_short_paragraph")
    if len(text.split()) > 1000:
        reasons.append("very_long_paragraph")
    if re.search(r"\b(?:[A-Za-z]\s){4,}[A-Za-z]\b", text):
        reasons.append("spaced_letters_ocr_artifact")
    if re.search(r"\w\|\w|[Il1]{4,}|rn", text):
        reasons.append("suspicious_ocr_artifact")
    if re.search(r"[.,;:!?]{4,}", text):
        reasons.append("punctuation_anomaly")
    if any(ord(character) < 32 and character not in "\n\r\t" for character in text):
        reasons.append("control_characters")
    return reasons


def _class_label_to_str(feature: ClassLabel, value: Any) -> str:
    try:
        return feature.int2str(int(value))
    except Exception:  # noqa: BLE001 - keep reports robust to malformed label ids.
        return str(value)


def _column_candidates(dataset: DatasetDict, text_like: bool) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {}
    for split_name, split_dataset in dataset.items():
        split_candidates: list[str] = []
        for column in split_dataset.column_names:
            values = split_dataset[column][: min(len(split_dataset), 20)]
            if text_like and any(isinstance(value, str) for value in values):
                split_candidates.append(column)
            if not text_like:
                feature = split_dataset.features.get(column)
                if isinstance(feature, ClassLabel) or column.lower() in {"label", "author", "author_name", "class", "target"}:
                    split_candidates.append(column)
        candidates[split_name] = split_candidates
    return candidates


def _json_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe_value(value) for key, value in row.items()}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    return str(value)


def _mean(values: list[int]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None
