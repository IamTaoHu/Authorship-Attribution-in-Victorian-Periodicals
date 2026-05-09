"""Select reproducible train-only few-shot examples for Phase 4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys
import warnings
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_datasets import infer_label_column, infer_text_column, normalize_author_label
from src.evaluation.label_mapping import get_author_id_to_name


DEFAULT_PHASE1_DATASET_PATH = PROJECT_ROOT / "outputs" / "phase1" / "periad_cleaned"
DEFAULT_DATASET_NAME = "celvaigh/periad"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "configs" / "phase4" / "few_shot_examples.json"
TARGET_EXAMPLES_PER_AUTHOR = 2
MIN_CHARS = 400
MAX_CHARS = 2200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Phase 4 few-shot examples from the PERIAD train split.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = select_and_save_few_shot_examples(args.output, seed=args.seed)
    print(f"Wrote {len(payload['examples'])} few-shot examples to {args.output}")


def select_and_save_few_shot_examples(output_path: str | Path = DEFAULT_OUTPUT_PATH, seed: int = 42) -> dict[str, Any]:
    """Select train-only examples and save them as JSON."""
    dataset = load_periad_dataset()
    train_split = get_train_split(dataset)
    text_column, label_column = infer_required_columns(train_split)
    examples = select_few_shot_examples(train_split, text_column, label_column, seed)
    payload = {
        "created_by": "src/inference/few_shot_selection.py",
        "seed": int(seed),
        "source_split": "train",
        "examples": examples,
    }
    path = resolve_repo_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_periad_dataset() -> DatasetDict:
    """Load local Phase 1 PERIAD data if present, otherwise fallback to Hugging Face."""
    if DEFAULT_PHASE1_DATASET_PATH.exists():
        return ensure_dataset_dict(load_from_disk(str(DEFAULT_PHASE1_DATASET_PATH)))
    warnings.warn(
        f"Local Phase 1 dataset not found at {DEFAULT_PHASE1_DATASET_PATH}; loading {DEFAULT_DATASET_NAME}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return ensure_dataset_dict(load_dataset(DEFAULT_DATASET_NAME))


def ensure_dataset_dict(dataset: Dataset | DatasetDict) -> DatasetDict:
    """Ensure a loaded dataset is represented as a DatasetDict."""
    if isinstance(dataset, DatasetDict):
        return dataset
    return DatasetDict({"train": dataset})


def get_train_split(dataset: DatasetDict) -> Dataset:
    """Return the train split, or fail clearly."""
    if "train" not in dataset:
        raise ValueError("PERIAD dataset must contain a train split for few-shot example selection.")
    return dataset["train"]


def get_test_split(dataset: DatasetDict) -> Dataset:
    """Return the test split, or fail clearly."""
    if "test" not in dataset:
        raise ValueError("PERIAD dataset must contain a test split for Phase 4 prompting.")
    return dataset["test"]


def infer_required_columns(dataset: Dataset) -> tuple[str, str]:
    """Infer text and label columns for a PERIAD split."""
    text_column = infer_text_column(dataset)
    label_column = infer_label_column(dataset)
    if text_column is None or label_column is None:
        raise ValueError(
            f"Could not infer text/label columns from dataset columns: {dataset.column_names}"
        )
    return text_column, label_column


def select_few_shot_examples(dataset: Dataset, text_column: str, label_column: str, seed: int) -> list[dict[str, Any]]:
    """Select two clean examples per canonical author where possible."""
    rng = random.Random(seed)
    author_id_to_name = get_author_id_to_name()
    candidate_rows: dict[int, list[dict[str, Any]]] = {label: [] for label in author_id_to_name}
    fallback_rows: dict[int, list[dict[str, Any]]] = {label: [] for label in author_id_to_name}

    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    for index in indices:
        row = dataset[index]
        label = parse_label(row[label_column])
        if label not in author_id_to_name:
            continue
        text = " ".join(str(row[text_column]).split())
        if not text:
            continue
        item = {
            "sample_id": sample_id_for_row(row, index),
            "label": int(label),
            "author": author_id_to_name[label],
            "text": text,
        }
        if is_clean_example(text):
            candidate_rows[label].append(item)
        elif is_usable_fallback(text):
            fallback_rows[label].append(item)

    selected: list[dict[str, Any]] = []
    for label, author in author_id_to_name.items():
        examples = candidate_rows[label][:TARGET_EXAMPLES_PER_AUTHOR]
        if len(examples) < TARGET_EXAMPLES_PER_AUTHOR:
            needed = TARGET_EXAMPLES_PER_AUTHOR - len(examples)
            examples.extend(fallback_rows[label][:needed])
        if len(examples) < TARGET_EXAMPLES_PER_AUTHOR:
            warnings.warn(
                f"Only found {len(examples)} usable few-shot examples for {author}; "
                f"target is {TARGET_EXAMPLES_PER_AUTHOR}. Saving best available train examples.",
                RuntimeWarning,
                stacklevel=2,
            )
        selected.extend(examples)
    selected.sort(key=lambda item: (int(item["label"]), str(item["sample_id"])))
    return selected


def parse_label(value: Any) -> int | None:
    """Parse numeric or author-name labels into canonical numeric labels."""
    author_id_to_name = get_author_id_to_name()
    name_to_id = {name: label for label, name in author_id_to_name.items()}
    try:
        label = int(value)
        return label if label in author_id_to_name else None
    except (TypeError, ValueError):
        normalized = normalize_author_label(value)
        return name_to_id.get(normalized)


def sample_id_for_row(row: dict[str, Any], index: int) -> str:
    """Return a stable sample identifier for a dataset row."""
    for key in ("row_id", "sample_id", "id", "doc_id", "paragraph_id"):
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return f"train_{index}"


def is_clean_example(text: str) -> bool:
    """Return whether text satisfies strict few-shot quality filters."""
    length = len(text)
    return MIN_CHARS <= length <= MAX_CHARS and not has_obvious_ocr_noise(text)


def is_usable_fallback(text: str) -> bool:
    """Return whether text is usable if strict filtering undershoots."""
    return 250 <= len(text) <= 3000 and not has_obvious_ocr_noise(text, relaxed=True)


def has_obvious_ocr_noise(text: str, relaxed: bool = False) -> bool:
    """Detect replacement characters and obvious OCR garbage."""
    if "\ufffd" in text:
        return True
    if text.count("?") > (8 if relaxed else 4):
        return True
    compact = text.replace(" ", "")
    if not compact:
        return True
    non_word_ratio = len(re.findall(r"[^A-Za-z0-9\s.,;:'\"!?()\-]", text)) / max(len(text), 1)
    if non_word_ratio > (0.08 if relaxed else 0.04):
        return True
    alpha_ratio = sum(char.isalpha() for char in compact) / max(len(compact), 1)
    return alpha_ratio < (0.55 if relaxed else 0.65)


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve repo-relative paths."""
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


if __name__ == "__main__":
    main()
