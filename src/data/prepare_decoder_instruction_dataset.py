"""Prepare PERIAD instruction JSONL files for Phase 5 decoder QLoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import warnings

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_datasets import infer_label_column, infer_text_column, normalize_author_label
from src.evaluation.label_mapping import get_author_id_to_name, get_label_to_author_name


DEFAULT_PHASE1_DATASET_PATH = PROJECT_ROOT / "outputs" / "phase1" / "periad_cleaned"
DEFAULT_DATASET_NAME = "celvaigh/periad"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase5" / "data"
EXPECTED_TRAIN_COUNT = 8279
EXPECTED_TEST_COUNT = 3549
CANONICAL_AUTHORS = list(get_author_id_to_name().values())
PROMPT_TEMPLATE = """### Instruction:
Predict the author of the following paragraph. Choose exactly one author from this list:
Leslie Stephen; John Morley; Eliza Lynn Linton; George Henry Lewes; Anne Mozley; James Fitzjames Stephen.

### Paragraph:
{text}

### Response:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 5 decoder instruction dataset.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--phase1_dataset_path", type=Path, default=DEFAULT_PHASE1_DATASET_PATH)
    parser.add_argument("--text_column", default=None)
    parser.add_argument("--label_column", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--max_samples_train", type=int, default=None)
    parser.add_argument("--max_samples_test", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    dataset = load_periad_dataset(resolve_repo_path(args.phase1_dataset_path), args.dataset_name)
    validate_split_counts(dataset, args.train_split, args.test_split)

    train_split = get_split(dataset, args.train_split)
    test_split = get_split(dataset, args.test_split)
    text_column = args.text_column or infer_required_text_column(train_split)
    label_column = args.label_column or infer_required_label_column(train_split)

    train_rows = convert_split(
        train_split,
        split_name=args.train_split,
        text_column=text_column,
        label_column=label_column,
        max_samples=args.max_samples_train,
    )
    test_rows = convert_split(
        test_split,
        split_name=args.test_split,
        text_column=text_column,
        label_column=label_column,
        max_samples=args.max_samples_test,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_rows, output_dir / "train_instruction.jsonl")
    write_jsonl(test_rows, output_dir / "test_instruction.jsonl")
    write_json(build_label_map(), output_dir / "label_map.json")
    write_json(
        build_dataset_stats(
            dataset=dataset,
            train_rows=train_rows,
            test_rows=test_rows,
            text_column=text_column,
            label_column=label_column,
            train_split=args.train_split,
            test_split=args.test_split,
            max_samples_train=args.max_samples_train,
            max_samples_test=args.max_samples_test,
        ),
        output_dir / "dataset_stats.json",
    )
    print(f"Wrote Phase 5 instruction dataset to {output_dir}")
    print(f"Train rows: {len(train_rows)}; test rows: {len(test_rows)}")


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def load_periad_dataset(phase1_dataset_path: Path, dataset_name: str) -> DatasetDict:
    """Load local Phase 1 PERIAD data first, then fall back to Hugging Face."""
    if phase1_dataset_path.exists():
        print(f"Loading local Phase 1 PERIAD dataset: {phase1_dataset_path}")
        return ensure_dataset_dict(load_from_disk(str(phase1_dataset_path)))
    warnings.warn(
        f"Local Phase 1 dataset not found at {phase1_dataset_path}; loading {dataset_name}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return ensure_dataset_dict(load_dataset(dataset_name))


def ensure_dataset_dict(dataset: Dataset | DatasetDict) -> DatasetDict:
    if isinstance(dataset, DatasetDict):
        return dataset
    return DatasetDict({"train": dataset})


def get_split(dataset: DatasetDict, split_name: str) -> Dataset:
    if split_name not in dataset:
        raise ValueError(f"Dataset is missing required split: {split_name!r}. Available splits: {list(dataset)}")
    return dataset[split_name]


def validate_split_counts(dataset: DatasetDict, train_split: str, test_split: str) -> None:
    """Log reference PERIAD split counts and warn if the loaded data differs."""
    if train_split in dataset:
        train_count = len(dataset[train_split])
        print(f"PERIAD {train_split} count: {train_count}")
        if train_count != EXPECTED_TRAIN_COUNT:
            warnings.warn(
                f"PERIAD train count differs from reference {EXPECTED_TRAIN_COUNT}: {train_count}",
                RuntimeWarning,
                stacklevel=2,
            )
    if test_split in dataset:
        test_count = len(dataset[test_split])
        print(f"PERIAD {test_split} count: {test_count}")
        if test_count != EXPECTED_TEST_COUNT:
            warnings.warn(
                f"PERIAD test count differs from reference {EXPECTED_TEST_COUNT}: {test_count}",
                RuntimeWarning,
                stacklevel=2,
            )


def infer_required_text_column(dataset: Dataset) -> str:
    text_column = infer_text_column(dataset)
    if text_column is None:
        raise ValueError(f"Could not infer text column from columns: {dataset.column_names}")
    return text_column


def infer_required_label_column(dataset: Dataset) -> str:
    label_column = infer_label_column(dataset)
    if label_column is None:
        raise ValueError(f"Could not infer label column from columns: {dataset.column_names}")
    return label_column


def convert_split(
    dataset: Dataset,
    split_name: str,
    text_column: str,
    label_column: str,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    author_id_to_name = get_author_id_to_name()
    limit = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    rows: list[dict[str, Any]] = []
    skipped = 0
    for index in range(limit):
        row = dataset[index]
        label = parse_label(row[label_column])
        if label is None or label not in author_id_to_name:
            skipped += 1
            continue
        text = normalize_text(row[text_column])
        if not text:
            skipped += 1
            continue
        author = author_id_to_name[int(label)]
        prompt = render_prompt(text)
        rows.append(
            {
                "sample_id": sample_id_for_row(row, split_name, index),
                "text": text,
                "true_label": int(label),
                "true_author": author,
                "prompt": prompt,
                "response": author,
                "full_text": f"{prompt}{author}",
            }
        )
    if skipped:
        warnings.warn(f"Skipped {skipped} unusable {split_name} rows during Phase 5 conversion.", RuntimeWarning, stacklevel=2)
    validate_instruction_rows(rows)
    return rows


def parse_label(value: Any) -> int | None:
    author_id_to_name = get_author_id_to_name()
    name_to_id = {name: label for label, name in author_id_to_name.items()}
    try:
        label = int(value)
        return label if label in author_id_to_name else None
    except (TypeError, ValueError):
        normalized = normalize_author_label(value)
        return name_to_id.get(normalized)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def render_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(text=text)


def sample_id_for_row(row: dict[str, Any], split_name: str, index: int) -> str:
    for key in ("row_id", "sample_id", "id", "doc_id", "paragraph_id"):
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return f"{split_name}_{index}"


def validate_instruction_rows(rows: list[dict[str, Any]]) -> None:
    valid_authors = set(CANONICAL_AUTHORS)
    invalid = sorted({row["true_author"] for row in rows if row["true_author"] not in valid_authors})
    if invalid:
        raise ValueError(f"Instruction dataset contains non-canonical authors: {invalid}")
    for row in rows[:10]:
        if not str(row["full_text"]).endswith(str(row["response"])):
            raise ValueError("Instruction row full_text must end with the response author label.")
        if "### Response:" not in str(row["prompt"]):
            raise ValueError("Instruction prompt is missing the response marker.")


def build_label_map() -> dict[str, Any]:
    label_to_author = get_label_to_author_name()
    return {
        "label_to_author_name": label_to_author,
        "author_name_to_label": {author: int(label) for label, author in label_to_author.items()},
        "canonical_authors": CANONICAL_AUTHORS,
        "prompt_template": PROMPT_TEMPLATE,
    }


def build_dataset_stats(
    dataset: DatasetDict,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    text_column: str,
    label_column: str,
    train_split: str,
    test_split: str,
    max_samples_train: int | None,
    max_samples_test: int | None,
) -> dict[str, Any]:
    return {
        "source": "outputs/phase1/periad_cleaned" if DEFAULT_PHASE1_DATASET_PATH.exists() else DEFAULT_DATASET_NAME,
        "text_column": text_column,
        "label_column": label_column,
        "train_split": train_split,
        "test_split": test_split,
        "reference_counts": {"train": EXPECTED_TRAIN_COUNT, "test": EXPECTED_TEST_COUNT},
        "loaded_counts": {split: len(dataset[split]) for split in dataset},
        "written_counts": {"train": len(train_rows), "test": len(test_rows)},
        "sample_limits": {"train": max_samples_train, "test": max_samples_test},
        "canonical_authors": CANONICAL_AUTHORS,
    }


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
