"""Dataset loading and reusable dataset preparation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


PERIAD_DATASET_NAME = "TODO_REPLACE_WITH_HF_DATASET_NAME"
MODIFIED_VEAA_DATASET_NAME = "NicholasSynovic/Modified-VEAA"

TEXT_COLUMN_CANDIDATES = (
    "text",
    "paragraph",
    "content",
    "article",
    "sentence",
    "body",
    "document",
)
LABEL_COLUMN_CANDIDATES = (
    "label",
    "author",
    "author_name",
    "class",
    "target",
    "authors",
    "name",
)


def load_periad(dataset_name: str, cache_dir: Optional[str] = None) -> DatasetDict:
    """Load PERIAD from Hugging Face using a configurable dataset name."""
    if not dataset_name or dataset_name == PERIAD_DATASET_NAME:
        raise ValueError(
            "PERIAD dataset name is missing. Pass --periad_dataset_name with the Hugging Face dataset identifier."
        )
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    return _ensure_dataset_dict(dataset)


def load_modified_veaa(
    cache_dir: Optional[str] = None,
    dataset_name: str = MODIFIED_VEAA_DATASET_NAME,
) -> DatasetDict:
    """Load ModifiedVEAA from Hugging Face, defaulting to NicholasSynovic/ModifiedVEAA."""
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    return _ensure_dataset_dict(dataset)


def infer_text_column(dataset: Dataset | DatasetDict) -> Optional[str]:
    """Infer the most likely text column from a Dataset or DatasetDict."""
    sample_dataset = _first_dataset(dataset)
    columns = list(sample_dataset.column_names)
    lowered = {column.lower(): column for column in columns}

    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]

    string_columns = _string_columns(sample_dataset)
    return string_columns[0] if string_columns else None


def infer_label_column(dataset: Dataset | DatasetDict) -> Optional[str]:
    """Infer the most likely label or author column from a Dataset or DatasetDict."""
    sample_dataset = _first_dataset(dataset)
    columns = list(sample_dataset.column_names)
    lowered = {column.lower(): column for column in columns}

    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]

    for column in columns:
        feature = sample_dataset.features.get(column)
        if isinstance(feature, ClassLabel):
            return column

    return None


def build_label_mapping(dataset: Dataset | DatasetDict, label_column: str) -> dict[str, int]:
    """Build a stable label-to-id mapping for string, numeric, or ClassLabel labels."""
    sample_dataset = _first_dataset(dataset)
    feature = sample_dataset.features.get(label_column)

    if isinstance(feature, ClassLabel):
        labels = [normalize_author_label(name) for name in feature.names]
    else:
        values: list[str] = []
        datasets = dataset.values() if isinstance(dataset, DatasetDict) else [dataset]
        for split_dataset in datasets:
            if label_column not in split_dataset.column_names:
                continue
            values.extend(normalize_author_label(value) for value in split_dataset[label_column])
        labels = values

    unique_labels = sorted({label for label in labels if label.lower() not in {"", "none", "nan"}})
    return {label: index for index, label in enumerate(unique_labels)}


def normalize_author_label(value: Any) -> str:
    """Normalize author labels while preserving stable author identity."""
    label = str(value).strip()
    compact = " ".join(label.split())
    lowered = compact.lower()

    if lowered in {"georges lewes", "george lewes", "george henry lewes"}:
        return "George Henry Lewes"
    return compact


def clean_dataset(dataset: Dataset | DatasetDict, text_column: str) -> Dataset | DatasetDict:
    """Clean a dataset's text column in place by returning a mapped dataset copy."""
    from src.data.cleaning import clean_text

    def _clean_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        return {text_column: [clean_text(value) for value in batch[text_column]]}

    return dataset.map(_clean_batch, batched=True)


def tokenize_dataset(
    dataset: Dataset | DatasetDict,
    tokenizer_name: str,
    text_column: str,
    label_column: str,
    max_length: int,
) -> Dataset | DatasetDict:
    """Tokenize a dataset for later model training without running any training."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(
            [str(text) for text in batch[text_column]],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        if label_column in batch:
            tokenized["labels"] = batch[label_column]
        return tokenized

    return dataset.map(_tokenize_batch, batched=True)


def build_torch_dataloader(
    tokenized_dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build a PyTorch DataLoader from a tokenized Hugging Face Dataset."""
    columns = [
        column
        for column in ("input_ids", "attention_mask", "token_type_ids", "labels")
        if column in tokenized_dataset.column_names
    ]
    torch_dataset = tokenized_dataset.with_format("torch", columns=columns)
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)


def safe_dataset_name(name: str) -> str:
    """Convert a Hugging Face dataset or tokenizer name to a filesystem-safe name."""
    return name.rstrip("/").split("/")[-1].lower().replace("_", "-")


def ensure_output_dir(path: str | Path) -> Path:
    """Create an output directory and return it as a Path."""
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _ensure_dataset_dict(dataset: Dataset | DatasetDict) -> DatasetDict:
    if isinstance(dataset, DatasetDict):
        return dataset
    return DatasetDict({"train": dataset})


def _first_dataset(dataset: Dataset | DatasetDict) -> Dataset:
    if isinstance(dataset, DatasetDict):
        first_split = next(iter(dataset))
        return dataset[first_split]
    return dataset


def _string_columns(dataset: Dataset) -> list[str]:
    string_columns: list[str] = []
    sample_count = min(len(dataset), 20)
    for column in dataset.column_names:
        values = dataset[column][:sample_count] if sample_count else []
        if any(isinstance(value, str) for value in values):
            string_columns.append(column)
    return string_columns
