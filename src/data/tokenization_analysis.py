"""Tokenizer length analysis for paragraph-level authorship data."""

from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Any, Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase


TOKENIZER_SPECS = [
    ("bert-base-uncased", ["bert-base-uncased"], "bert-base-uncased"),
    ("deberta-v3-base", ["microsoft/deberta-v3-base"], "deberta-v3-base"),
    (
        "mistral-7b",
        ["mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.3"],
        "mistral-7b",
    ),
]
TRUNCATION_LIMITS = (512, 1024, 2048, 4096)


def load_tokenizer_with_fallbacks(
    tokenizer_names: list[str],
    warn: Optional[Callable[[str], None]] = None,
) -> tuple[Optional[PreTrainedTokenizerBase], Optional[str]]:
    """Load the first available tokenizer from a list of fallback names."""
    for tokenizer_name in tokenizer_names:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer, tokenizer_name
        except Exception as exc:  # noqa: BLE001 - tokenizer loading can fail for many HF/network reasons.
            if warn:
                warn(f"Tokenizer load failed for {tokenizer_name}: {exc}")
    return None, None


def collect_text_sample(
    dataset: DatasetDict,
    text_column: str,
    max_samples: int,
) -> list[str]:
    """Collect a deterministic text sample across splits for tokenization analysis."""
    texts: list[str] = []
    for split_dataset in dataset.values():
        if text_column not in split_dataset.column_names:
            continue
        for value in split_dataset[text_column]:
            texts.append("" if value is None else str(value))
            if len(texts) >= max_samples:
                return texts
    return texts


def compute_token_lengths(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    """Compute untruncated token lengths for a list of texts."""
    lengths: list[int] = []
    for text in texts:
        encoded = tokenizer(text, truncation=False, add_special_tokens=True)
        lengths.append(len(encoded["input_ids"]))
    return lengths


def summarize_token_lengths(
    tokenizer_label: str,
    tokenizer_name: str,
    lengths: list[int],
) -> dict[str, Any]:
    """Summarize token lengths and truncation rates for one tokenizer."""
    if not lengths:
        return {
            "tokenizer": tokenizer_label,
            "tokenizer_name": tokenizer_name,
            "num_texts": 0,
            "avg_token_length": None,
            "median_token_length": None,
            "p90_token_length": None,
            "p95_token_length": None,
            "max_token_length": None,
            **{f"truncation_rate_at_{limit}": None for limit in TRUNCATION_LIMITS},
        }

    sorted_lengths = sorted(lengths)
    summary: dict[str, Any] = {
        "tokenizer": tokenizer_label,
        "tokenizer_name": tokenizer_name,
        "num_texts": len(lengths),
        "avg_token_length": sum(lengths) / len(lengths),
        "median_token_length": median(lengths),
        "p90_token_length": _percentile(sorted_lengths, 0.90),
        "p95_token_length": _percentile(sorted_lengths, 0.95),
        "max_token_length": max(lengths),
    }
    for limit in TRUNCATION_LIMITS:
        summary[f"truncation_rate_at_{limit}"] = sum(length > limit for length in lengths) / len(lengths)
    return summary


def run_tokenization_analysis(
    dataset: DatasetDict,
    text_column: str,
    output_dir: str | Path,
    max_samples: int,
    warn: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """Run token length analysis for configured tokenizers and save reports/plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenized_dir = output_path / "tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    texts = collect_text_sample(dataset, text_column, max_samples)
    rows: list[dict[str, Any]] = []

    for tokenizer_label, tokenizer_names, safe_name in TOKENIZER_SPECS:
        tokenizer, loaded_name = load_tokenizer_with_fallbacks(tokenizer_names, warn=warn)
        if tokenizer is None or loaded_name is None:
            if warn:
                warn(f"Skipping tokenizer analysis for {tokenizer_label}; no fallback tokenizer loaded.")
            continue

        lengths = compute_token_lengths(texts, tokenizer)
        rows.append(summarize_token_lengths(tokenizer_label, loaded_name, lengths))
        plot_token_length_histogram(
            lengths,
            output_path / f"token_length_histogram_{safe_name}.png",
            title=f"Token Lengths: {tokenizer_label}",
        )

    report = pd.DataFrame(rows)
    report.to_csv(output_path / "tokenization_report.csv", index=False)
    return report


def plot_token_length_histogram(lengths: list[int], output_path: str | Path, title: str) -> None:
    """Save a matplotlib histogram for token lengths."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color="#4C78A8", edgecolor="white")
    plt.title(title)
    plt.xlabel("Token length")
    plt.ylabel("Paragraph count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _percentile(sorted_values: list[int], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
