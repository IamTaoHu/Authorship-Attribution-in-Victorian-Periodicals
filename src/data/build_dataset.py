"""Phase 1 dataset pipeline orchestration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datasets import DatasetDict

from src.data.cleaning import clean_dataset
from src.data.inspect_dataset import (
    analyze_text_quality,
    get_label_values,
    inspect_dataset,
    label_mapping_report,
    make_periad_class_distribution,
    save_json,
    validate_periad_splits,
)
from src.data.load_datasets import build_label_mapping, tokenize_dataset
from src.data.tokenization_analysis import TOKENIZER_SPECS, load_tokenizer_with_fallbacks, run_tokenization_analysis


def prepare_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create and return Phase 1 output directories."""
    root = Path(output_dir)
    paths = {
        "root": root,
        "periad_cleaned": root / "periad_cleaned",
        "tokenized": root / "tokenized",
        "plots": root / "plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_warnings(warnings: list[str], output_dir: str | Path) -> None:
    """Write collected pipeline warnings to outputs/phase1/warnings.txt."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    warnings_path = output_path / "warnings.txt"
    with warnings_path.open("w", encoding="utf-8") as file:
        if warnings:
            file.write("\n".join(warnings) + "\n")
        else:
            file.write("No warnings.\n")


def save_dataset_inspection_outputs(
    reports: dict[str, dict],
    distributions: list[pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Save combined dataset inspection reports and class distributions."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_json(reports, output_path / "dataset_report.json")

    frames = [frame for frame in distributions if not frame.empty]
    distribution = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["dataset", "split", "label", "count"]
    )
    distribution.to_csv(output_path / "class_distribution.csv", index=False)


def process_periad_dataset(
    dataset: DatasetDict,
    text_column: str,
    label_column: str,
    output_dir: str | Path,
    skip_tokenization: bool,
    max_samples_for_token_analysis: int,
    max_length: int = 512,
    warn: Optional[Callable[[str], None]] = None,
) -> DatasetDict:
    """Run PERIAD-specific Phase 1 processing and save outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    label_report = label_mapping_report(dataset, label_column)
    save_json(label_report, output_path / "label_mapping.json")

    class_distribution = make_periad_class_distribution(dataset, label_column)
    class_distribution.to_csv(output_path / "periad_class_distribution.csv", index=False)
    plot_class_distribution(class_distribution, plots_dir / "periad_samples_per_author.png")

    quality_report, suspicious_examples = analyze_text_quality(dataset, text_column, output_path)
    save_json(quality_report, output_path / "text_quality_report.json")
    suspicious_examples.to_csv(output_path / "suspicious_examples.csv", index=False)

    plot_length_distributions(dataset, text_column, plots_dir)

    cleaned_dataset = clean_dataset(dataset, text_column)
    cleaned_dataset.save_to_disk(str(output_path / "periad_cleaned"))

    if not skip_tokenization:
        run_tokenization_analysis(
            cleaned_dataset,
            text_column=text_column,
            output_dir=output_path,
            max_samples=max_samples_for_token_analysis,
            warn=warn,
        )
        save_tokenized_datasets(
            cleaned_dataset,
            text_column=text_column,
            label_column=label_column,
            output_dir=output_path,
            max_length=max_length,
            warn=warn,
        )

    return cleaned_dataset


def save_tokenized_datasets(
    dataset: DatasetDict,
    text_column: str,
    label_column: str,
    output_dir: str | Path,
    max_length: int,
    warn: Optional[Callable[[str], None]] = None,
) -> None:
    """Tokenize and save PERIAD for each available tokenizer."""
    tokenized_root = Path(output_dir) / "tokenized"
    tokenized_root.mkdir(parents=True, exist_ok=True)
    label_mapping = build_label_mapping(dataset, label_column)

    encoded_splits = {}
    for split_name, split_dataset in dataset.items():
        if "__label_id" in split_dataset.column_names:
            split_dataset = split_dataset.remove_columns(["__label_id"])
        split_labels = get_label_values(split_dataset, label_column)
        encoded_labels = [label_mapping.get(label, -1) for label in split_labels]
        encoded_splits[split_name] = split_dataset.add_column("__label_id", encoded_labels)
    encoded_dataset = DatasetDict(encoded_splits)

    for tokenizer_label, tokenizer_names, safe_name in TOKENIZER_SPECS:
        tokenizer, loaded_name = load_tokenizer_with_fallbacks(tokenizer_names, warn=warn)
        if tokenizer is None or loaded_name is None:
            if warn:
                warn(f"Skipping tokenized dataset save for {tokenizer_label}; tokenizer unavailable.")
            continue
        try:
            tokenized = tokenize_dataset(
                encoded_dataset,
                tokenizer_name=loaded_name,
                text_column=text_column,
                label_column="__label_id",
                max_length=max_length,
            )
            tokenized.save_to_disk(str(tokenized_root / safe_name))
        except Exception as exc:  # noqa: BLE001 - keep pipeline running on tokenizer/schema issues.
            if warn:
                warn(f"Failed to save tokenized dataset for {tokenizer_label}: {exc}")


def plot_class_distribution(distribution: pd.DataFrame, output_path: str | Path) -> None:
    """Plot PERIAD samples per author using matplotlib."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if distribution.empty:
        return

    plot_data = distribution[distribution["split"] == "all"].sort_values("count", ascending=False)
    plt.figure(figsize=(11, 6))
    plt.bar(plot_data["label"], plot_data["count"], color="#4C78A8")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Samples")
    plt.title("PERIAD Samples per Author")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_length_distributions(dataset: DatasetDict, text_column: str, plots_dir: str | Path) -> None:
    """Plot paragraph character and word length distributions into the plots directory."""
    texts: list[str] = []
    for split_dataset in dataset.values():
        if text_column not in split_dataset.column_names:
            continue
        texts.extend("" if value is None else str(value) for value in split_dataset[text_column])

    char_lengths = [len(text) for text in texts]
    word_lengths = [len(text.split()) for text in texts]
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    _plot_histogram(char_lengths, plots_path / "paragraph_char_lengths.png", "Paragraph Character Lengths", "Characters")
    _plot_histogram(word_lengths, plots_path / "paragraph_word_lengths.png", "Paragraph Word Lengths", "Words")


def inspect_and_save_dataset(
    dataset: DatasetDict,
    dataset_name: str,
) -> tuple[dict, pd.DataFrame]:
    """Inspect one dataset and return report artifacts for later combined saving."""
    return inspect_dataset(dataset, dataset_name=dataset_name)


def _plot_histogram(values: list[int], output_path: Path, title: str, xlabel: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, color="#4C78A8", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Paragraph count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
