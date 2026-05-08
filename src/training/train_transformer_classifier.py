"""Reusable Hugging Face Trainer workflow for transformer classifiers."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from src.evaluation.error_analysis import save_error_analysis
from src.evaluation.label_mapping import (
    get_author_id_to_name,
    get_label_names,
    probability_column_name,
    save_author_mapping,
    validate_author_labels,
)
from src.evaluation.metrics import (
    build_classification_report,
    build_confusion_matrices,
    compute_classification_metrics,
    trainer_compute_metrics,
)
from src.evaluation.plots import save_confusion_matrix_plot
from src.evaluation.reports import (
    build_baseline_comparison,
    build_split_summary,
    save_baseline_comparison,
    save_classification_report,
    save_confusion_matrix_csv,
    save_json,
)
from src.models.bert_classifier import load_sequence_classifier, load_tokenizer
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed


TEXT_COLUMN = "text"
LABEL_COLUMN = "author"


@dataclass(frozen=True)
class TransformerTrainingConfig:
    """Configuration for one transformer classification run."""

    model_name: str = "bert-base-uncased"
    seed: int = 42
    epochs: float = 3.0
    learning_rate: float = 2e-5
    batch_size: int = 8
    weight_decay: float = 0.01
    max_length: int = 512
    validation_size: float = 0.1
    output_dir: Path = Path("outputs/phase2")
    phase1_dataset_path: Path = Path("outputs/phase1/periad_cleaned")
    save_total_limit: int = 2
    fp16: str = "auto"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_test_samples: int | None = None
    run_name: str | None = None


def run_transformer_baseline(config: TransformerTrainingConfig) -> dict[str, Any]:
    """Train, evaluate, and save artifacts for one transformer baseline run."""
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    paths = prepare_phase2_dirs(output_dir)
    run_slug = config.run_name or f"bert_base_seed{config.seed}"
    logger = setup_logger(run_slug, log_file=str(paths["logs"] / f"{run_slug}.log"))
    logger.info("Starting Phase 2 transformer baseline: %s", run_slug)

    label_id_to_name = get_author_id_to_name()
    label_ids = sorted(label_id_to_name)
    label_names = get_label_names(label_ids)
    save_author_mapping(paths["reports"] / "author_label_mapping.json")

    raw_dataset = load_phase1_dataset(config.phase1_dataset_path)
    validate_dataset_schema(raw_dataset)
    dataset = add_sample_ids(raw_dataset)
    warnings = validate_author_labels(list(dataset["train"][LABEL_COLUMN]) + list(dataset["test"][LABEL_COLUMN]))
    for warning in warnings:
        logger.warning(warning)

    train_split, validation_split = make_validation_split(dataset["train"], config.validation_size, config.seed)
    train_split = maybe_limit_split(train_split, config.max_train_samples)
    validation_split = maybe_limit_split(validation_split, config.max_eval_samples)
    test_split = maybe_limit_split(dataset["test"], config.max_test_samples)
    split_summary = build_split_summary(
        {
            "train": [int(label) for label in train_split[LABEL_COLUMN]],
            "validation": [int(label) for label in validation_split[LABEL_COLUMN]],
            "test": [int(label) for label in test_split[LABEL_COLUMN]],
        },
        label_id_to_name,
    )
    save_json(split_summary, paths["reports"] / "split_summary.json")

    tokenizer = load_tokenizer(config.model_name)
    model = load_sequence_classifier(config.model_name, label_id_to_name)
    tokenized = DatasetDict(
        {
            "train": tokenize_for_training(train_split, tokenizer, config.max_length),
            "validation": tokenize_for_training(validation_split, tokenizer, config.max_length),
            "test": tokenize_for_training(test_split, tokenizer, config.max_length),
        }
    )

    training_args = build_training_arguments(config, paths["checkpoints"] / run_slug, run_slug)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_metrics = trainer_compute_metrics(label_ids, label_names)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    validation_metrics = trainer.evaluate(tokenized["validation"], metric_key_prefix="validation")
    prediction_output = trainer.predict(tokenized["test"], metric_key_prefix="test")

    logits = prediction_output.predictions
    probabilities = softmax(logits)
    y_true = np.asarray(prediction_output.label_ids, dtype=int)
    y_pred = probabilities.argmax(axis=1)
    test_metrics = compute_classification_metrics(y_true, y_pred, label_ids, label_names)
    test_metrics["validation"] = {key: float(value) for key, value in validation_metrics.items() if isinstance(value, (int, float))}
    test_metrics["model_name"] = config.model_name
    test_metrics["seed"] = config.seed
    test_metrics["run_name"] = run_slug
    save_json(test_metrics, paths["metrics"] / f"{run_slug}_metrics.json")

    report = build_classification_report(y_true, y_pred, label_ids, label_names)
    save_classification_report(
        report,
        paths["reports"] / f"{run_slug}_classification_report.json",
        paths["reports"] / f"{run_slug}_classification_report.csv",
    )

    raw_cm, normalized_cm = build_confusion_matrices(y_true, y_pred, label_ids)
    save_confusion_matrix_csv(raw_cm, label_names, paths["reports"] / f"{run_slug}_confusion_matrix.csv")
    save_confusion_matrix_csv(
        normalized_cm,
        label_names,
        paths["reports"] / f"{run_slug}_confusion_matrix_normalized.csv",
    )
    save_confusion_matrix_plot(raw_cm, label_names, paths["plots"] / f"{run_slug}_confusion_matrix.png")

    predictions = build_prediction_frame(test_split, y_true, y_pred, probabilities, label_id_to_name)
    save_predictions(predictions, paths["predictions"] / f"{run_slug}_predictions.csv", paths["predictions"] / f"{run_slug}_predictions.jsonl")
    save_error_analysis(
        predictions,
        paths["reports"] / f"{run_slug}_error_analysis.csv",
        paths["reports"] / f"{run_slug}_high_confidence_errors.csv",
        paths["reports"] / f"{run_slug}_low_confidence_hard_samples.csv",
    )

    if config.seed == 42 and run_slug == "bert_base_seed42":
        comparison = build_baseline_comparison(test_metrics)
        save_baseline_comparison(
            comparison,
            paths["reports"] / "baseline_comparison.json",
            paths["reports"] / "baseline_comparison.md",
        )

    logger.info("Phase 2 transformer baseline complete: %s", run_slug)
    return {
        "run_name": run_slug,
        "metrics": test_metrics,
        "predictions_path": str(paths["predictions"] / f"{run_slug}_predictions.csv"),
    }


def prepare_phase2_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create Phase 2 output directories."""
    root = Path(output_dir)
    paths = {
        "root": root,
        "metrics": root / "metrics",
        "plots": root / "plots",
        "predictions": root / "predictions",
        "reports": root / "reports",
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_phase1_dataset(dataset_path: str | Path) -> DatasetDict:
    """Load the cleaned Phase 1 PERIAD dataset."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned PERIAD dataset not found at {path}. Run Phase 1 first or pass --phase1_dataset_path."
        )
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected a DatasetDict at {path}, got {type(dataset)!r}.")
    return dataset


def validate_dataset_schema(dataset: DatasetDict) -> None:
    """Validate expected PERIAD split and column names."""
    for split in ("train", "test"):
        if split not in dataset:
            raise ValueError(f"Expected split '{split}' in Phase 1 dataset.")
        missing = [column for column in (TEXT_COLUMN, LABEL_COLUMN) if column not in dataset[split].column_names]
        if missing:
            raise ValueError(f"Split '{split}' is missing required columns: {missing}.")


def add_sample_ids(dataset: DatasetDict) -> DatasetDict:
    """Add stable sample IDs when no sample_id column exists."""
    splits = {}
    for split_name, split_dataset in dataset.items():
        if "sample_id" in split_dataset.column_names:
            splits[split_name] = split_dataset
        else:
            splits[split_name] = split_dataset.add_column(
                "sample_id",
                [f"{split_name}_{index}" for index in range(len(split_dataset))],
            )
    return DatasetDict(splits)


def make_validation_split(train_dataset: Dataset, validation_size: float, seed: int) -> tuple[Dataset, Dataset]:
    """Create a stratified validation split from the training split."""
    labels = np.asarray([int(label) for label in train_dataset[LABEL_COLUMN]])
    indices = np.arange(len(train_dataset))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        stratify=labels,
        random_state=seed,
    )
    return train_dataset.select(train_indices.tolist()), train_dataset.select(validation_indices.tolist())


def maybe_limit_split(dataset: Dataset, max_samples: int | None) -> Dataset:
    """Optionally limit a split for smoke testing."""
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return dataset.select(range(max_samples))


def tokenize_for_training(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    """Tokenize a split and add Trainer-compatible labels."""

    def _tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(
            [str(text) for text in batch[TEXT_COLUMN]],
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = [int(label) for label in batch[LABEL_COLUMN]]
        return tokenized

    return dataset.map(_tokenize, batched=True)


def build_training_arguments(config: TransformerTrainingConfig, checkpoint_dir: Path, run_slug: str) -> TrainingArguments:
    """Build TrainingArguments with Transformers version compatibility."""
    kwargs: dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "weight_decay": config.weight_decay,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": config.save_total_limit,
        "logging_dir": str(Path(config.output_dir) / "logs" / run_slug),
        "logging_steps": 50,
        "report_to": [],
        "seed": config.seed,
        "data_seed": config.seed,
        "fp16": should_use_fp16(config.fp16),
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def should_use_fp16(fp16: str) -> bool:
    """Resolve fp16 CLI policy."""
    normalized = str(fp16).lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return torch.cuda.is_available()


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute stable softmax probabilities."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def build_prediction_frame(
    raw_split: Dataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    label_id_to_name: dict[int, str],
) -> pd.DataFrame:
    """Build the saved prediction table."""
    rows: list[dict[str, Any]] = []
    label_ids = sorted(label_id_to_name)
    for index, (true_label, predicted_label) in enumerate(zip(y_true, y_pred)):
        probs = probabilities[index]
        sorted_probs = np.sort(probs)[::-1]
        confidence = float(sorted_probs[0])
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else confidence
        text = str(raw_split[index][TEXT_COLUMN])
        row: dict[str, Any] = {
            "sample_id": raw_split[index].get("sample_id", f"test_{index}"),
            "split": "test",
            "true_label": int(true_label),
            "true_label_name": label_id_to_name.get(int(true_label), f"label_{true_label}"),
            "predicted_label": int(predicted_label),
            "predicted_label_name": label_id_to_name.get(int(predicted_label), f"label_{predicted_label}"),
            "confidence": confidence,
            "margin": margin,
            "text_preview": text[:300],
            "text": text,
        }
        for label_id in label_ids:
            row[probability_column_name(label_id)] = float(probs[label_id])
        rows.append(row)
    return pd.DataFrame(rows)


def save_predictions(predictions: pd.DataFrame, csv_path: str | Path, jsonl_path: str | Path) -> None:
    """Save predictions as CSV and JSONL."""
    csv_output = Path(csv_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(csv_output, index=False)
    jsonl_output = Path(jsonl_path)
    jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_json(jsonl_output, orient="records", lines=True, force_ascii=False)
