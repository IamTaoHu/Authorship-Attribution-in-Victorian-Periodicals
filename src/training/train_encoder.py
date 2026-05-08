"""Generic Phase 3 encoder training pipeline for PERIAD authorship attribution."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - optional dependency guarded at runtime
    LoraConfig = None
    TaskType = None
    get_peft_model = None

from src.data.load_datasets import build_label_mapping, infer_label_column, infer_text_column
from src.evaluation.label_mapping import get_author_id_to_name
from src.evaluation.metrics import (
    build_classification_report,
    build_confusion_matrices,
    compute_classification_metrics,
    trainer_compute_metrics,
)
from src.evaluation.reports import save_json
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed


DEFAULT_PHASE1_DATASET_PATH = Path("outputs/phase1/periad_cleaned")
DEFAULT_LORA_TARGETS = ["query_proj", "value_proj"]
DEFAULT_LORA_MODULES_TO_SAVE = ["classifier", "pooler"]
LORA_CANDIDATE_TERMS = ("query", "value", "key", "dense", "proj", "attention", "self")
TEXT_PREVIEW_CHARS = 300


@dataclass(frozen=True)
class EncoderRunConfig:
    """Resolved configuration for one Phase 3 encoder experiment."""

    experiment_name: str
    model_name: str
    enabled: bool = True
    notes: str | None = None
    dataset_name: str | None = "celvaigh/periad"
    text_column: str | None = "text"
    label_column: str | None = "author"
    output_dir: Path = Path("outputs/phase3")
    phase1_dataset_path: Path = DEFAULT_PHASE1_DATASET_PATH
    batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: float = 5.0
    max_length: int = 512
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    seed: int = 42
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    bf16: bool = False
    save_total_limit: int = 2
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 50
    validation_size: float = 0.1
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | str | None = None
    modules_to_save: list[str] | str | None = None
    save_embeddings: bool = True
    save_logits: bool = True
    save_predictions: bool = True
    dry_run_samples: int = 32
    use_safetensors: bool = True
    low_cpu_mem_usage: bool = True
    torch_dtype: str | None = None

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or dry-run a Phase 3 encoder experiment.")
    parser.add_argument("--config", required=True, help="Path to a Phase 3 YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup and exit before training.")
    parser.add_argument(
        "--list-linear-modules",
        action="store_true",
        help="Print model torch.nn.Linear module names and likely LoRA candidates, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    raw_config = load_config(str(config_path))
    config = resolve_config(raw_config)
    if args.list_linear_modules:
        list_linear_modules_for_config(config)
        return
    run_encoder(config, dry_run=args.dry_run)


def list_linear_modules_for_config(config: EncoderRunConfig) -> None:
    """Load only enough state to inspect Linear module names for LoRA targeting."""
    set_seed(config.seed)
    logger = setup_logger(f"{config.experiment_name}_linear_modules")
    logger.info("Loading labels for model initialization; full tokenization and training will not run.")
    dataset = load_encoder_dataset(config, logger)
    label_column = config.label_column or infer_label_column(dataset)
    if label_column is None:
        raise ValueError("Could not infer label_column. Set label_column in the config.")
    validate_columns_for_labels(dataset, label_column)
    dataset = prepare_labels(dataset, label_column)
    label_id_to_name = infer_id_to_label(dataset, label_column)
    label_ids = sorted(label_id_to_name)
    model = load_sequence_classifier(config, label_ids, label_id_to_name, logger)

    linear_modules = get_linear_module_names(model)
    candidate_modules = filter_lora_candidate_modules(linear_modules)
    print_module_listing("All torch.nn.Linear modules", linear_modules)
    print_module_listing("Likely LoRA candidate Linear modules", candidate_modules)


def run_encoder(config: EncoderRunConfig, dry_run: bool = False) -> dict[str, Any]:
    set_seed(config.seed)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(config.experiment_name, log_file=str(run_dir / "train.log"))
    logger.info("Starting Phase 3 encoder experiment: %s", config.experiment_name)
    logger.info("Model: %s", config.model_name)
    if not config.enabled:
        message = "This experiment is disabled in config. Exiting without training."
        save_resolved_config(config, run_dir / "run_config_resolved.yaml", dry_run=dry_run)
        logger.warning(message)
        if config.notes:
            logger.warning("Disabled note: %s", config.notes)
        print(message)
        return {"experiment_name": config.experiment_name, "run_status": "disabled", "notes": config.notes}
    if dry_run:
        logger.info("Dry-run mode enabled; training will not start.")

    save_resolved_config(config, run_dir / "run_config_resolved.yaml", dry_run=dry_run)
    dataset = load_encoder_dataset(config, logger)
    text_column = config.text_column or infer_text_column(dataset)
    label_column = config.label_column or infer_label_column(dataset)
    if text_column is None or label_column is None:
        raise ValueError("Could not infer text_column and label_column. Set both in the config.")
    validate_columns(dataset, text_column, label_column)

    dataset = prepare_labels(dataset, label_column)
    dataset = add_row_ids(dataset)
    label_id_to_name = infer_id_to_label(dataset, label_column)
    label_ids = sorted(label_id_to_name)
    label_names = [label_id_to_name[label_id] for label_id in label_ids]
    save_label_mapping(label_id_to_name, run_dir / "author_label_mapping.json")

    train_split, validation_split, test_split = prepare_splits(dataset, config, label_column)
    if dry_run:
        train_split = limit_split(train_split, config.dry_run_samples)
        validation_split = limit_split(validation_split, min(config.dry_run_samples, len(validation_split)))
        test_split = limit_split(test_split, min(config.dry_run_samples, len(test_split)))

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer_stats = compute_tokenizer_stats(
        tokenizer,
        DatasetDict({"train": train_split, "validation": validation_split, "test": test_split}),
        text_column,
        config.max_length,
    )
    save_json(tokenizer_stats, run_dir / "tokenizer_stats.json")
    logger.info("Tokenizer stats: %s", json.dumps(tokenizer_stats, indent=2))

    tokenized = DatasetDict(
        {
            "train": tokenize_split(train_split, tokenizer, text_column, label_column, config.max_length),
            "validation": tokenize_split(validation_split, tokenizer, text_column, label_column, config.max_length),
            "test": tokenize_split(test_split, tokenizer, text_column, label_column, config.max_length),
        }
    )

    model = load_sequence_classifier(config, label_ids, label_id_to_name, logger)
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if config.use_lora:
        model = apply_lora(model, config, logger)

    parameter_counts = count_parameters(model)
    logger.info(
        "Parameters: trainable=%s total=%s trainable_pct=%.4f",
        parameter_counts["trainable_params"],
        parameter_counts["total_params"],
        parameter_counts["trainable_param_pct"],
    )

    if dry_run:
        dry_metrics = {
            "experiment_name": config.experiment_name,
            "model_name": config.model_name,
            "dry_run": True,
            **parameter_counts,
            "tokenizer_stats": tokenizer_stats,
        }
        save_json(dry_metrics, run_dir / "dry_run_summary.json")
        print(json.dumps(dry_metrics, indent=2))
        logger.info("Dry run complete: %s", config.experiment_name)
        return dry_metrics

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    training_args = build_training_arguments(config, run_dir / "checkpoints")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        label_ids=label_ids,
        label_names=label_names,
    )

    train_start = perf_counter()
    train_result = trainer.train()
    train_elapsed = perf_counter() - train_start
    trainer.save_model(str(run_dir / "checkpoint"))
    tokenizer.save_pretrained(str(run_dir / "checkpoint"))
    trainer.save_state()

    validation_metrics = trainer.evaluate(tokenized["validation"], metric_key_prefix="validation")
    prediction_output = trainer.predict(tokenized["test"], metric_key_prefix="test")
    logits = np.asarray(prediction_output.predictions)
    probabilities = softmax(logits)
    y_true = np.asarray(prediction_output.label_ids, dtype=int)
    y_pred = probabilities.argmax(axis=1)

    metrics = compute_classification_metrics(y_true, y_pred, label_ids, label_names)
    metrics.update(
        {
            "experiment_name": config.experiment_name,
            "model_name": config.model_name,
            "max_length": config.max_length,
            "use_lora": config.use_lora,
            "lora_r": config.lora_r if config.use_lora else None,
            "seed": config.seed,
            "train_runtime": extract_train_runtime(train_result.metrics, train_elapsed),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "validation": numeric_dict(validation_metrics),
            **parameter_counts,
            **gpu_metrics(),
        }
    )
    if contains_nan_or_inf(metrics):
        metrics["run_status"] = "failed_nan"
        (run_dir / "FAILED_NAN.txt").write_text(
            "Run marked invalid because metrics contain NaN or Inf values. Do not use this run in Phase 3 tables.\n",
            encoding="utf-8",
        )
        logger.warning("Run marked invalid: metrics contain NaN/Inf values and should not be used in tables.")
    else:
        metrics["run_status"] = "ok"
    save_json(metrics, run_dir / "metrics.json")

    report = build_classification_report(y_true, y_pred, label_ids, label_names)
    save_json(report, run_dir / "classification_report.json")

    raw_cm, normalized_cm = build_confusion_matrices(y_true, y_pred, label_ids)
    save_confusion_matrix_csv(raw_cm, label_names, run_dir / "confusion_matrix.csv")
    save_confusion_matrix_csv(normalized_cm, label_names, run_dir / "confusion_matrix_normalized.csv")
    save_confusion_matrix_plot(raw_cm, label_names, run_dir / "confusion_matrix.png", config.experiment_name)

    if config.save_predictions:
        predictions = build_predictions(test_split, y_true, y_pred, probabilities, logits, label_id_to_name, text_column)
        predictions.to_csv(run_dir / "predictions.csv", index=False)
    if config.save_logits:
        np.save(run_dir / "logits.npy", logits)
    if config.save_embeddings:
        embeddings = extract_embeddings(model, tokenized["test"], tokenizer, config.eval_batch_size)
        np.save(run_dir / "embeddings.npy", embeddings)

    state_path = Path(training_args.output_dir) / "trainer_state.json"
    if state_path.exists():
        copy_text_file(state_path, run_dir / "trainer_state.json")

    logger.info("Phase 3 encoder experiment complete: %s", config.experiment_name)
    return metrics


def resolve_config(raw: dict[str, Any]) -> EncoderRunConfig:
    payload = dict(raw)
    if "experiment_name" not in payload:
        raise ValueError("Config must define experiment_name.")
    if "model_name" not in payload:
        raise ValueError("Config must define model_name.")
    for key in ("output_dir", "phase1_dataset_path"):
        if key in payload and payload[key] is not None:
            payload[key] = resolve_repo_path(payload[key])
    payload.setdefault("lora_target_modules", None)
    if payload["lora_target_modules"] in ("", []):
        payload["lora_target_modules"] = None
    payload.setdefault("modules_to_save", None)
    if payload["modules_to_save"] in ("", []):
        payload["modules_to_save"] = None
    if payload.get("use_lora") and payload["modules_to_save"] is None:
        payload["modules_to_save"] = list(DEFAULT_LORA_MODULES_TO_SAVE)
    if "torch_dtype" not in payload:
        payload["torch_dtype"] = "auto" if torch.cuda.is_available() else None
    return EncoderRunConfig(**payload)


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def save_resolved_config(config: EncoderRunConfig, output_path: Path, dry_run: bool) -> None:
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.__dict__.items()
    }
    payload["dry_run"] = dry_run
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def load_encoder_dataset(config: EncoderRunConfig, logger) -> DatasetDict:
    local_path = Path(config.phase1_dataset_path)
    if local_path.exists():
        logger.info("Loading local Phase 1 dataset: %s", local_path)
        dataset = load_from_disk(str(local_path))
    else:
        if not config.dataset_name:
            raise FileNotFoundError(f"Local dataset not found at {local_path} and dataset_name is not configured.")
        logger.info("Loading Hugging Face dataset: %s", config.dataset_name)
        dataset = load_dataset(config.dataset_name)
    if isinstance(dataset, DatasetDict):
        return dataset
    return DatasetDict({"train": dataset})


def validate_columns(dataset: DatasetDict, text_column: str, label_column: str) -> None:
    for split_name, split_dataset in dataset.items():
        missing = [column for column in (text_column, label_column) if column not in split_dataset.column_names]
        if missing:
            raise ValueError(f"Split '{split_name}' is missing required columns: {missing}.")


def validate_columns_for_labels(dataset: DatasetDict, label_column: str) -> None:
    for split_name, split_dataset in dataset.items():
        if label_column not in split_dataset.column_names:
            raise ValueError(f"Split '{split_name}' is missing required label column: {label_column}.")


def prepare_labels(dataset: DatasetDict, label_column: str) -> DatasetDict:
    mapping = build_label_mapping(dataset, label_column)
    if not mapping:
        raise ValueError(f"No labels found in column '{label_column}'.")

    def _encode(batch: dict[str, list[Any]]) -> dict[str, list[int]]:
        encoded = []
        for value in batch[label_column]:
            key = str(value).strip()
            if key in mapping:
                encoded.append(int(mapping[key]))
            else:
                encoded.append(int(value))
        return {label_column: encoded}

    try:
        return dataset.map(_encode, batched=True)
    except (ValueError, TypeError):
        return dataset.map(lambda batch: {label_column: [int(value) for value in batch[label_column]]}, batched=True)


def add_row_ids(dataset: DatasetDict) -> DatasetDict:
    splits = {}
    for split_name, split_dataset in dataset.items():
        if "row_id" in split_dataset.column_names:
            splits[split_name] = split_dataset
        elif "sample_id" in split_dataset.column_names:
            splits[split_name] = split_dataset.rename_column("sample_id", "row_id")
        else:
            splits[split_name] = split_dataset.add_column(
                "row_id",
                [f"{split_name}_{index}" for index in range(len(split_dataset))],
            )
    return DatasetDict(splits)


def infer_id_to_label(dataset: DatasetDict, label_column: str) -> dict[int, str]:
    observed = sorted({int(label) for split in dataset.values() for label in split[label_column]})
    canonical = get_author_id_to_name()
    if observed == sorted(canonical):
        return canonical
    return {label_id: f"label_{label_id}" for label_id in observed}


def save_label_mapping(label_id_to_name: dict[int, str], output_path: Path) -> None:
    payload = {
        "label_to_author_name": {str(label_id): name for label_id, name in sorted(label_id_to_name.items())},
        "author_name_to_label": {name: int(label_id) for label_id, name in sorted(label_id_to_name.items())},
        "note": "Numeric labels are preserved for training; author names are used for reports and plots.",
    }
    save_json(payload, output_path)


def prepare_splits(
    dataset: DatasetDict,
    config: EncoderRunConfig,
    label_column: str,
) -> tuple[Dataset, Dataset, Dataset]:
    if "train" not in dataset:
        raise ValueError("Dataset must contain a train split.")
    test_split = dataset["test"] if "test" in dataset else dataset["train"]
    if "validation" in dataset:
        return dataset["train"], dataset["validation"], test_split

    labels = np.asarray([int(label) for label in dataset["train"][label_column]])
    indices = np.arange(len(dataset["train"]))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=config.validation_size,
        stratify=labels,
        random_state=config.seed,
    )
    return dataset["train"].select(train_indices.tolist()), dataset["train"].select(validation_indices.tolist()), test_split


def limit_split(dataset: Dataset, max_samples: int) -> Dataset:
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return dataset.select(range(max_samples))


def compute_tokenizer_stats(tokenizer, dataset: DatasetDict, text_column: str, max_length: int) -> dict[str, Any]:
    lengths: list[int] = []
    split_stats: dict[str, Any] = {}
    for split_name, split_dataset in dataset.items():
        split_lengths: list[int] = []
        for text in split_dataset[text_column]:
            encoded = tokenizer(str(text), truncation=False, add_special_tokens=True)
            split_lengths.append(len(encoded["input_ids"]))
        truncation_count = sum(1 for length in split_lengths if length > max_length)
        split_stats[split_name] = {
            "num_rows": len(split_lengths),
            "avg_token_length": float(np.mean(split_lengths)) if split_lengths else 0.0,
            "max_observed_token_length": int(max(split_lengths)) if split_lengths else 0,
            "truncation_count": int(truncation_count),
            "truncation_percentage": float(truncation_count / len(split_lengths) * 100.0) if split_lengths else 0.0,
        }
        lengths.extend(split_lengths)
    truncation_count = sum(1 for length in lengths if length > max_length)
    return {
        "max_length": int(max_length),
        "num_rows": len(lengths),
        "avg_token_length": float(np.mean(lengths)) if lengths else 0.0,
        "max_observed_token_length": int(max(lengths)) if lengths else 0,
        "truncation_count": int(truncation_count),
        "truncation_percentage": float(truncation_count / len(lengths) * 100.0) if lengths else 0.0,
        "splits": split_stats,
    }


def tokenize_split(dataset: Dataset, tokenizer, text_column: str, label_column: str, max_length: int) -> Dataset:
    def _tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer([str(text) for text in batch[text_column]], truncation=True, max_length=max_length)
        tokenized["labels"] = [int(label) for label in batch[label_column]]
        return tokenized

    return dataset.map(_tokenize, batched=True)


def apply_lora(model, config: EncoderRunConfig, logger):
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError("PEFT is required for use_lora=true. Install peft or disable LoRA.")
    target_modules = resolve_lora_target_modules(model, config.lora_target_modules, logger)
    modules_to_save = resolve_modules_to_save(config.modules_to_save, logger)
    logger.info(
        "Applying LoRA with r=%s, alpha=%s, targets=%s, modules_to_save=%s",
        config.lora_r,
        config.lora_alpha,
        target_modules,
        modules_to_save,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    log_trainable_parameter_names(peft_model, logger)
    return peft_model


def resolve_modules_to_save(configured_modules: list[str] | str | None, logger) -> list[str]:
    """Keep classification head modules trainable when using PEFT LoRA."""
    if configured_modules is None:
        default_modules = list(DEFAULT_LORA_MODULES_TO_SAVE)
        logger.info("No modules_to_save configured for LoRA; using defaults: %s", default_modules)
        return default_modules
    if isinstance(configured_modules, str):
        module = configured_modules.strip()
        if module:
            return [module]
        default_modules = list(DEFAULT_LORA_MODULES_TO_SAVE)
        logger.info("Empty modules_to_save configured for LoRA; using defaults: %s", default_modules)
        return default_modules
    modules = [str(module) for module in configured_modules if str(module).strip()]
    if not modules:
        default_modules = list(DEFAULT_LORA_MODULES_TO_SAVE)
        logger.info("Empty modules_to_save configured for LoRA; using defaults: %s", default_modules)
        return default_modules
    return modules


def log_trainable_parameter_names(model, logger) -> None:
    """Log trainable parameter groups after PEFT wraps the model."""
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    group_counts: dict[str, int] = {}
    for name in trainable_names:
        group = infer_trainable_parameter_group(name)
        group_counts[group] = group_counts.get(group, 0) + 1
    logger.info("Trainable parameter groups after LoRA: %s", group_counts)
    logger.info("Trainable parameter names after LoRA: %s", trainable_names)
    if not any("classifier" in name.lower() for name in trainable_names):
        logger.warning(
            "No trainable parameter name contains 'classifier'. "
            "For sequence classification with LoRA, keep the classifier head trainable via modules_to_save."
        )


def infer_trainable_parameter_group(parameter_name: str) -> str:
    lowered = parameter_name.lower()
    if "classifier" in lowered:
        return "classifier"
    if "pooler" in lowered:
        return "pooler"
    if "lora" in lowered:
        return "lora"
    return parameter_name.split(".", 1)[0]


def resolve_lora_target_modules(model, configured_targets: list[str] | str | None, logger) -> list[str]:
    """Resolve LoRA target module suffixes, optionally by inspecting the loaded model."""
    if configured_targets is None:
        logger.info("No lora_target_modules configured; using defaults: %s", DEFAULT_LORA_TARGETS)
        return DEFAULT_LORA_TARGETS
    if isinstance(configured_targets, str):
        normalized = configured_targets.strip()
        if normalized.lower() == "auto":
            targets = auto_lora_target_modules(model)
            logger.info("Auto-selected LoRA target module suffixes: %s", targets)
            return targets
        if normalized:
            return [normalized]
        logger.info("Empty lora_target_modules configured; using defaults: %s", DEFAULT_LORA_TARGETS)
        return DEFAULT_LORA_TARGETS
    if not configured_targets:
        logger.info("Empty lora_target_modules configured; using defaults: %s", DEFAULT_LORA_TARGETS)
        return DEFAULT_LORA_TARGETS
    return [str(target) for target in configured_targets]


def auto_lora_target_modules(model) -> list[str]:
    """Choose PEFT LoRA target suffixes from actual Linear module names."""
    linear_modules = get_linear_module_names(model)
    module_names = set(linear_modules)
    suffixes = {module_name.rsplit(".", 1)[-1] for module_name in linear_modules}

    if {"query_proj", "value_proj"}.issubset(suffixes):
        return ["query_proj", "value_proj"]
    if any("attention.self.query_proj" in name for name in module_names) and any(
        "attention.self.value_proj" in name for name in module_names
    ):
        return ["query_proj", "value_proj"]

    fused_projection_targets = [target for target in ("in_proj", "out_proj") if target in suffixes]
    if fused_projection_targets:
        return fused_projection_targets

    if any("attention.self.query" in name for name in module_names) and any(
        "attention.self.value" in name for name in module_names
    ):
        return ["query", "value"]
    if {"query", "value"}.issubset(suffixes):
        return ["query", "value"]

    candidate_modules = filter_lora_candidate_modules(linear_modules)
    print_module_listing("Likely LoRA candidate Linear modules", candidate_modules)
    raise ValueError(
        "Could not auto-select LoRA target modules from the loaded model. "
        "Run `python src/training/train_encoder.py --config <config> --list-linear-modules` "
        "and set lora_target_modules to PEFT suffix names from the printed Linear modules."
    )


def get_linear_module_names(model) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]


def filter_lora_candidate_modules(module_names: list[str]) -> list[str]:
    return [
        name
        for name in module_names
        if any(term in name.lower() for term in LORA_CANDIDATE_TERMS)
    ]


def print_module_listing(title: str, module_names: list[str]) -> None:
    print(f"\n{title} ({len(module_names)}):")
    if not module_names:
        print("  <none>")
        return
    for module_name in module_names:
        print(f"  {module_name}")


def load_sequence_classifier(
    config: EncoderRunConfig,
    label_ids: list[int],
    label_id_to_name: dict[int, str],
    logger,
):
    """Load a sequence classifier with safetensors-first defaults and clear torch.load errors."""
    model_load_kwargs: dict[str, Any] = {
        "num_labels": len(label_ids),
        "id2label": {int(label_id): label_id_to_name[int(label_id)] for label_id in label_ids},
        "label2id": {label_id_to_name[int(label_id)]: int(label_id) for label_id in label_ids},
        "use_safetensors": config.use_safetensors,
        "low_cpu_mem_usage": config.low_cpu_mem_usage,
    }
    dtype = resolve_torch_dtype(config.torch_dtype)
    if dtype is not None:
        model_load_kwargs["torch_dtype"] = dtype

    logger.info("Model load options: use_safetensors=%s low_cpu_mem_usage=%s torch_dtype=%s", config.use_safetensors, config.low_cpu_mem_usage, dtype)
    try:
        return AutoModelForSequenceClassification.from_pretrained(config.model_name, **model_load_kwargs)
    except ValueError as exc:
        message = str(exc)
        lowered = message.lower()
        if config.use_safetensors and is_missing_safetensors_error(lowered):
            logger.error(
                "No safetensors weights found for this model. Falling back to PyTorch .bin is blocked on torch<2.6."
            )
        if "torch.load" in lowered or "torch to at least v2.6" in lowered:
            logger.error(
                "Model loading failed because the checkpoint is trying to load a PyTorch .bin file. "
                "The active torch version is below 2.6, and Transformers blocks torch.load for this path "
                "because of a CVE safety restriction. Try use_safetensors: true. If this model repository "
                "has no safetensors weights, upgrade torch>=2.6 or switch to a safetensors-compatible model."
            )
        raise


def resolve_torch_dtype(value: str | None):
    """Resolve YAML torch_dtype values for Transformers model loading."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    if normalized == "auto":
        return "auto"
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float32":
        return torch.float32
    raise ValueError("torch_dtype must be one of: auto, float16, bfloat16, float32, null.")


def is_missing_safetensors_error(lowered_message: str) -> bool:
    """Return True when Transformers reports that safetensors weights are unavailable."""
    safetensor_terms = ("safetensor", "safetensors")
    missing_terms = ("not found", "no file", "does not appear", "cannot be loaded", "couldn't find", "could not find")
    return any(term in lowered_message for term in safetensor_terms) and any(
        term in lowered_message for term in missing_terms
    )


def count_parameters(model) -> dict[str, Any]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    pct = (trainable / total * 100.0) if total else 0.0
    return {"trainable_params": int(trainable), "total_params": int(total), "trainable_param_pct": float(pct)}


def build_training_arguments(config: EncoderRunConfig, checkpoint_dir: Path) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "warmup_ratio": config.warmup_ratio,
        "save_strategy": config.save_strategy,
        "load_best_model_at_end": config.save_strategy == config.eval_strategy,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": config.save_total_limit,
        "logging_dir": str(config.run_dir / "logs"),
        "logging_steps": config.logging_steps,
        "report_to": [],
        "seed": config.seed,
        "data_seed": config.seed,
        "fp16": bool(config.fp16),
        "bf16": bool(config.bf16),
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = config.eval_strategy
    else:
        kwargs["evaluation_strategy"] = config.eval_strategy
    return TrainingArguments(**kwargs)


def build_trainer(
    model,
    tokenizer,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    data_collator,
    label_ids: list[int],
    label_names: list[str],
) -> Trainer:
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": trainer_compute_metrics(label_ids, label_names),
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def build_predictions(
    raw_split: Dataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    logits: np.ndarray,
    label_id_to_name: dict[int, str],
    text_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    label_ids = sorted(label_id_to_name)
    for index, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        probs = probabilities[index]
        row: dict[str, Any] = {
            "row_id": raw_split[index].get("row_id", f"test_{index}"),
            "text": str(raw_split[index][text_column]),
            "text_preview": str(raw_split[index][text_column])[:TEXT_PREVIEW_CHARS],
            "true_label": int(true_label),
            "true_label_name": label_id_to_name.get(int(true_label), f"label_{true_label}"),
            "pred_label": int(pred_label),
            "pred_label_name": label_id_to_name.get(int(pred_label), f"label_{pred_label}"),
            "predicted_label": int(pred_label),
            "predicted_label_name": label_id_to_name.get(int(pred_label), f"label_{pred_label}"),
            "correct": bool(int(true_label) == int(pred_label)),
            "probability_pred": float(probs[int(pred_label)]),
            "confidence": float(probs[int(pred_label)]),
        }
        for label_id in label_ids:
            label_name = label_id_to_name[label_id].lower().replace(" ", "_").replace("-", "_")
            row[f"prob_{label_id}_{label_name}"] = float(probs[label_id])
            row[f"logit_{label_id}"] = float(logits[index][label_id])
        rows.append(row)
    return pd.DataFrame(rows)


def save_confusion_matrix_csv(matrix: np.ndarray, label_names: list[str], output_path: Path) -> None:
    frame = pd.DataFrame(matrix, index=label_names, columns=label_names)
    frame.index.name = "true_label_name"
    frame.columns.name = "pred_label_name"
    frame.to_csv(output_path)


def save_confusion_matrix_plot(matrix: np.ndarray, label_names: list[str], output_path: Path, experiment_name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{experiment_name} confusion matrix")
    ax.set_xlabel("Predicted author")
    ax.set_ylabel("True author")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=35, ha="right")
    ax.set_yticklabels(label_names)
    threshold = float(matrix.max()) / 2.0 if matrix.size and matrix.max() > 0 else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                str(int(value)),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def extract_embeddings(model, tokenized_test: Dataset, tokenizer, batch_size: int) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    columns = [column for column in ("input_ids", "attention_mask", "token_type_ids") if column in tokenized_test.column_names]
    torch_dataset = tokenized_test.remove_columns(
        [column for column in tokenized_test.column_names if column not in columns]
    ).with_format("torch")
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, collate_fn=collator)
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            embeddings.append(last_hidden[:, 0, :].detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0))


def extract_train_runtime(train_metrics: dict[str, Any], fallback: float) -> float:
    value = train_metrics.get("train_runtime", fallback)
    return float(value) if value is not None else float(fallback)


def gpu_metrics() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "gpu_max_allocated_gb": None,
            "gpu_max_reserved_gb": None,
        }
    gb = 1024.0**3
    return {
        "gpu_max_allocated_gb": float(torch.cuda.max_memory_allocated() / gb),
        "gpu_max_reserved_gb": float(torch.cuda.max_memory_reserved() / gb),
    }


def numeric_dict(values: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in values.items() if isinstance(value, (int, float))}


def contains_nan_or_inf(value: Any) -> bool:
    """Return True if a nested metrics object contains NaN or Inf."""
    if isinstance(value, dict):
        return any(contains_nan_or_inf(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_nan_or_inf(item) for item in value)
    if isinstance(value, (int, float, np.floating)):
        return not math.isfinite(float(value))
    return False


def copy_text_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
