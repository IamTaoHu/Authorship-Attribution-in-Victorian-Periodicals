"""Train Phase 2 encoder baselines on the processed PERIAD split."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import checkpoint_dir, dataset_dir, phase_artifact_dir  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase2" / "bert_base.yaml"
REQUIRED_COLUMNS = ("sample_id", "text", "label", "author")
PROBABILITY_COLUMNS = [f"prob_{index}" for index in range(6)]


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def read_label_map(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Required PERIAD label map not found: {path}")
    label_map = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(label_map, dict) or sorted(label_map.values()) != list(range(len(label_map))):
        raise ValueError(f"Invalid label_map.json: {path}")
    return {str(author): int(label) for author, label in label_map.items()}


def read_split(path: Path, split_name: str, max_samples: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required PERIAD {split_name} split not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["label"] = frame["label"].astype(int)
    if max_samples is not None:
        frame = frame.head(int(max_samples)).copy()
    return frame


def make_dataset_dict(train_frame: pd.DataFrame, eval_frame: pd.DataFrame) -> DatasetDict:
    train_dataset = Dataset.from_pandas(train_frame, preserve_index=False)
    eval_dataset = Dataset.from_pandas(eval_frame, preserve_index=False)
    return DatasetDict({"train": train_dataset, "test": eval_dataset})


def tokenize_dataset(dataset: DatasetDict, tokenizer: Any, max_length: int) -> DatasetDict:
    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(batch["text"], truncation=True, max_length=max_length)
        tokenized["labels"] = batch["label"]
        return tokenized

    return dataset.map(tokenize, batched=True)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def resolve_training_argument_kwargs(config: dict[str, Any], output_dir: Path, fp16: bool) -> dict[str, Any]:
    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters
    eval_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    eval_value = config.get("eval_strategy", config.get("evaluation_strategy", "epoch"))
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": int(config["batch_size"]),
        "per_device_eval_batch_size": int(config["batch_size"]),
        "learning_rate": float(config["learning_rate"]),
        "num_train_epochs": float(config["epochs"]),
        "weight_decay": float(config["weight_decay"]),
        "seed": int(config["seed"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        eval_key: eval_value,
        "save_strategy": config.get("save_strategy", eval_value),
        "metric_for_best_model": config.get("metric_for_best_model", "macro_f1"),
        "greater_is_better": True,
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "load_best_model_at_end": bool(config.get("load_best_model_at_end", True)),
        "logging_steps": int(config.get("logging_steps", 50)),
        "fp16": fp16,
        "report_to": [],
    }
    return {key: value for key, value in kwargs.items() if key in params}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")


def write_reports(
    eval_frame: pd.DataFrame,
    probabilities: np.ndarray,
    id_to_label: dict[int, str],
    run_dir: Path,
) -> dict[str, float]:
    pred_labels = np.argmax(probabilities, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(eval_frame["label"], pred_labels)),
        "macro_f1": float(f1_score(eval_frame["label"], pred_labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(eval_frame["label"], pred_labels, average="weighted", zero_division=0)),
        "precision_macro": float(
            precision_score(eval_frame["label"], pred_labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(eval_frame["label"], pred_labels, average="macro", zero_division=0)),
    }
    predictions = eval_frame.copy()
    predictions["pred_label"] = pred_labels.astype(int)
    predictions["pred_author"] = predictions["pred_label"].map(id_to_label)
    predictions["correct"] = predictions["label"] == predictions["pred_label"]
    for index, column in enumerate(PROBABILITY_COLUMNS):
        predictions[column] = probabilities[:, index]
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    print(f"Wrote {run_dir / 'predictions.csv'}")

    label_ids = sorted(id_to_label)
    label_names = [id_to_label[index] for index in label_ids]
    report = classification_report(
        eval_frame["label"],
        pred_labels,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
    report_frame.to_csv(run_dir / "classification_report.csv", index=False)
    print(f"Wrote {run_dir / 'classification_report.csv'}")

    matrix = confusion_matrix(eval_frame["label"], pred_labels, labels=label_ids)
    matrix_frame = pd.DataFrame(matrix, index=label_names, columns=label_names)
    matrix_frame.index.name = "true_author"
    matrix_frame.to_csv(run_dir / "confusion_matrix.csv")
    print(f"Wrote {run_dir / 'confusion_matrix.csv'}")
    return metrics


def write_training_log(trainer: Trainer, run_dir: Path) -> None:
    log_history = trainer.state.log_history if trainer.state else []
    frame = pd.DataFrame(log_history)
    frame.to_csv(run_dir / "training_log.csv", index=False)
    print(f"Wrote {run_dir / 'training_log.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    started_at = time.time()
    config = load_config(args.config)
    experiment_name = str(config["experiment_name"])
    model_name = str(config["model_name"])
    max_train_samples = args.max_train_samples or config.get("max_train_samples")
    max_eval_samples = args.max_eval_samples or config.get("max_eval_samples")
    set_seed(int(config["seed"]))

    periad_dir = dataset_dir("processed", "periad")
    run_dir = phase_artifact_dir("phase2", "runs", experiment_name)
    output_dir = checkpoint_dir("phase2", experiment_name)
    label_map = read_label_map(periad_dir / "label_map.json")
    id_to_label = {label: author for author, label in label_map.items()}

    train_frame = read_split(periad_dir / "train.csv", "train", max_train_samples)
    eval_frame = read_split(periad_dir / "test.csv", "test", max_eval_samples)
    num_labels = len(label_map)
    if num_labels != 6:
        raise ValueError(f"Expected 6 PERIAD labels, found {num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    dataset = make_dataset_dict(train_frame, eval_frame)
    tokenized = tokenize_dataset(dataset, tokenizer, int(config["max_length"]))
    sample = tokenized["train"][0]
    if "input_ids" not in sample or "labels" not in sample:
        raise ValueError("Tokenization did not produce required input_ids and labels fields.")

    cuda_available = torch.cuda.is_available()
    requested_fp16 = bool(config.get("fp16", False))
    resolved_fp16 = bool(requested_fp16 and cuda_available)
    if requested_fp16 and not cuda_available:
        print("WARNING: fp16 requested but CUDA is unavailable; using fp16=False.")

    training_kwargs = resolve_training_argument_kwargs(config, output_dir, resolved_fp16)
    training_args = TrainingArguments(**training_kwargs)
    resolved_config = {
        "config_path": str(Path(args.config).resolve()),
        "experiment_name": experiment_name,
        "model_name": model_name,
        "dataset_dir": str(periad_dir),
        "run_artifact_dir": str(run_dir),
        "checkpoint_dir": str(output_dir),
        "max_train_samples": max_train_samples,
        "max_eval_samples": max_eval_samples,
        "cuda_available": cuda_available,
        "requested_fp16": requested_fp16,
        "resolved_training_arguments": training_args.to_dict(),
        "config": config,
    }
    (run_dir / "run_config_resolved.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Wrote {run_dir / 'run_config_resolved.yaml'}")
    shutil.copy2(periad_dir / "label_map.json", run_dir / "label_map.json")
    print(f"Wrote {run_dir / 'label_map.json'}")

    if args.dry_run:
        runtime = {
            "status": "dry_run_ok",
            "runtime_seconds": round(time.time() - started_at, 4),
            "train_rows": len(train_frame),
            "eval_rows": len(eval_frame),
            "model_loaded": False,
        }
        write_json(run_dir / "runtime.json", runtime)
        print("Dry run passed; training was not started.")
        return 0

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_map,
    )
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["test"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_metrics = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in train_result.metrics.items()
    }
    write_json(run_dir / "train_metrics.json", train_metrics)

    eval_metrics = trainer.evaluate()
    eval_metrics = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in eval_metrics.items()
    }
    write_json(run_dir / "eval_metrics.json", eval_metrics)

    prediction_output = trainer.predict(tokenized["test"])
    probabilities = softmax(prediction_output.predictions)
    report_metrics = write_reports(eval_frame, probabilities, id_to_label, run_dir)
    write_json(run_dir / "eval_metrics.json", {**eval_metrics, **report_metrics})

    trainer.state.save_to_json(str(run_dir / "trainer_state.json"))
    print(f"Wrote {run_dir / 'trainer_state.json'}")
    write_training_log(trainer, run_dir)
    runtime = {
        "status": "completed",
        "runtime_seconds": round(time.time() - started_at, 4),
        "train_rows": len(train_frame),
        "eval_rows": len(eval_frame),
        "train_runtime": train_metrics.get("train_runtime"),
        "eval_runtime": eval_metrics.get("eval_runtime"),
    }
    write_json(run_dir / "runtime.json", runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
