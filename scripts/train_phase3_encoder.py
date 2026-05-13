"""Train Phase 3 advanced encoder experiments on the processed PERIAD split."""

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
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.train_encoder_baseline import (  # noqa: E402
    load_config,
    make_dataset_dict,
    read_label_map,
    read_split,
    softmax,
    tokenize_dataset,
)
from src.utils.paths import checkpoint_dir, dataset_dir, phase_artifact_dir  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase3" / "deberta_v3_base.yaml"
BASELINE_REFERENCE = "roberta_large_phase2"
PROBABILITY_COLUMNS = [f"prob_{index}" for index in range(6)]
PREDICTION_COLUMNS = (
    "sample_id",
    "text",
    "true_label",
    "true_author",
    "predicted_label",
    "predicted_author",
    "correct",
)
NAN_WATCH_KEYS = ("loss", "train_loss", "eval_loss", "grad_norm")
KNOWN_INVALID_RUNS = {
    "deberta_v3_base_seed42": "nan_loss_or_grad_norm_detected_in_completed_run",
    "deberta_v3_base_seed42_stable": "nan_or_inf_grad_norm_at_step_25",
}


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return calculate_metrics(labels, predictions)


def calculate_metrics(labels: Any, predictions: Any) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
    }


def is_bad_number(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isnan(numeric) or np.isinf(numeric))


def resolve_torch_dtype(name: Any) -> torch.dtype | None:
    normalized = str(name or "").lower()
    if normalized in ("", "none", "auto", "null"):
        return None
    if normalized in ("float32", "fp32", "torch.float32"):
        return torch.float32
    if normalized in ("float16", "fp16", "torch.float16"):
        return torch.float16
    if normalized in ("bfloat16", "bf16", "torch.bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported torch_dtype config value: {name!r}")


def summarize_model_dtypes(model: torch.nn.Module) -> dict[str, Any]:
    all_params: dict[str, int] = {}
    trainable_params: dict[str, int] = {}
    non_float32_trainable = []
    for name, param in model.named_parameters():
        dtype_name = str(param.dtype)
        all_params[dtype_name] = all_params.get(dtype_name, 0) + int(param.numel())
        if param.requires_grad:
            trainable_params[dtype_name] = trainable_params.get(dtype_name, 0) + int(param.numel())
            if param.dtype != torch.float32:
                non_float32_trainable.append(name)
    return {
        "all_params": all_params,
        "trainable_params": trainable_params,
        "all_trainable_params_float32": len(non_float32_trainable) == 0,
        "non_float32_trainable_params": non_float32_trainable[:20],
        "non_float32_trainable_param_count": len(non_float32_trainable),
    }


def find_invalid_reason(log_history: list[dict[str, Any]]) -> str | None:
    for entry in log_history:
        for key in NAN_WATCH_KEYS:
            if key in entry and is_bad_number(entry[key]):
                step = entry.get("step", "unknown")
                return f"nan_or_inf_{key}_at_step_{step}"
    return None


class NanStoppingCallback(TrainerCallback):
    """Stop training as soon as Trainer logs a NaN/inf stability signal."""

    def __init__(self) -> None:
        self.invalid_reason: str | None = None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if self.invalid_reason is not None or not logs:
            return control
        for key in NAN_WATCH_KEYS:
            if key in logs and is_bad_number(logs[key]):
                self.invalid_reason = f"nan_or_inf_{key}_at_step_{state.global_step}"
                control.should_training_stop = True
                control.should_epoch_stop = True
                break
        return control


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda" in text and ("out of memory" in text or "cuda oom" in text)


def is_fp16_unscale_error(exc: BaseException) -> bool:
    return "attempting to unscale fp16 gradients" in str(exc).lower()


def is_amp_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    amp_markers = (
        "attempting to unscale fp16 gradients",
        "grad scaler",
        "autocast",
        "amp",
        "fp16",
        "bf16",
    )
    return any(marker in text for marker in amp_markers)


def get_config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    return default


def make_smoke_subset(frame: pd.DataFrame, per_label: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby("label", sort=True):
        sample_size = min(int(per_label), len(group))
        parts.append(group.sample(n=sample_size, random_state=seed))
    subset = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return subset


def resolve_run_name(config: dict[str, Any], smoke_test: bool) -> str:
    experiment_name = str(config["experiment_name"])
    if smoke_test and not experiment_name.endswith("_smoke"):
        return f"{experiment_name}_smoke"
    return experiment_name


def resolve_training_argument_kwargs(
    config: dict[str, Any],
    output_dir: Path,
    precision: dict[str, Any],
    smoke_test: bool,
) -> dict[str, Any]:
    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters
    smoke_config = config.get("smoke_test", {}) if smoke_test else {}
    if not isinstance(smoke_config, dict):
        smoke_config = {}

    def cfg(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in smoke_config:
                return smoke_config[key]
        return get_config_value(config, *keys, default=default)

    eval_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    eval_value = cfg("eval_strategy", "evaluation_strategy", default="epoch")
    save_value = cfg("save_strategy", default=eval_value)
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": int(cfg("per_device_train_batch_size", "batch_size")),
        "per_device_eval_batch_size": int(cfg("per_device_eval_batch_size", "batch_size")),
        "learning_rate": float(cfg("learning_rate")),
        "num_train_epochs": float(cfg("num_train_epochs", "epochs")),
        "weight_decay": float(cfg("weight_decay")),
        "warmup_ratio": float(cfg("warmup_ratio", default=0.0)),
        "seed": int(config["seed"]),
        "gradient_accumulation_steps": int(cfg("gradient_accumulation_steps", default=1)),
        eval_key: eval_value,
        "save_strategy": save_value,
        "metric_for_best_model": cfg("metric_for_best_model", default="macro_f1"),
        "greater_is_better": True,
        "save_total_limit": int(cfg("save_total_limit", default=2)),
        "load_best_model_at_end": bool(cfg("load_best_model_at_end", default=True)),
        "logging_steps": int(cfg("logging_steps", default=50)),
        "dataloader_num_workers": int(cfg("dataloader_num_workers", default=0)),
        "max_grad_norm": float(cfg("max_grad_norm", default=1.0)),
        "optim": cfg("optim", default="adamw_torch"),
        "fp16": bool(precision["resolved_fp16"]),
        "bf16": bool(precision["resolved_bf16"]),
        "report_to": [],
    }
    amp_backend = precision.get("amp_backend")
    if amp_backend and amp_backend != "none":
        kwargs["half_precision_backend"] = amp_backend
    if save_value == "no" or eval_value == "no":
        kwargs["load_best_model_at_end"] = False
    return {key: value for key, value in kwargs.items() if key in params}


def resolve_precision_config(
    config: dict[str, Any],
    smoke_test: bool,
    cuda_available: bool,
    precision_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    smoke_config = config.get("smoke_test", {}) if smoke_test else {}
    if not isinstance(smoke_config, dict):
        smoke_config = {}

    def scoped_value(key: str, default: Any) -> Any:
        if precision_override and key in precision_override:
            return precision_override[key]
        if key in smoke_config:
            return smoke_config[key]
        return config.get(key, default)

    requested_fp16 = bool(scoped_value("fp16", False))
    requested_bf16 = bool(scoped_value("bf16", False))
    amp_backend = str(scoped_value("amp_backend", "none")).lower()
    if amp_backend in ("", "null", "none"):
        amp_backend = "none"

    if requested_fp16 and not cuda_available:
        print("WARNING: fp16 requested but CUDA is unavailable; using fp16=False.")
    if requested_bf16 and not cuda_available:
        print("WARNING: bf16 requested but CUDA is unavailable; using bf16=False.")
    if requested_fp16 and requested_bf16:
        raise ValueError("Only one of fp16 or bf16 may be enabled.")

    return {
        "requested_fp16": requested_fp16,
        "resolved_fp16": bool(requested_fp16 and cuda_available),
        "requested_bf16": requested_bf16,
        "resolved_bf16": bool(requested_bf16 and cuda_available),
        "amp_backend": amp_backend,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")


def write_runtime_note(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "runtime.json", payload)


def write_reports(
    eval_frame: pd.DataFrame,
    probabilities: np.ndarray,
    id_to_label: dict[int, str],
    run_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, float]:
    pred_labels = np.argmax(probabilities, axis=1)
    true_labels = eval_frame["label"].to_numpy()
    metrics = calculate_metrics(true_labels, pred_labels)

    predictions = pd.DataFrame(
        {
            "sample_id": eval_frame["sample_id"].to_numpy(),
            "text": eval_frame["text"].to_numpy(),
            "true_label": eval_frame["label"].astype(int).to_numpy(),
            "true_author": eval_frame["author"].to_numpy(),
            "predicted_label": pred_labels.astype(int),
            "predicted_author": [id_to_label[int(label)] for label in pred_labels],
        }
    )
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]
    for index, column in enumerate(PROBABILITY_COLUMNS):
        predictions[column] = probabilities[:, index]
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    print(f"Wrote {run_dir / 'predictions.csv'}")

    label_ids = sorted(id_to_label)
    label_names = [id_to_label[index] for index in label_ids]
    report = classification_report(
        true_labels,
        pred_labels,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"}).to_csv(
        run_dir / "classification_report.csv",
        index=False,
    )
    print(f"Wrote {run_dir / 'classification_report.csv'}")

    matrix = confusion_matrix(true_labels, pred_labels, labels=label_ids)
    matrix_frame = pd.DataFrame(matrix, index=label_names, columns=label_names)
    matrix_frame.index.name = "true_author"
    matrix_frame.to_csv(run_dir / "confusion_matrix.csv")
    print(f"Wrote {run_dir / 'confusion_matrix.csv'}")

    metrics_payload = {**metrics, **metadata}
    write_json(run_dir / "metrics.json", metrics_payload)
    return metrics_payload


def write_training_log(trainer: Trainer, run_dir: Path) -> None:
    log_history = trainer.state.log_history if trainer.state else []
    pd.DataFrame(log_history).to_csv(run_dir / "training_log.csv", index=False)
    print(f"Wrote {run_dir / 'training_log.csv'}")


def mark_known_invalid_runs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "valid_run" not in frame.columns:
        frame["valid_run"] = True
    if "invalid_reason" not in frame.columns:
        frame["invalid_reason"] = ""
    for run_name, reason in KNOWN_INVALID_RUNS.items():
        mask = frame["run_name"] == run_name if "run_name" in frame.columns else pd.Series(False, index=frame.index)
        frame.loc[mask, "valid_run"] = False
        frame.loc[mask, "invalid_reason"] = reason
    return frame


def update_advanced_results(row: dict[str, Any]) -> Path:
    table_dir = phase_artifact_dir("phase3", "tables")
    table_path = table_dir / "advanced_encoder_results.csv"
    if table_path.exists() and table_path.stat().st_size > 0:
        frame = pd.read_csv(table_path)
        frame = mark_known_invalid_runs(frame)
        frame = frame[frame["run_name"] != row["run_name"]] if "run_name" in frame.columns else frame
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame = mark_known_invalid_runs(frame)
    sort_columns = ["valid_run", "smoke_test", "macro_f1", "accuracy"]
    present_sort_columns = [column for column in sort_columns if column in frame.columns]
    if present_sort_columns:
        ascending = [False if column == "valid_run" else True if column == "smoke_test" else False for column in present_sort_columns]
        frame = frame.sort_values(present_sort_columns, ascending=ascending, na_position="last")
    frame.to_csv(table_path, index=False)
    print(f"Wrote {table_path}")
    return table_path


def build_dataset_dict(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    config: dict[str, Any],
    smoke_test: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, DatasetDict]:
    if smoke_test:
        smoke_config = config.get("smoke_test", {})
        if not isinstance(smoke_config, dict):
            smoke_config = {}
        train_per_label = int(smoke_config.get("train_per_label", 2))
        eval_per_label = int(smoke_config.get("eval_per_label", 1))
        seed = int(config["seed"])
        train_frame = make_smoke_subset(train_frame, train_per_label, seed)
        eval_frame = make_smoke_subset(eval_frame, eval_per_label, seed)
    return train_frame, eval_frame, make_dataset_dict(train_frame, eval_frame)


def run_experiment(
    config_path: str | Path,
    smoke_test: bool,
    resume_from_checkpoint: str | None = None,
    precision_override: dict[str, Any] | None = None,
) -> int:
    started_at = time.time()
    config = load_config(config_path)
    phase = str(config.get("phase", "phase3"))
    if phase != "phase3":
        raise ValueError(f"Expected phase3 config, found {phase!r}")

    run_name = resolve_run_name(config, smoke_test)
    model_name = str(config["model_name"])
    trust_remote_code = bool(config.get("trust_remote_code", False))
    seed = int(config["seed"])
    set_seed(seed)

    periad_dir = dataset_dir("processed", "periad")
    run_dir = phase_artifact_dir("phase3", "runs", run_name)
    phase_artifact_dir("phase3", "tables")
    phase_artifact_dir("phase3", "plots")
    output_dir = checkpoint_dir("phase3", run_name)
    label_map = read_label_map(periad_dir / "label_map.json")
    id_to_label = {label: author for author, label in label_map.items()}
    num_labels = len(label_map)
    if num_labels != 6:
        raise ValueError(f"Expected 6 PERIAD labels, found {num_labels}")

    train_frame = read_split(periad_dir / "train.csv", "train")
    eval_frame = read_split(periad_dir / "test.csv", "test")
    train_frame, eval_frame, dataset = build_dataset_dict(train_frame, eval_frame, config, smoke_test)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    tokenized = tokenize_dataset(dataset, tokenizer, int(config["max_length"]))
    sample = tokenized["train"][0]
    if "input_ids" not in sample or "labels" not in sample:
        raise ValueError("Tokenization did not produce required input_ids and labels fields.")

    cuda_available = torch.cuda.is_available()
    precision = resolve_precision_config(config, smoke_test, cuda_available, precision_override)

    training_kwargs = resolve_training_argument_kwargs(config, output_dir, precision, smoke_test)
    training_args = TrainingArguments(**training_kwargs)
    resolved_config = {
        "config_path": str(Path(config_path).resolve()),
        "run_name": run_name,
        "experiment_name": str(config["experiment_name"]),
        "model_name": model_name,
        "baseline_reference": str(config.get("baseline_reference", BASELINE_REFERENCE)),
        "local_practical": bool(config.get("local_practical", False)),
        "final_benchmark": bool(config.get("final_benchmark", False)),
        "trust_remote_code": trust_remote_code,
        "smoke_test": smoke_test,
        "dataset_dir": str(periad_dir),
        "run_artifact_dir": str(run_dir),
        "checkpoint_dir": str(output_dir),
        "n_train": len(train_frame),
        "n_test": len(eval_frame),
        "cuda_available": cuda_available,
        "requested_fp16": precision["requested_fp16"],
        "resolved_fp16": precision["resolved_fp16"],
        "requested_bf16": precision["requested_bf16"],
        "resolved_bf16": precision["resolved_bf16"],
        "amp_backend": precision["amp_backend"],
        "precision_override": precision_override,
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

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_map,
        use_safetensors=bool(config.get("use_safetensors", True)),
        torch_dtype=resolve_torch_dtype(config.get("torch_dtype")) if bool(config.get("force_float32", False)) else None,
        trust_remote_code=trust_remote_code,
    )
    if bool(config.get("force_float32", False)):
        model.to(torch.float32)
    dtype_summary = summarize_model_dtypes(model)
    if bool(config.get("force_float32", False)) and not dtype_summary["all_trainable_params_float32"]:
        runtime = {
            "status": "failed_dtype_verification",
            "runtime_seconds": round(time.time() - started_at, 4),
            "smoke_test": bool(smoke_test),
            "run_name": run_name,
            "valid_run": False,
            "invalid_reason": "trainable_parameters_not_float32",
            "requested_torch_dtype": str(config.get("torch_dtype")),
            "resolved_model_dtype_summary": dtype_summary,
            "all_trainable_params_float32": False,
        }
        write_json(run_dir / "runtime.json", runtime)
        raise ValueError(f"Trainable parameters are not all float32: {dtype_summary}")
    resolved_config["requested_torch_dtype"] = str(config.get("torch_dtype"))
    resolved_config["resolved_model_dtype_summary"] = dtype_summary
    resolved_config["all_trainable_params_float32"] = bool(dtype_summary["all_trainable_params_float32"])
    (run_dir / "run_config_resolved.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Wrote {run_dir / 'run_config_resolved.yaml'}")
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
    nan_callback = NanStoppingCallback()
    trainer.add_callback(nan_callback)

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    train_metrics = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in train_result.metrics.items()
    }
    write_json(run_dir / "train_metrics.json", train_metrics)
    log_history = trainer.state.log_history if trainer.state else []
    invalid_reason = nan_callback.invalid_reason or find_invalid_reason(log_history)
    if invalid_reason:
        trainer.state.save_to_json(str(run_dir / "trainer_state.json"))
        print(f"Wrote {run_dir / 'trainer_state.json'}")
        write_training_log(trainer, run_dir)
        metadata = {
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "precision_macro": None,
            "recall_macro": None,
            "n_train": int(len(train_frame)),
            "n_test": int(len(eval_frame)),
            "model_name": model_name,
            "seed": seed,
            "smoke_test": bool(smoke_test),
            "run_name": run_name,
            "baseline_reference": str(config.get("baseline_reference", BASELINE_REFERENCE)),
            "local_practical": bool(config.get("local_practical", False)),
            "final_benchmark": bool(config.get("final_benchmark", False)),
            "valid_run": False,
            "invalid_reason": invalid_reason,
            "requested_torch_dtype": str(config.get("torch_dtype")),
            "resolved_model_dtype_summary": dtype_summary,
            "all_trainable_params_float32": bool(dtype_summary["all_trainable_params_float32"]),
        }
        write_json(run_dir / "metrics.json", metadata)
        runtime = {
            "status": "invalid",
            "invalid_reason": invalid_reason,
            "runtime_seconds": round(time.time() - started_at, 4),
            "train_rows": len(train_frame),
            "eval_rows": len(eval_frame),
            "train_runtime": train_metrics.get("train_runtime"),
            "smoke_test": bool(smoke_test),
            "run_name": run_name,
            "valid_run": False,
            "requested_torch_dtype": str(config.get("torch_dtype")),
            "resolved_model_dtype_summary": dtype_summary,
            "all_trainable_params_float32": bool(dtype_summary["all_trainable_params_float32"]),
            "local_practical": bool(config.get("local_practical", False)),
            "final_benchmark": bool(config.get("final_benchmark", False)),
        }
        write_json(run_dir / "runtime.json", runtime)
        update_advanced_results(
            {
                "run_name": run_name,
                "experiment_name": str(config["experiment_name"]),
                "model_name": model_name,
                "baseline_reference": str(config.get("baseline_reference", BASELINE_REFERENCE)),
                "local_practical": bool(config.get("local_practical", False)),
                "final_benchmark": bool(config.get("final_benchmark", False)),
                "smoke_test": bool(smoke_test),
                "valid_run": False,
                "invalid_reason": invalid_reason,
                "requested_torch_dtype": str(config.get("torch_dtype")),
                "all_trainable_params_float32": bool(dtype_summary["all_trainable_params_float32"]),
                "accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "precision_macro": None,
                "recall_macro": None,
                "n_train": len(train_frame),
                "n_test": len(eval_frame),
                "seed": seed,
                "checkpoint_dir": str(output_dir),
                "metrics_path": str(run_dir / "metrics.json"),
                "predictions_path": "",
            }
        )
        return 0

    eval_metrics = trainer.evaluate()
    eval_metrics = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in eval_metrics.items()
    }
    write_json(run_dir / "eval_metrics.json", eval_metrics)

    prediction_output = trainer.predict(tokenized["test"])
    probabilities = softmax(prediction_output.predictions)
    metadata = {
        "n_train": int(len(train_frame)),
        "n_test": int(len(eval_frame)),
        "model_name": model_name,
        "seed": seed,
        "smoke_test": bool(smoke_test),
        "run_name": run_name,
        "baseline_reference": str(config.get("baseline_reference", BASELINE_REFERENCE)),
        "local_practical": bool(config.get("local_practical", False)),
        "final_benchmark": bool(config.get("final_benchmark", False)),
        "valid_run": True,
        "invalid_reason": "",
        "requested_torch_dtype": str(config.get("torch_dtype")),
        "resolved_model_dtype_summary": dtype_summary,
        "all_trainable_params_float32": bool(dtype_summary["all_trainable_params_float32"]),
    }
    metrics = write_reports(eval_frame, probabilities, id_to_label, run_dir, metadata)
    write_json(run_dir / "eval_metrics.json", {**eval_metrics, **metrics})

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
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
        "smoke_test": bool(smoke_test),
        "run_name": run_name,
        "valid_run": True,
        "invalid_reason": "",
        "requested_torch_dtype": str(config.get("torch_dtype")),
        "resolved_model_dtype_summary": dtype_summary,
        "all_trainable_params_float32": bool(dtype_summary["all_trainable_params_float32"]),
        "local_practical": bool(config.get("local_practical", False)),
        "final_benchmark": bool(config.get("final_benchmark", False)),
    }
    write_json(run_dir / "runtime.json", runtime)

    update_advanced_results(
        {
            "run_name": run_name,
            "experiment_name": str(config["experiment_name"]),
            "model_name": model_name,
            "baseline_reference": str(config.get("baseline_reference", BASELINE_REFERENCE)),
            "local_practical": bool(config.get("local_practical", False)),
            "final_benchmark": bool(config.get("final_benchmark", False)),
            "smoke_test": bool(smoke_test),
            "valid_run": True,
            "invalid_reason": "",
            "requested_torch_dtype": str(config.get("torch_dtype")),
            "all_trainable_params_float32": bool(dtype_summary["all_trainable_params_float32"]),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "n_train": len(train_frame),
            "n_test": len(eval_frame),
            "seed": seed,
            "checkpoint_dir": str(output_dir),
            "metrics_path": str(run_dir / "metrics.json"),
            "predictions_path": str(run_dir / "predictions.csv"),
        }
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    if not args.smoke_test and not torch.cuda.is_available():
        config = load_config(args.config)
        run_name = resolve_run_name(config, False)
        run_dir = phase_artifact_dir("phase3", "runs", run_name)
        write_runtime_note(
            run_dir,
            {
                "status": "full_run_skipped_no_cuda",
                "message": "CUDA is unavailable; automatically retrying smoke_test.",
                "smoke_test": False,
                "run_name": run_name,
            },
        )
        print("WARNING: CUDA is unavailable; running Phase 3 smoke test instead of full DeBERTa training.")
        return run_experiment(args.config, smoke_test=True, resume_from_checkpoint=None)

    try:
        return run_experiment(args.config, smoke_test=args.smoke_test, resume_from_checkpoint=args.resume_from_checkpoint)
    except (RuntimeError, ValueError) as exc:
        if args.smoke_test:
            raise
        if is_amp_error(exc):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            config = load_config(args.config)
            run_name = resolve_run_name(config, False)
            run_dir = phase_artifact_dir("phase3", "runs", run_name)
            write_runtime_note(
                run_dir,
                {
                    "status": "full_run_retrying_without_amp",
                    "message": str(exc),
                    "smoke_test": False,
                    "run_name": run_name,
                },
            )
            print("WARNING: Full DeBERTa training hit an AMP-related error; retrying full run with fp16=False, bf16=False, amp_backend=none.")
            try:
                return run_experiment(
                    args.config,
                    smoke_test=False,
                    resume_from_checkpoint=None,
                    precision_override={"fp16": False, "bf16": False, "amp_backend": "none"},
                )
            except RuntimeError as retry_exc:
                if not is_cuda_oom(retry_exc):
                    raise
                exc = retry_exc
            except ValueError:
                raise

        if not isinstance(exc, RuntimeError) or not is_cuda_oom(exc):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        config = load_config(args.config)
        run_name = resolve_run_name(config, False)
        run_dir = phase_artifact_dir("phase3", "runs", run_name)
        write_runtime_note(
            run_dir,
            {
                "status": "full_run_failed_cuda_oom",
                "message": str(exc),
                "smoke_test": False,
                "run_name": run_name,
            },
        )
        print("WARNING: Full DeBERTa training hit CUDA OOM; retrying with --smoke_test.")
        return run_experiment(args.config, smoke_test=True, resume_from_checkpoint=None)


if __name__ == "__main__":
    raise SystemExit(main())
