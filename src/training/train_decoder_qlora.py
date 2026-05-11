"""Train Phase 5 decoder QLoRA adapters for PERIAD authorship attribution."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any
import warnings

from datasets import Dataset
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.author_parser import parse_author_prediction
from src.utils.config import load_config
from src.utils.reproducibility import set_seed

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except ImportError:  # pragma: no cover - optional dependency guarded at runtime
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
OFFICIAL_MODEL_MARKERS = (
    "mistralai/mistral-7b-instruct",
    "meta-llama/meta-llama-3-8b-instruct",
    "google/gemma-2-9b-it",
)
REQUIRED_CONFIG_FIELDS = {
    "experiment_name",
    "model_name",
    "model_short_name",
    "output_dir",
    "max_length",
    "batch_size",
    "eval_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "epochs",
    "warmup_ratio",
    "weight_decay",
    "lr_scheduler_type",
    "seed",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "bnb_4bit",
    "bnb_4bit_quant_type",
    "bnb_4bit_compute_dtype",
    "bnb_4bit_use_double_quant",
    "gradient_checkpointing",
    "fp16",
    "bf16",
    "logging_steps",
    "eval_steps",
    "save_steps",
    "save_total_limit",
    "report_to",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Phase 5 decoder QLoRA adapter.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train_file", required=True, type=Path)
    parser.add_argument("--eval_file", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_phase5_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = str(resolve_repo_path(args.output_dir))
    set_seed(int(config["seed"]))
    train_rows = limit_rows(read_jsonl(resolve_repo_path(args.train_file)), args.max_train_samples or config.get("num_train_samples"))
    eval_rows = limit_rows(read_jsonl(resolve_repo_path(args.eval_file)), args.max_eval_samples or config.get("num_eval_samples"))
    if not train_rows or not eval_rows:
        raise ValueError("Training and evaluation files must contain at least one row each.")

    run_dir = resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, run_dir / "run_config_resolved.yaml", dry_run=args.dry_run)

    if args.dry_run:
        dry_run(config, train_rows, eval_rows, run_dir)
        return

    assert_safe_to_train(config, train_rows, args.max_train_samples)
    train(config, train_rows, eval_rows, run_dir, args.resume_from_checkpoint)


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def load_phase5_config(path: Path) -> dict[str, Any]:
    config = load_config(str(resolve_repo_path(path)))
    missing = sorted(REQUIRED_CONFIG_FIELDS - set(config))
    if missing:
        raise ValueError(f"Phase 5 QLoRA config is missing required fields: {missing}")
    config = dict(config)
    if not config.get("lora_target_modules"):
        config["lora_target_modules"] = list(DEFAULT_LORA_TARGETS)
    config["output_dir"] = str(resolve_repo_path(config["output_dir"]))
    return config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Instruction JSONL not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def limit_rows(rows: list[dict[str, Any]], limit: Any) -> list[dict[str, Any]]:
    if limit in (None, "", "null"):
        return rows
    max_rows = int(limit)
    return rows[:max_rows] if max_rows > 0 else rows


def resolve_run_dir(config: dict[str, Any]) -> Path:
    return Path(config["output_dir"]) / str(config["experiment_name"])


def save_resolved_config(config: dict[str, Any], output_path: Path, dry_run: bool) -> None:
    payload = dict(config)
    payload["dry_run"] = bool(dry_run)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def dry_run(config: dict[str, Any], train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], run_dir: Path) -> None:
    tokenizer = load_tokenizer_for_dry_run(config)
    tokenized_preview = [tokenize_example(row, tokenizer, int(config["max_length"])) for row in train_rows[:3]]
    masking_checks = [masking_summary(item) for item in tokenized_preview]
    parser_checks = run_parser_checks()
    summary = {
        "experiment_name": config["experiment_name"],
        "model_name": config["model_name"],
        "dry_run": True,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "max_length": int(config["max_length"]),
        "lora_r": int(config["lora_r"]),
        "masking_checks": masking_checks,
        "parser_checks": parser_checks,
        "official_training_guard": "dry_run did not load or train the official model",
    }
    if not all(check["response_label_tokens"] > 0 and check["prompt_masked_tokens"] > 0 for check in masking_checks):
        raise ValueError("Dry-run masking check failed: prompt masking or response labels are missing.")
    if not all(check["passed"] for check in parser_checks.values()):
        raise ValueError("Dry-run parser checks failed.")
    write_json(summary, run_dir / "dry_run_summary.json")
    print(json.dumps(summary, indent=2))


class WhitespaceTokenizer:
    """Small tokenizer fallback for dry-run masking checks when HF tokenizers are unavailable."""

    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 1

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, list[int]]:
        tokens = str(text).split()
        ids = [abs(hash(token)) % 30000 + 2 for token in tokens]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def load_tokenizer_for_dry_run(config: dict[str, Any]) -> Any:
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True, local_files_only=True)
        ensure_pad_token(tokenizer)
        return tokenizer
    except Exception as exc:
        warnings.warn(
            f"Could not load cached tokenizer for dry-run ({exc}); using whitespace fallback for masking validation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return WhitespaceTokenizer()


def run_parser_checks() -> dict[str, dict[str, Any]]:
    cases = {
        "valid_full": ("Answer: John Morley", 1),
        "partial_unambiguous": ("Morley.", 1),
        "ambiguous": ("Leslie Stephen or James Fitzjames Stephen", None),
        "hallucinated": ("Final answer: Charles Dickens", None),
    }
    checks: dict[str, dict[str, Any]] = {}
    for name, (text, expected_label) in cases.items():
        parsed = parse_author_prediction(text)
        actual = parsed["parsed_label"]
        passed = actual == expected_label
        if name in {"ambiguous", "hallucinated"}:
            passed = actual is None and str(parsed["parse_status"]).startswith("invalid")
        checks[name] = {
            "input": text,
            "expected_label": expected_label,
            "parsed_label": actual,
            "parse_status": parsed["parse_status"],
            "passed": bool(passed),
        }
    return checks


def assert_safe_to_train(config: dict[str, Any], train_rows: list[dict[str, Any]], max_train_samples: int | None) -> None:
    model_name = str(config["model_name"]).lower()
    is_official = any(marker in model_name for marker in OFFICIAL_MODEL_MARKERS)
    sample_limited = max_train_samples is not None or config.get("num_train_samples") not in (None, "", "null")
    running_on_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"))
    explicit_local = os.environ.get("PHASE5_ALLOW_LOCAL_FULL_TRAINING") == "1"
    if is_official and not running_on_kaggle and not explicit_local and not sample_limited:
        raise RuntimeError(
            "Refusing to start full official 7B/8B/9B QLoRA training outside Kaggle. "
            "Use --dry_run locally, set a small --max_train_samples for smoke tests, or run on Kaggle. "
            "Set PHASE5_ALLOW_LOCAL_FULL_TRAINING=1 only if you intentionally accept the local GPU risk."
        )
    if is_official and len(train_rows) > 256 and not running_on_kaggle and not explicit_local:
        raise RuntimeError(
            "Refusing local official-model training above 256 samples. Use Kaggle GPU for full Phase 5 runs."
        )


def train(
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    run_dir: Path,
    resume_from_checkpoint: Path | None,
) -> None:
    require_training_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    ensure_pad_token(tokenizer)
    model = load_qlora_model(config)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(config["gradient_checkpointing"]))
    model = apply_lora(model, config)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = Dataset.from_list([tokenize_example(row, tokenizer, int(config["max_length"])) for row in train_rows])
    eval_dataset = Dataset.from_list([tokenize_example(row, tokenizer, int(config["max_length"])) for row in eval_rows])
    parameter_counts = count_parameters(model)
    write_json(parameter_counts, run_dir / "trainable_params.json")

    training_args = build_training_arguments(config, run_dir / "trainer_checkpoints")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    log_callback = CsvLogCallback(run_dir / "training_log.csv")
    trainer = build_trainer(model, tokenizer, training_args, train_dataset, eval_dataset, data_collator, log_callback)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    try:
        train_result = trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware-specific
        raise RuntimeError(
            "CUDA out of memory during Phase 5 QLoRA. Try reducing max_length/batch_size, increasing "
            "gradient_accumulation_steps, using a smaller LoRA rank, or moving the run to Kaggle with more VRAM."
        ) from exc
    elapsed = perf_counter() - start

    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    adapter_dir = run_dir / "adapter"
    tokenizer_dir = run_dir / "tokenizer"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(tokenizer_dir))

    train_metrics = dict(train_result.metrics)
    train_metrics["train_elapsed_sec"] = float(elapsed)
    train_metrics.update(gpu_metrics())
    write_json(train_metrics, run_dir / "train_metrics.json")
    write_json(numeric_dict(eval_metrics), run_dir / "eval_loss_metrics.json")
    write_json(build_adapter_manifest(config, run_dir, adapter_dir, parameter_counts), run_dir / "adapter_manifest.json")
    print(f"Wrote Phase 5 QLoRA adapter to {adapter_dir}")


def require_training_dependencies() -> None:
    if any(item is None for item in (LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training)):
        raise ImportError("PEFT is required for Phase 5 QLoRA training. Install peft before training.")
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-specific
        raise ImportError("bitsandbytes is required for 4-bit QLoRA training.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bitsandbytes 4-bit QLoRA training. Use Kaggle GPU for official runs.")


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def load_qlora_model(config: dict[str, Any]) -> Any:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bool(config["bnb_4bit"]),
        bnb_4bit_quant_type=str(config["bnb_4bit_quant_type"]),
        bnb_4bit_compute_dtype=resolve_torch_dtype(config["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=bool(config["bnb_4bit_use_double_quant"]),
    )
    return AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=False,
    )


def resolve_torch_dtype(value: Any) -> torch.dtype:
    normalized = str(value).lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError("bnb_4bit_compute_dtype must be one of: bfloat16, float16, float32.")


def apply_lora(model: Any, config: dict[str, Any]) -> Any:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=[str(item) for item in config.get("lora_target_modules") or DEFAULT_LORA_TARGETS],
        bias="none",
    )
    return get_peft_model(model, lora_config)


def tokenize_example(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    prompt = str(row["prompt"])
    response = str(row["response"])
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False, truncation=False)["input_ids"]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        response_ids = response_ids + [int(eos_id)]
    if len(response_ids) >= max_length:
        raise ValueError("Response token length exceeds max_length; refusing to truncate away author label.")
    prompt_budget = max_length - len(response_ids)
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + list(response_ids)
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def masking_summary(item: dict[str, Any]) -> dict[str, int]:
    labels = item["labels"]
    return {
        "input_tokens": len(item["input_ids"]),
        "prompt_masked_tokens": int(sum(1 for label in labels if int(label) == -100)),
        "response_label_tokens": int(sum(1 for label in labels if int(label) != -100)),
    }


def build_training_arguments(config: dict[str, Any], checkpoint_dir: Path) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": float(config["epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "per_device_train_batch_size": int(config["batch_size"]),
        "per_device_eval_batch_size": int(config["eval_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "warmup_ratio": float(config["warmup_ratio"]),
        "weight_decay": float(config["weight_decay"]),
        "lr_scheduler_type": str(config["lr_scheduler_type"]),
        "logging_steps": int(config["logging_steps"]),
        "save_steps": int(config["save_steps"]),
        "eval_steps": int(config["eval_steps"]),
        "save_total_limit": int(config["save_total_limit"]),
        "report_to": config.get("report_to") or [],
        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),
        "fp16": bool(config["fp16"]),
        "bf16": bool(config["bf16"]),
        "gradient_checkpointing": bool(config["gradient_checkpointing"]),
        "remove_unused_columns": False,
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    return TrainingArguments(**kwargs)


def build_trainer(
    model: Any,
    tokenizer: Any,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    data_collator: Any,
    log_callback: TrainerCallback,
) -> Trainer:
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "callbacks": [log_callback],
    }
    parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


class CsvLogCallback(TrainerCallback):
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.fieldnames: list[str] = []

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return
        row = {"step": state.global_step, **{key: value for key, value in logs.items() if isinstance(value, (int, float, str))}}
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.fieldnames:
            self.fieldnames = list(row)
            with self.output_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerow(row)
        else:
            for key in row:
                if key not in self.fieldnames:
                    self.fieldnames.append(key)
            with self.output_path.open("a", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writerow({key: row.get(key) for key in self.fieldnames})


def count_parameters(model: Any) -> dict[str, Any]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_param_pct": float(trainable / total * 100.0) if total else 0.0,
    }


def gpu_metrics() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"peak_gpu_memory_allocated_gb": None, "peak_gpu_memory_reserved_gb": None}
    gb = 1024.0**3
    return {
        "peak_gpu_memory_allocated_gb": float(torch.cuda.max_memory_allocated() / gb),
        "peak_gpu_memory_reserved_gb": float(torch.cuda.max_memory_reserved() / gb),
    }


def numeric_dict(values: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in values.items() if isinstance(value, (int, float, np.floating))}


def build_adapter_manifest(
    config: dict[str, Any],
    run_dir: Path,
    adapter_dir: Path,
    parameter_counts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "phase5",
        "model_name": config["model_name"],
        "experiment_name": config["experiment_name"],
        "lora_r": int(config["lora_r"]),
        "seed": int(config["seed"]),
        "adapter_dir": str(adapter_dir),
        "predictions_path": str(PROJECT_ROOT / "outputs" / "phase5" / "eval" / str(config["experiment_name"]) / "predictions.csv"),
        "accuracy": None,
        "macro_f1": None,
        "run_dir": str(run_dir),
        **parameter_counts,
    }


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
