"""Train Phase 5 decoder LLM adapters with QLoRA-style LoRA SFT."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - depends on installed transformers version
    BitsAndBytesConfig = None

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.prompting.phase4_parser import CANONICAL_AUTHORS  # noqa: E402
from src.utils.paths import dataset_dir  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase5" / "diagnostic_tinyllama_qlora.yaml"
REQUIRED_COLUMNS = ("sample_id", "text", "label", "author")
COMMON_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    if config.get("phase") != "phase5":
        raise ValueError(f"Expected phase5 config, found: {config.get('phase')!r}")
    return config


def repo_relative_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def output_path(config: dict[str, Any], key: str) -> Path:
    return repo_relative_path(config["paths"][key])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_label_map(periad_dir: Path) -> dict[str, int]:
    path = periad_dir / "label_map.json"
    if not path.exists():
        raise FileNotFoundError(f"Required PERIAD label map not found: {path}")
    label_map = json.loads(path.read_text(encoding="utf-8"))
    if list(label_map.keys()) != list(CANONICAL_AUTHORS):
        raise ValueError(f"Unexpected PERIAD author order in {path}: {list(label_map.keys())}")
    return {str(author): int(label) for author, label in label_map.items()}


def read_split(periad_dir: Path, split_name: str, max_samples: int | None) -> pd.DataFrame:
    path = periad_dir / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required PERIAD split not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["author"] = frame["author"].astype(str)
    frame["label"] = frame["label"].astype(int)
    if max_samples:
        frame = frame.head(int(max_samples)).copy()
    return frame


def build_instruction(text: str, config: dict[str, Any]) -> str:
    prompt_config = config.get("prompt", {})
    max_chars = prompt_config.get("max_input_chars")
    text = str(text)
    if max_chars:
        text = text[: int(max_chars)]
    authors = "\n".join(f"- {author}" for author in CANONICAL_AUTHORS)
    return (
        f"{prompt_config.get('system_message', '')}\n\n"
        "Choose the author of the passage from this list:\n"
        f"{authors}\n\n"
        "Passage:\n"
        f"{text}\n\n"
        "Answer:"
    ).strip()


def render_training_text(tokenizer: Any, instruction: str, author: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": author},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return f"{instruction} {author}{tokenizer.eos_token or ''}"


def render_prompt_text(tokenizer: Any, instruction: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{instruction} "


def tokenize_row(row: dict[str, Any], tokenizer: Any, config: dict[str, Any]) -> dict[str, Any]:
    max_length = int(config["prompt"]["max_length"])
    instruction = build_instruction(str(row["text"]), config)
    prompt_text = render_prompt_text(tokenizer, instruction)
    full_text = render_training_text(tokenizer, instruction, str(row["author"]))
    full = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
    prompt = tokenizer(prompt_text, truncation=True, max_length=max_length, padding=False)
    prompt_len = min(len(prompt["input_ids"]), len(full["input_ids"]))
    labels = list(full["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len
    if labels and all(label == -100 for label in labels):
        labels[-1] = full["input_ids"][-1]
    full["labels"] = labels
    return full


def resolve_dtype(value: str) -> torch.dtype:
    normalized = str(value or "float16").lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    return torch.float16


def build_quantization_config(config: dict[str, Any]) -> Any | None:
    quant = config.get("quantization", {})
    if not bool(quant.get("load_in_4bit", False)):
        return None
    if BitsAndBytesConfig is None:
        raise RuntimeError("4-bit quantization requested but BitsAndBytesConfig is unavailable.")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(quant.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=resolve_dtype(str(quant.get("bnb_4bit_compute_dtype", "float16"))),
    )


def load_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any, str]:
    model_name = str(config["model_name"])
    token = os.environ.get(str(config.get("hf_token_env", "HF_TOKEN")))
    kwargs: dict[str, Any] = {"trust_remote_code": bool(config.get("trust_remote_code", False))}
    if token:
        kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_kwargs: dict[str, Any] = {
        **kwargs,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
    quantization_status = "none"
    try:
        quantization = build_quantization_config(config)
        if quantization is not None:
            model_kwargs["quantization_config"] = quantization
            quantization_status = "4bit"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as exc:
        allow_fallback = bool(config.get("quantization", {}).get("allow_diagnostic_fallback_without_4bit", False))
        if not allow_fallback:
            raise
        print(f"WARNING: 4-bit load failed; diagnostic fallback will load without quantization: {exc}")
        model_kwargs.pop("quantization_config", None)
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        quantization_status = "fallback_no_4bit"
    model.config.use_cache = False
    return tokenizer, model, quantization_status


def resolve_lora_targets(config: dict[str, Any]) -> list[str]:
    targets = config.get("lora", {}).get("target_modules", "auto")
    if targets == "auto":
        return list(COMMON_LORA_TARGETS)
    if isinstance(targets, str):
        return [item.strip() for item in targets.split(",") if item.strip()]
    return [str(item) for item in targets]


def build_training_args(config: dict[str, Any], checkpoint_path: Path) -> TrainingArguments:
    train = config["training"]
    kwargs: dict[str, Any] = {
        "output_dir": str(checkpoint_path),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": int(train["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(train["gradient_accumulation_steps"]),
        "learning_rate": float(train["learning_rate"]),
        "num_train_epochs": float(train.get("num_train_epochs", 1)),
        "max_steps": int(train.get("max_steps", -1)),
        "warmup_steps": int(train.get("warmup_steps", 0)),
        "warmup_ratio": float(train.get("warmup_ratio", 0.0)),
        "logging_steps": int(train.get("logging_steps", 10)),
        "save_steps": int(train.get("save_steps", 250)),
        "save_total_limit": int(train.get("save_total_limit", 2)),
        "fp16": bool(train.get("fp16", False) and torch.cuda.is_available()),
        "bf16": bool(train.get("bf16", False) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "optim": str(train.get("optim", "adamw_torch")),
        "report_to": train.get("report_to", []),
        "remove_unused_columns": False,
        "seed": int(train.get("seed", 42)),
    }
    if kwargs["max_steps"] <= 0:
        kwargs.pop("max_steps")
    if kwargs["optim"] == "paged_adamw_8bit":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("WARNING: bitsandbytes optimizer unavailable; using adamw_torch for this run.")
            kwargs["optim"] = "adamw_torch"
    if not torch.cuda.is_available() and kwargs["optim"] == "paged_adamw_8bit":
        kwargs["optim"] = "adamw_torch"
    signature = inspect.signature(TrainingArguments.__init__).parameters
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in signature})


class JsonlTrainer(Trainer):
    def __init__(self, *args: Any, log_path: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_path = log_path

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:  # type: ignore[override]
        super().log(logs, start_time=start_time) if "start_time" in inspect.signature(super().log).parameters else super().log(logs)
        payload = dict(logs)
        payload["step"] = int(self.state.global_step)
        append_jsonl(self.log_path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    started = time.time()
    config_path = Path(args.config)
    config = load_config(config_path)
    set_seed(int(config["training"].get("seed", 42)))
    run_dir = output_path(config, "run_dir")
    checkpoint_path = output_path(config, "checkpoint_dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    periad_dir = dataset_dir("processed", "periad")
    label_map = read_label_map(periad_dir)
    train_cfg = config["dataset"]
    train_frame = read_split(periad_dir, str(train_cfg.get("train_split", "train")), train_cfg.get("max_train_samples"))
    config_dump = {**config, "config_path": str(config_path.resolve()), "dataset_dir": str(periad_dir)}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_dump, sort_keys=False), encoding="utf-8")
    write_json(run_dir / "label_mapping.json", label_map)
    (run_dir / "training_log.jsonl").write_text("", encoding="utf-8")

    tokenizer, model, quantization_status = load_model_and_tokenizer(config)
    if quantization_status == "4bit":
        model = prepare_model_for_kbit_training(model)
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
        target_modules=resolve_lora_targets(config),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = Dataset.from_pandas(train_frame, preserve_index=False)
    tokenized = dataset.map(lambda row: tokenize_row(row, tokenizer, config), remove_columns=list(train_frame.columns))
    args_train = build_training_args(config, checkpoint_path)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args_train,
        "train_dataset": tokenized,
        "log_path": run_dir / "training_log.jsonl",
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = JsonlTrainer(**trainer_kwargs)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)

    train_metrics = {
        "run_name": config["run_name"],
        "model_name": config["model_name"],
        "mode": config.get("mode", ""),
        "train_samples": int(len(train_frame)),
        "quantization_status": quantization_status,
        "runtime_seconds": round(time.time() - started, 4),
        **{key: float(value) if isinstance(value, (int, float)) else value for key, value in result.metrics.items()},
    }
    write_json(run_dir / "train_metrics.json", train_metrics)
    print(f"Saved adapter to {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
