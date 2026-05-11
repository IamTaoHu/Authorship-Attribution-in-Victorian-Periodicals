"""Evaluate Phase 5 decoder QLoRA adapters with generation-based metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.label_mapping import get_author_id_to_name
from src.inference.author_parser import parse_author_prediction
from src.inference.run_prompting import clean_generated_output
from src.training.train_decoder_qlora import resolve_torch_dtype
from src.utils.config import load_config
from src.utils.reproducibility import set_seed


PREDICTION_COLUMNS = [
    "sample_id",
    "text",
    "true_label",
    "true_author",
    "pred_label",
    "predicted_author",
    "raw_output",
    "cleaned_output",
    "parsed_status",
    "correct",
    "input_tokens",
    "output_tokens",
    "inference_time_sec",
    "model_name",
    "experiment_name",
    "lora_r",
    "seed",
    "adapter_dir",
    "predictions_path",
    "accuracy",
    "macro_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 5 QLoRA adapter.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--adapter_dir", required=True, type=Path)
    parser.add_argument("--test_file", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/phase5/eval"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_phase5_config(args.config)
    seed = int(args.seed if args.seed is not None else config["seed"])
    set_seed(seed)
    rows = limit_rows(read_jsonl(resolve_repo_path(args.test_file)), args.max_samples)
    if not rows:
        raise ValueError("No test rows available for Phase 5 evaluation.")
    output_dir = resolve_repo_path(args.output_dir) / str(config["experiment_name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model_tokenizer_adapter(config, resolve_repo_path(args.adapter_dir))
    predictions = generate_predictions(
        rows=rows,
        tokenizer=tokenizer,
        model=model,
        config=config,
        adapter_dir=resolve_repo_path(args.adapter_dir),
        output_dir=output_dir,
        max_new_tokens=int(args.max_new_tokens if args.max_new_tokens is not None else config["max_new_tokens"]),
    )
    metrics = compute_metrics(predictions, config)
    for row in predictions:
        row["accuracy"] = metrics["accuracy"]
        row["macro_f1"] = metrics["macro_f1"]
    write_predictions(predictions, output_dir / "predictions.csv")
    write_metrics_files(predictions, metrics, config, output_dir)
    update_adapter_manifest(resolve_repo_path(args.adapter_dir), output_dir / "predictions.csv", metrics)
    print(f"Wrote Phase 5 evaluation outputs to {output_dir}")


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def load_phase5_config(path: Path) -> dict[str, Any]:
    config = load_config(str(resolve_repo_path(path)))
    return dict(config)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return rows if limit is None else rows[: max(0, int(limit))]


def load_model_tokenizer_adapter(config: dict[str, Any], adapter_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=resolve_torch_dtype(config.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return tokenizer, model


def generate_predictions(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    config: dict[str, Any],
    adapter_dir: Path,
    output_dir: Path,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    raw_path = output_dir / "raw_generations.jsonl"
    predictions: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for row in rows:
            generation = generate_one(str(row["prompt"]), tokenizer, model, max_new_tokens)
            cleaned = clean_generated_output(generation["raw_output"], row["prompt"])
            parsed = parse_author_prediction(cleaned)
            pred_label = parsed["parsed_label"]
            pred_author = parsed["parsed_author"]
            correct = pred_label is not None and int(pred_label) == int(row["true_label"])
            prediction = {
                "sample_id": row["sample_id"],
                "text": row["text"],
                "true_label": int(row["true_label"]),
                "true_author": row["true_author"],
                "pred_label": pred_label,
                "predicted_author": pred_author,
                "raw_output": generation["raw_output"],
                "cleaned_output": cleaned,
                "parsed_status": parsed["parse_status"],
                "correct": bool(correct),
                "input_tokens": generation["input_tokens"],
                "output_tokens": generation["output_tokens"],
                "inference_time_sec": generation["inference_time_sec"],
                "model_name": config["model_name"],
                "experiment_name": config["experiment_name"],
                "lora_r": int(config["lora_r"]),
                "seed": int(config["seed"]),
                "adapter_dir": str(adapter_dir),
                "predictions_path": str(output_dir / "predictions.csv"),
                "accuracy": None,
                "macro_f1": None,
            }
            predictions.append(prediction)
            raw_file.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "prompt": row["prompt"],
                        "true_author": row["true_author"],
                        "raw_generation": generation["raw_output"],
                        "cleaned_generation": cleaned,
                        "parsed_prediction": parsed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return predictions


def generate_one(prompt: str, tokenizer: Any, model: Any, max_new_tokens: int) -> dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[1])
    start = perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = perf_counter() - start
    generated_ids = outputs[0][input_tokens:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "raw_output": raw_output,
        "input_tokens": input_tokens,
        "output_tokens": int(generated_ids.shape[0]),
        "inference_time_sec": float(elapsed),
    }


def compute_metrics(predictions: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    valid_authors = list(get_author_id_to_name().values())
    y_true = [str(row["true_author"]) for row in predictions]
    y_pred = [str(row["predicted_author"] or "__INVALID__") for row in predictions]
    metric_pred = [pred if pred in valid_authors else "__INVALID__" for pred in y_pred]
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, metric_pred, labels=valid_authors, average="macro", zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, metric_pred, labels=valid_authors, average="weighted", zero_division=0)
    correct = [bool(row["correct"]) for row in predictions]
    invalid = [not (str(row["parsed_status"]).startswith("ok") and row["predicted_author"] in valid_authors) for row in predictions]
    return {
        "experiment_name": config["experiment_name"],
        "model_name": config["model_name"],
        "lora_r": int(config["lora_r"]),
        "seed": int(config["seed"]),
        "n_samples": len(predictions),
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "invalid_output_rate": float(np.mean(invalid)) if invalid else 0.0,
        "avg_inference_time_sec": float(np.mean([row["inference_time_sec"] for row in predictions])) if predictions else 0.0,
        "total_inference_time_sec": float(sum(row["inference_time_sec"] for row in predictions)),
        **gpu_metrics(),
    }


def write_predictions(predictions: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(predictions)


def write_metrics_files(predictions: list[dict[str, Any]], metrics: dict[str, Any], config: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    valid_authors = list(get_author_id_to_name().values())
    y_true = [str(row["true_author"]) for row in predictions]
    y_pred = [str(row["predicted_author"] or "__INVALID__") for row in predictions]
    metric_pred = [pred if pred in valid_authors else "__INVALID__" for pred in y_pred]
    report = classification_report(y_true, metric_pred, labels=valid_authors, target_names=valid_authors, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")
    matrix = confusion_matrix(y_true, metric_pred, labels=valid_authors)
    pd.DataFrame(matrix, index=valid_authors, columns=valid_authors).to_csv(output_dir / "confusion_matrix.csv")
    runtime = {
        **metrics,
        "max_new_tokens": int(config["max_new_tokens"]),
        "use_4bit": True,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


def update_adapter_manifest(adapter_dir: Path, predictions_path: Path, metrics: dict[str, Any]) -> None:
    manifest_path = adapter_dir.parent / "adapter_manifest.json"
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["predictions_path"] = str(predictions_path)
    payload["accuracy"] = metrics.get("accuracy")
    payload["macro_f1"] = metrics.get("macro_f1")
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def gpu_metrics() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"peak_gpu_memory_allocated_gb": None, "peak_gpu_memory_reserved_gb": None}
    gb = 1024.0**3
    return {
        "peak_gpu_memory_allocated_gb": float(torch.cuda.max_memory_allocated() / gb),
        "peak_gpu_memory_reserved_gb": float(torch.cuda.max_memory_reserved() / gb),
    }


if __name__ == "__main__":
    main()
