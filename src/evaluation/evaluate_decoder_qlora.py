"""Evaluate Phase 5 QLoRA decoder adapters on PERIAD."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import torch
import yaml
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.prompting.phase4_parser import CANONICAL_AUTHORS, is_valid_parse, parse_phase4_output  # noqa: E402
from src.training.train_decoder_qlora import build_instruction, load_config, output_path, read_label_map, read_split, resolve_dtype  # noqa: E402
from src.utils.paths import dataset_dir  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase5" / "diagnostic_tinyllama_qlora.yaml"
PREDICTION_COLUMNS = (
    "sample_id",
    "text",
    "true_author",
    "raw_output",
    "cleaned_output",
    "predicted_author",
    "parsed_status",
    "correct",
    "generated_length",
    "inference_time_sec",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    numeric = display.select_dtypes(include="number").columns
    display[numeric] = display[numeric].round(4)
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(lines)


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


def render_prompt(tokenizer: Any, instruction: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{instruction} "


def load_adapter_model(config: dict[str, Any]) -> tuple[Any, Any]:
    model_name = str(config["model_name"])
    adapter_dir = output_path(config, "checkpoint_dir")
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing Phase 5 adapter checkpoint: {adapter_dir}")
    token = os.environ.get(str(config.get("hf_token_env", "HF_TOKEN")))
    kwargs: dict[str, Any] = {"trust_remote_code": bool(config.get("trust_remote_code", False))}
    if token:
        kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else model_name, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {**kwargs, "device_map": "auto" if torch.cuda.is_available() else None}
    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
    try:
        quantization = build_quantization_config(config)
        if quantization is not None:
            model_kwargs["quantization_config"] = quantization
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as exc:
        allow_fallback = bool(config.get("quantization", {}).get("allow_diagnostic_fallback_without_4bit", False))
        if not allow_fallback:
            raise
        print(f"WARNING: 4-bit eval load failed; diagnostic fallback will load without quantization: {exc}")
        model_kwargs.pop("quantization_config", None)
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return tokenizer, model


def generate_one(tokenizer: Any, model: Any, text: str, config: dict[str, Any]) -> tuple[str, int, float]:
    prompt = render_prompt(tokenizer, build_instruction(text, config))
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(config["prompt"]["max_length"]))
    input_len = int(inputs["input_ids"].shape[-1])
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    gen_cfg = config.get("generation", {})
    generation_kwargs = {
        "max_new_tokens": int(config["prompt"].get("max_new_tokens", 16)),
        "do_sample": bool(gen_cfg.get("do_sample", False)),
        "temperature": float(gen_cfg.get("temperature", 0.0)),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if not generation_kwargs["do_sample"]:
        generation_kwargs.pop("temperature", None)
    started = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    elapsed = time.time() - started
    generated_ids = output_ids[0, input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), int(generated_ids.shape[-1]), elapsed


def write_tables_and_report(config: dict[str, Any], metrics: dict[str, Any], report_frame: pd.DataFrame) -> None:
    phase_dir = REPO_ROOT / "outputs" / "phase5"
    if config.get("mode") == "diagnostic":
        tables_dir = phase_dir / "diagnostic" / "tables"
        reports_dir = phase_dir / "diagnostic" / "reports"
        tables_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        row = {key: metrics.get(key) for key in ("run_name", "model_name", "accuracy", "macro_f1", "weighted_f1", "invalid_output_rate", "n_samples")}
        table = pd.DataFrame([row])
        csv_path = tables_dir / "diagnostic_decoder_qlora_results.csv"
        md_path = tables_dir / "diagnostic_decoder_qlora_results.md"
        table.to_csv(csv_path, index=False)
        table_md = markdown_table(table)
        md_path.write_text(table_md + "\n", encoding="utf-8")
        report_path = reports_dir / "diagnostic_report.md"
        report_path.write_text(
            "\n".join(
                [
                    "# Phase 5 Diagnostic QLoRA Report",
                    "",
                    "This is a diagnostic-only run for validating the Phase 5 code path on local hardware.",
                    "It is not a full research benchmark.",
                    "",
                    table_md,
                    "",
                    "Full Mistral/Llama/Gemma result plots are skipped until Colab runs produce valid full tables.",
                ]
            ),
            encoding="utf-8",
        )
        print(f"Wrote {csv_path}")
        print(f"Wrote {md_path}")
        print(f"Wrote {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = output_path(config, "run_dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    periad_dir = dataset_dir("processed", "periad")
    label_map = read_label_map(periad_dir)
    write_json(run_dir / "label_mapping.json", label_map)
    eval_cfg = config["dataset"]
    eval_frame = read_split(periad_dir, str(eval_cfg.get("eval_split", "test")), eval_cfg.get("max_eval_samples"))
    tokenizer, model = load_adapter_model(config)
    rows = []
    for _, row in eval_frame.iterrows():
        try:
            raw_output, generated_length, elapsed = generate_one(tokenizer, model, str(row["text"]), config)
            parsed = parse_phase4_output(raw_output)
        except Exception as exc:
            raw_output = ""
            generated_length = 0
            elapsed = 0.0
            parsed = parse_phase4_output("")
            parsed = parsed.__class__("", "invalid_inference_failed", str(exc).replace("\n", " "), raw_output, "")
        predicted = parsed.predicted_author
        rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "text": str(row["text"]),
                "true_author": str(row["author"]),
                "raw_output": raw_output,
                "cleaned_output": parsed.cleaned_output,
                "predicted_author": predicted,
                "parsed_status": parsed.parsed_status,
                "correct": bool(predicted == str(row["author"])) if predicted else False,
                "generated_length": generated_length,
                "inference_time_sec": elapsed,
            }
        )
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    print(f"Wrote {run_dir / 'predictions.csv'}")
    y_true = predictions["true_author"].astype(str)
    y_pred = predictions["predicted_author"].fillna("").astype(str)
    valid_count = int(predictions["parsed_status"].map(is_valid_parse).sum())
    metrics = {
        "run_name": config["run_name"],
        "model_name": config["model_name"],
        "mode": config.get("mode", ""),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(predictions) else None,
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CANONICAL_AUTHORS), average="macro", zero_division=0)) if len(predictions) else None,
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(CANONICAL_AUTHORS), average="weighted", zero_division=0)) if len(predictions) else None,
        "invalid_output_rate": float(1 - valid_count / len(predictions)) if len(predictions) else None,
        "average_generated_length": float(pd.to_numeric(predictions["generated_length"], errors="coerce").mean()) if len(predictions) else None,
        "avg_inference_time_sec": float(pd.to_numeric(predictions["inference_time_sec"], errors="coerce").mean()) if len(predictions) else None,
        "total_inference_time_sec": float(pd.to_numeric(predictions["inference_time_sec"], errors="coerce").sum()) if len(predictions) else 0.0,
        "n_samples": int(len(predictions)),
        "n_valid_predictions": valid_count,
    }
    write_json(run_dir / "metrics.json", metrics)
    report_text = classification_report(y_true, y_pred, labels=list(CANONICAL_AUTHORS), zero_division=0)
    (run_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    report_dict = classification_report(y_true, y_pred, labels=list(CANONICAL_AUTHORS), output_dict=True, zero_division=0)
    report_frame = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})
    report_frame.to_csv(run_dir / "classification_report.csv", index=False)
    print(f"Wrote {run_dir / 'classification_report.txt'}")
    print(f"Wrote {run_dir / 'classification_report.csv'}")
    write_tables_and_report(config, metrics, report_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
