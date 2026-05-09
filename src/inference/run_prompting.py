"""Run Phase 4 decoder prompting experiments."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.label_mapping import get_author_id_to_name
from src.inference.author_parser import canonical_author_from_single_line, parse_author_prediction
from src.inference.few_shot_selection import (
    get_test_split,
    infer_required_columns,
    load_periad_dataset,
    parse_label,
    sample_id_for_row,
    select_and_save_few_shot_examples,
)
from src.inference.prompt_templates import load_few_shot_examples, render_prompt
from src.inference.prompting_config import PromptingConfig, load_prompting_config
from src.utils.reproducibility import set_seed


PREDICTION_COLUMNS = [
    "sample_id",
    "text",
    "true_label",
    "true_author",
    "true_label_name",
    "pred_label",
    "pred_label_name",
    "predicted_author",
    "raw_output",
    "cleaned_output",
    "parsed_status",
    "parser_notes",
    "prompt_type",
    "model_name",
    "input_tokens",
    "output_tokens",
    "inference_time_sec",
    "correct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 decoder prompting.")
    parser.add_argument("--config", required=True, type=Path, help="Path to a Phase 4 YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data/prompts/parser without model loading.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override config max_samples for this run.")
    parser.add_argument("--select-few-shot", action="store_true", help="Generate few_shot_examples.json and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_prompting_config(args.config)
    if args.max_samples is not None:
        config = override_max_samples(config, args.max_samples)
    set_seed(config.seed)
    random.seed(config.seed)

    if args.select_few_shot:
        select_and_save_few_shot_examples(config.few_shot_examples_path, seed=config.seed)
        return

    if args.dry_run:
        dry_run(config)
        return

    run_generation(config)


def override_max_samples(config: PromptingConfig, max_samples: int) -> PromptingConfig:
    """Return a config copy with max_samples overridden."""
    if max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer.")
    return PromptingConfig(
        model_name=config.model_name,
        prompt_type=config.prompt_type,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=config.do_sample,
        top_p=config.top_p,
        few_shot_examples_path=config.few_shot_examples_path,
        batch_size=config.batch_size,
        output_dir=config.output_dir,
        use_4bit=config.use_4bit,
        max_input_chars=config.max_input_chars,
        max_samples=max_samples,
        seed=config.seed,
    )


def dry_run(config: PromptingConfig) -> None:
    """Validate a run without loading model weights or generating text."""
    run_dir = config.output_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = run_metadata(config)
    dataset, text_column, label_column, rows = load_test_rows(config)
    del dataset
    few_shot_examples = examples_for_config(config, allow_empty=True)
    first_row = rows[0]
    prompt = render_prompt(config, first_row["text"], few_shot_examples)
    parser_checks = build_parser_checks(prompt)
    preview = {
        "run_name": config.run_name,
        "model_name": config.model_name,
        "prompt_type": config.prompt_type,
        **metadata,
        "output_run_dir": str(run_dir),
        "dataset_columns": {"text_column": text_column, "label_column": label_column},
        "num_rows_loaded": len(rows),
        "first_sample": {
            "sample_id": first_row["sample_id"],
            "true_label": first_row["true_label"],
            "true_author": first_row["true_author"],
            "prompt_preview": prompt[:2000],
            "prompt_tail_preview": prompt[-2000:],
            "prompt_ends_with_answer_slot": prompt.rstrip().endswith("Answer:"),
            "final_target_format": "plain_target_answer_slot" if config.prompt_type == "few_shot" else None,
        },
        "few_shot_example_count": len(few_shot_examples),
        "parser_checks": parser_checks,
    }
    write_json(preview, run_dir / "dry_run_preview.json")
    print(json.dumps(preview, indent=2))


def run_generation(config: PromptingConfig) -> None:
    """Run decoder generation over the PERIAD test split."""
    run_dir = config.output_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    _, _, _, rows = load_test_rows(config)
    few_shot_examples = examples_for_config(config, allow_empty=False)
    tokenizer, model = load_model_and_tokenizer(config)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    raw_path = run_dir / "raw_outputs.jsonl"
    predictions: list[dict[str, Any]] = []
    total_output_tokens = 0
    total_time_start = perf_counter()
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for row in rows:
            prompt = render_prompt(config, row["text"], few_shot_examples)
            generation = generate_one(prompt, tokenizer, model, config)
            generation["cleaned_output"] = clean_generated_output(generation["raw_output"], prompt)
            parsed_prediction = parse_author_prediction(generation["cleaned_output"])
            prediction_row = build_prediction_row(row, parsed_prediction, generation, config)
            predictions.append(prediction_row)
            total_output_tokens += int(generation["output_tokens"])
            raw_file.write(
                json.dumps(
                    {
                        "input_text": row["text"],
                        "prompt": prompt,
                        "raw_generation": generation["raw_output"],
                        "cleaned_generation": generation["cleaned_output"],
                        "parsed_prediction": parsed_prediction,
                        "true_label": row["true_label"],
                        "true_author": row["true_author"],
                        "prompt_type": config.prompt_type,
                        "model_name": config.model_name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total_time = perf_counter() - total_time_start
    clear_cuda_cache()
    write_predictions(predictions, run_dir / "predictions.csv")
    write_json(build_runtime(config, len(rows), total_time, total_output_tokens), run_dir / "runtime.json")
    print(f"Wrote Phase 4 prompting outputs to {run_dir}")


def load_test_rows(config: PromptingConfig) -> tuple[Any, str, str, list[dict[str, Any]]]:
    """Load and normalize PERIAD test rows."""
    dataset = load_periad_dataset()
    test_split = get_test_split(dataset)
    text_column, label_column = infer_required_columns(test_split)
    label_to_name = get_author_id_to_name()
    limit = config.max_samples if config.max_samples is not None else len(test_split)
    rows: list[dict[str, Any]] = []
    for index in range(min(limit, len(test_split))):
        row = test_split[index]
        label = parse_label(row[label_column])
        if label is None or label not in label_to_name:
            warnings.warn(f"Skipping test row {index} with unmapped label {row[label_column]!r}.", RuntimeWarning, stacklevel=2)
            continue
        text = " ".join(str(row[text_column]).split())
        rows.append(
            {
                "sample_id": sample_id_for_row(row, index).replace("train_", "test_"),
                "text": text,
                "true_label": int(label),
                "true_author": label_to_name[int(label)],
            }
        )
    if not rows:
        raise ValueError("No usable test rows loaded for Phase 4 prompting.")
    return dataset, text_column, label_column, rows


def examples_for_config(config: PromptingConfig, allow_empty: bool) -> list[dict[str, Any]]:
    """Load few-shot examples when the prompt type needs them."""
    if config.prompt_type != "few_shot":
        return []
    examples = load_few_shot_examples(config.few_shot_examples_path)
    if not examples and allow_empty:
        examples_payload = select_and_save_few_shot_examples(config.few_shot_examples_path, seed=config.seed)
        examples = examples_payload["examples"]
    if not examples:
        raise ValueError(f"few_shot prompt requires non-empty examples at {config.few_shot_examples_path}.")
    return examples


def build_parser_checks(prompt: str) -> dict[str, Any]:
    """Build dry-run parser checks that exercise cleanup before parsing."""
    raw_cases = {
        "full_name": "Eliza Lynn Linton",
        "last_name": "The author is Morley.",
        "cot_final_answer": "Reasoning here.\nFinal answer: George Lewes",
        "georges_lewes_alias": "Final answer: Georges Lewes",
        "g_h_lewes_alias": "Final answer: G. H. Lewes",
        "gh_lewes_alias": "Final answer: GH Lewes",
        "multiple_authors": "John Morley or Anne Mozley",
        "hallucinated": "Final answer: Charles Dickens",
        "answer_prefix": "Answer: Anne Mozley",
        "author_prefix": "Author: John Morley",
        "answer_slot_echo": "Text: some echoed text\nAnswer: John Morley",
        "multiple_answer_slots": "Text: example\nAnswer: Leslie Stephen\nTarget:\nText: target text\nAnswer: Eliza Lynn Linton",
        "old_prompt_echo": "Example 1:\nParagraph: old prompt echo",
        "target_text_echo": "Target:\nText: copied target",
        "text_echo_with_answer": "Text: copied target\nAnswer: Anne Mozley",
        "plain_target_echo_with_author": "Classify this target text.\nJohn Morley",
        "author_first_full_name_with_echo": "Leslie Stephen\n\nExamples:\nText: echoed text",
        "author_first_last_name_with_echo": "Morley\n\nExplanation: short reasoning",
        "author_first_lewes_alias_with_echo": "G. H. Lewes\n\nExamples:\nText: echoed text",
        "author_first_hallucinated_with_echo": "Bulwer Lytton\n\nExplanation: ...",
        "author_first_multiple_authors": "John Morley or Anne Mozley",
        "example_prefix": "Example:\nLeslie Stephen",
        "prompt_echo": f"{prompt}\nEliza Lynn Linton",
    }
    checks: dict[str, Any] = {}
    for name, raw_output in raw_cases.items():
        cleaned = clean_generated_output(raw_output, prompt)
        checks[name] = {
            "raw_output": raw_output,
            "cleaned_output": cleaned,
            "parsed_prediction": parse_author_prediction(cleaned),
        }
    return checks


def load_model_and_tokenizer(config: PromptingConfig):
    """Load tokenizer and causal LM for generation."""
    assert_local_vram_is_suitable(config)
    model_kwargs: dict[str, Any] = {}
    if config.use_4bit:
        require_bitsandbytes_4bit()
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        if accelerate_available():
            model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    if not config.use_4bit and "device_map" not in model_kwargs:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    model.eval()
    return tokenizer, model


def require_bitsandbytes_4bit() -> None:
    """Fail clearly unless bitsandbytes 4-bit loading is available."""
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError("use_4bit is true, but bitsandbytes is not installed. Install bitsandbytes or set use_4bit false.")
    if not torch.cuda.is_available():
        raise RuntimeError("use_4bit is true, but CUDA is not available. bitsandbytes 4-bit inference requires CUDA here.")
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(f"use_4bit is true, but bitsandbytes could not be imported: {exc}") from exc


def generate_one(prompt: str, tokenizer, model, config: PromptingConfig) -> dict[str, Any]:
    """Generate one response and return timing/token metadata."""
    clear_cuda_cache()
    inputs = tokenizer(prompt, return_tensors="pt")
    device = infer_model_input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[1])
    start = perf_counter()
    try:
        with torch.inference_mode():
            generation_kwargs: dict[str, Any] = {
                **inputs,
                "max_new_tokens": config.max_new_tokens,
                "do_sample": config.do_sample,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "repetition_penalty": 1.05,
            }
            if config.do_sample:
                generation_kwargs["temperature"] = config.temperature
                generation_kwargs["top_p"] = config.top_p
            outputs = model.generate(**generation_kwargs)
    finally:
        clear_cuda_cache()
    elapsed = perf_counter() - start
    generated_ids = outputs[0][input_tokens:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "raw_output": raw_output,
        "cleaned_output": clean_generated_output(raw_output, prompt),
        "input_tokens": input_tokens,
        "output_tokens": int(generated_ids.shape[0]),
        "inference_time_sec": float(elapsed),
    }


def clean_generated_output(raw_output: str, prompt: str | None = None) -> str:
    """Clean generated text before parser normalization while preserving raw output separately."""
    original = str(raw_output or "").strip()
    text = original
    if not text:
        return ""

    if prompt:
        prompt_text = str(prompt).strip()
        if text.startswith(prompt_text):
            text = text[len(prompt_text):].strip()

    lower_text = text.lower()
    if "final answer:" in lower_text:
        return text

    first_line_author = canonical_author_from_first_output_line(text)
    if first_line_author is not None:
        return first_line_author

    answer_slot = extract_last_answer_slot(text)
    if answer_slot is not None:
        return answer_slot or original

    for prefix in ("author:", "answer:"):
        if lower_text.startswith(prefix):
            cleaned = text[len(prefix):].strip()
            return cleaned or original

    clear_line_author = extract_author_after_prompt_echo(text)
    if clear_line_author is not None:
        return clear_line_author

    text = strip_prompt_echo_sections(text)
    lower_text = text.lower()
    answer_slot = extract_last_answer_slot(text)
    if answer_slot is not None:
        return answer_slot or original

    clear_line_author = extract_author_after_prompt_echo(text)
    if clear_line_author is not None:
        return clear_line_author

    for prefix in ("example:", "explanation:"):
        if lower_text.startswith(prefix):
            remainder = text[len(prefix):].strip()
            if has_single_parseable_author(remainder):
                return first_nonempty_line(remainder)
            return remainder or original

    return text or original


def canonical_author_from_first_output_line(text: str) -> str | None:
    """Return a canonical author if the first non-empty output line is exactly one valid label or alias."""
    return canonical_author_from_single_line(first_nonempty_line(text))


def extract_last_answer_slot(text: str) -> str | None:
    """Return text after the last Answer: marker, limited to the first non-empty line."""
    marker = "answer:"
    lowered = text.lower()
    index = lowered.rfind(marker)
    if index < 0:
        return None
    answer_text = text[index + len(marker):].strip()
    return first_nonempty_line(answer_text)


def strip_prompt_echo_sections(text: str) -> str:
    """Strip answer-slot prompt echo sections before parser normalization."""
    cleaned = str(text).strip()
    lowered = cleaned.lower()
    if "target:" in lowered:
        target_index = lowered.rfind("target:")
        cleaned = cleaned[target_index + len("target:"):].strip()
        lowered = cleaned.lower()
    if lowered.startswith("text:"):
        answer_slot = extract_last_answer_slot(cleaned)
        if answer_slot is not None:
            return answer_slot
        return ""
    return cleaned


def extract_author_after_prompt_echo(text: str) -> str | None:
    """Extract a clear author label on a line after echoed prompt-control text."""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    prompt_terms = ("classify this target text", "target:", "text:")
    if not any(lines[0].lower().startswith(term) for term in prompt_terms):
        return None
    for line in lines[1:]:
        lowered = line.lower()
        if lowered.startswith(("target:", "text:", "answer:")):
            continue
        if has_single_parseable_author(line):
            return line
    return None


def has_single_parseable_author(text: str) -> bool:
    """Return whether text parses to one valid author without ambiguity."""
    parsed = parse_author_prediction(text)
    return parsed["parsed_label"] is not None and parsed["parse_status"] != "invalid_multiple_authors"


def first_nonempty_line(text: str) -> str:
    """Return the first non-empty line from text."""
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return str(text).strip()


def infer_model_input_device(model) -> torch.device:
    """Infer the device to place input tensors on."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_prediction_row(
    row: dict[str, Any],
    parsed_prediction: dict[str, Any],
    generation: dict[str, Any],
    config: PromptingConfig,
) -> dict[str, Any]:
    """Build one predictions.csv row."""
    pred_label = parsed_prediction["parsed_label"]
    pred_author = parsed_prediction["parsed_author"]
    correct = is_valid_integer_label(pred_label) and int(pred_label) == int(row["true_label"])
    return {
        "sample_id": row["sample_id"],
        "text": row["text"],
        "true_label": row["true_label"],
        "true_author": row["true_author"],
        "true_label_name": row["true_author"],
        "pred_label": pred_label,
        "pred_label_name": pred_author,
        "predicted_author": pred_author,
        "raw_output": generation["raw_output"],
        "cleaned_output": generation["cleaned_output"],
        "parsed_status": parsed_prediction["parse_status"],
        "parser_notes": parsed_prediction["parser_notes"],
        "prompt_type": config.prompt_type,
        "model_name": config.model_name,
        "input_tokens": generation["input_tokens"],
        "output_tokens": generation["output_tokens"],
        "inference_time_sec": generation["inference_time_sec"],
        "correct": bool(correct),
    }


def write_predictions(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write prediction rows as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_runtime(config: PromptingConfig, num_samples: int, total_time: float, output_tokens: int) -> dict[str, Any]:
    """Build runtime metadata."""
    cuda_available = torch.cuda.is_available()
    return {
        "model_name": config.model_name,
        "prompt_type": config.prompt_type,
        "run_name": config.run_name,
        **run_metadata(config),
        "num_samples": int(num_samples),
        "total_time_sec": float(total_time),
        "avg_time_sec": float(total_time / num_samples) if num_samples else 0.0,
        "tokens_per_second": float(output_tokens / total_time) if total_time > 0 else 0.0,
        "cuda_available": bool(cuda_available),
        "peak_cuda_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if cuda_available else None,
        "peak_cuda_memory_reserved_mb": float(torch.cuda.max_memory_reserved() / 1024**2) if cuda_available else None,
        "use_4bit": bool(config.use_4bit),
        "max_new_tokens": int(config.max_new_tokens),
        "temperature": float(config.temperature),
        "do_sample": bool(config.do_sample),
        "top_p": float(config.top_p),
        "max_input_chars": int(config.max_input_chars),
        "seed": int(config.seed),
    }


def run_metadata(config: PromptingConfig) -> dict[str, Any]:
    """Return smoke/reportability metadata for output files."""
    smoke = is_smoke_test(config)
    return {
        "is_smoke_test": smoke,
        "hardware_note": hardware_note(config),
        "reportable_result": not smoke and is_official_decoder_model(config),
    }


def is_smoke_test(config: PromptingConfig) -> bool:
    """Return whether a config is a local smoke-test run."""
    return "tinyllama" in config.model_name.lower() or config.run_name.startswith("smoke_")


def is_official_decoder_model(config: PromptingConfig) -> bool:
    """Return whether a config points at a reportable Phase 4 decoder model."""
    lowered = config.model_name.lower()
    return (
        "mistralai/mistral-7b-instruct" in lowered
        or "meta-llama/meta-llama-3-8b-instruct" in lowered
        or "google/gemma-2-9b-it" in lowered
    )


def cuda_total_vram_gb() -> float | None:
    """Return total VRAM for the active CUDA device, if available."""
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(0)
    return float(properties.total_memory / 1024**3)


def hardware_note(config: PromptingConfig) -> str:
    """Describe hardware/reportability constraints for this run."""
    total_vram = cuda_total_vram_gb()
    if total_vram is None:
        device_note = "CUDA is not available; generation will use CPU unless model loading chooses another backend."
    else:
        device_name = torch.cuda.get_device_name(0)
        device_note = f"CUDA device: {device_name}; total VRAM: {total_vram:.2f} GB."
    if is_smoke_test(config):
        return f"{device_note} TinyLlama smoke-test output is for pipeline validation only and is not reportable Phase 4 evidence."
    if is_official_decoder_model(config):
        return f"{device_note} Official Phase 4 decoder config; use a larger GPU/Colab if local VRAM is insufficient."
    return device_note


def assert_local_vram_is_suitable(config: PromptingConfig) -> None:
    """Fail early for official 7B/9B configs on very small local GPUs."""
    total_vram = cuda_total_vram_gb()
    if total_vram is None:
        return
    if total_vram < 6.0 and is_official_decoder_model(config):
        raise RuntimeError(
            f"This GPU has only {total_vram:.2f} GB VRAM. 7B/9B Phase 4 models are not suitable "
            "for local real inference. Use smoke_tinyllama_* configs locally, or run official configs "
            "on a larger GPU/Colab."
        )


def accelerate_available() -> bool:
    """Return whether accelerate is importable for device_map='auto'."""
    return importlib.util.find_spec("accelerate") is not None


def clear_cuda_cache() -> None:
    """Clear CUDA cache when CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write JSON with stable indentation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def is_valid_integer_label(value: Any) -> bool:
    """Return whether a parser label can be compared to numeric true labels."""
    if value is None or value == "":
        return False
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    main()
