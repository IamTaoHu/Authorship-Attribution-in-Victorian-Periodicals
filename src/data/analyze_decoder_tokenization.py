"""Analyze decoder tokenization for Phase 5 instruction data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any
import warnings

import pandas as pd
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


DEFAULT_DATA_DIR = PROJECT_ROOT / "outputs" / "phase5" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase5" / "tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase 5 decoder tokenization.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train_file", type=Path, default=DEFAULT_DATA_DIR / "train_instruction.jsonl")
    parser.add_argument("--test_file", type=Path, default=DEFAULT_DATA_DIR / "test_instruction.jsonl")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_phase5_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    ensure_pad_token(tokenizer)
    rows = read_jsonl(resolve_repo_path(args.train_file)) + read_jsonl(resolve_repo_path(args.test_file))
    if not rows:
        raise ValueError("No instruction rows found for tokenization analysis.")

    stats = compute_stats(rows, tokenizer, int(config["max_length"]))
    stats["model_name"] = config["model_name"]
    stats["model_short_name"] = config["model_short_name"]
    stats["max_length"] = int(config["max_length"])
    stats["num_rows"] = len(rows)

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tokenization_stats_{config['model_short_name']}"
    pd.DataFrame([stats]).to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def load_phase5_config(path: Path) -> dict[str, Any]:
    config = load_config(str(resolve_repo_path(path)))
    required = {"model_name", "model_short_name", "max_length"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Phase 5 config missing required tokenization fields: {missing}")
    return config


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        warnings.warn(f"Instruction file not found and will be skipped: {path}", RuntimeWarning, stacklevel=2)
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compute_stats(rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> dict[str, Any]:
    lengths: list[int] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    truncation_count = 0
    response_too_long_count = 0
    for row in rows:
        prompt = str(row.get("prompt", ""))
        response = str(row.get("response", ""))
        prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False, truncation=False)["input_ids"]
        total_length = len(prompt_ids) + len(response_ids)
        lengths.append(total_length)
        prompt_lengths.append(len(prompt_ids))
        response_lengths.append(len(response_ids))
        if total_length > max_length:
            truncation_count += 1
        if len(response_ids) >= max_length:
            response_too_long_count += 1
    sorted_lengths = sorted(lengths)
    return {
        "avg_token_length": float(statistics.fmean(lengths)),
        "median_token_length": float(statistics.median(lengths)),
        "p90_token_length": percentile(sorted_lengths, 0.90),
        "p95_token_length": percentile(sorted_lengths, 0.95),
        "max_token_length": int(max(lengths)),
        "avg_prompt_tokens": float(statistics.fmean(prompt_lengths)),
        "avg_response_tokens": float(statistics.fmean(response_lengths)),
        "truncation_count": int(truncation_count),
        "truncation_rate": float(truncation_count / len(rows)),
        "response_too_long_count": int(response_too_long_count),
        "response_preserved": response_too_long_count == 0,
    }


def percentile(sorted_values: list[int], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    index = (len(sorted_values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


if __name__ == "__main__":
    main()
