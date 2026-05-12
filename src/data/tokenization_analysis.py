"""Compute tokenizer length statistics for Phase 1 PERIAD texts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import dataset_dir, phase_artifact_dir  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase1" / "dataset_pipeline.yaml"


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def read_texts(max_samples: int | None) -> list[str]:
    periad_dir = dataset_dir("processed", "periad")
    train = pd.read_csv(periad_dir / "train.csv", usecols=["text"])
    test = pd.read_csv(periad_dir / "test.csv", usecols=["text"])
    texts = pd.concat([train, test], ignore_index=True)["text"].fillna("").astype(str)
    if max_samples is not None:
        texts = texts.head(max_samples)
    return texts.tolist()


def tokenizer_stats(tokenizer_name: str, texts: list[str], max_lengths: list[int]) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=str(dataset_dir("cache")))
    lengths = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=True, truncation=False)
        lengths.append(len(encoded["input_ids"]))

    array = np.asarray(lengths, dtype=float)
    row: dict[str, Any] = {
        "tokenizer": tokenizer_name,
        "samples": len(texts),
        "avg_tokens": float(np.mean(array)) if len(array) else 0.0,
        "median_tokens": float(np.median(array)) if len(array) else 0.0,
        "p90_tokens": float(np.percentile(array, 90)) if len(array) else 0.0,
        "p95_tokens": float(np.percentile(array, 95)) if len(array) else 0.0,
        "max_tokens": int(np.max(array)) if len(array) else 0,
        "status": "ok",
        "warning": "",
    }
    for max_length in max_lengths:
        row[f"truncation_rate_at_{max_length}"] = float(np.mean(array > max_length)) if len(array) else 0.0
    return row


def failed_row(tokenizer_name: str, max_lengths: list[int], message: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "tokenizer": tokenizer_name,
        "samples": 0,
        "avg_tokens": None,
        "median_tokens": None,
        "p90_tokens": None,
        "p95_tokens": None,
        "max_tokens": None,
        "status": "failed",
        "warning": message,
    }
    for max_length in max_lengths:
        row[f"truncation_rate_at_{max_length}"] = None
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--skip_missing_tokenizers", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    tokenization = config["tokenization"]
    tokenizer_names = list(tokenization["tokenizers"])
    max_lengths = [int(value) for value in tokenization["max_lengths"]]
    texts = read_texts(args.max_samples)

    rows = []
    failures = 0
    for tokenizer_name in tokenizer_names:
        try:
            rows.append(tokenizer_stats(tokenizer_name, texts, max_lengths))
            print(f"Computed tokenization stats for {tokenizer_name}")
        except Exception as exc:
            failures += 1
            message = str(exc).replace("\n", " ")[:500]
            if not args.skip_missing_tokenizers:
                raise
            print(f"WARNING: Skipping tokenizer {tokenizer_name}: {message}")
            rows.append(failed_row(tokenizer_name, max_lengths, message))

    table_dir = phase_artifact_dir("phase1", "tables")
    frame = pd.DataFrame(rows)
    csv_path = table_dir / "tokenization_stats.csv"
    json_path = table_dir / "tokenization_stats.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    return 1 if failures and not args.skip_missing_tokenizers else 0


if __name__ == "__main__":
    raise SystemExit(main())
