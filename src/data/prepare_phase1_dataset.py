"""Prepare standardized Phase 1 datasets outside the repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml
from datasets import DatasetDict, load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.validate_periad import detect_column, validate_periad, values_as_authors, write_reports  # noqa: E402
from src.utils.paths import dataset_dir, phase_artifact_dir  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase1" / "dataset_pipeline.yaml"
TEXT_CANDIDATES = ("text", "article", "content", "body", "document", "essay")
AUTHOR_CANDIDATES = ("author", "authors", "label", "labels", "target", "class")


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def load_dataset_dict(name: str) -> DatasetDict:
    loaded = load_dataset(name, cache_dir=str(dataset_dir("cache")))
    if not isinstance(loaded, DatasetDict):
        return DatasetDict({"train": loaded})
    return loaded


def _label_id(author: str, label_map: dict[str, int]) -> int:
    if author not in label_map:
        raise ValueError(f"Author is not in canonical label map: {author}")
    return label_map[author]


def standardize_split(
    dataset: DatasetDict,
    split_name: str,
    text_column: str,
    label_column: str,
    label_map: dict[str, int],
    canonical_authors: list[str],
) -> pd.DataFrame:
    frame = dataset[split_name].to_pandas()
    feature = dataset[split_name].features.get(label_column)
    authors = values_as_authors(frame, label_column, feature, canonical_authors).astype("string")
    text = frame[text_column].astype("string").fillna("")
    standardized = pd.DataFrame(
        {
            "sample_id": [f"periad_{split_name}_{index:05d}" for index in range(len(frame))],
            "split": split_name,
            "text": text,
            "author": authors,
        }
    )
    standardized["label"] = standardized["author"].map(lambda author: _label_id(str(author), label_map))
    standardized["text_length_chars"] = standardized["text"].str.len().astype(int)
    standardized["text_length_words"] = standardized["text"].str.split().map(len).astype(int)
    return standardized[
        [
            "sample_id",
            "split",
            "text",
            "label",
            "author",
            "text_length_chars",
            "text_length_words",
        ]
    ]


def write_frame(frame: pd.DataFrame, csv_path: Path, jsonl_path: Path) -> None:
    frame.to_csv(csv_path, index=False)
    frame.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"Wrote {csv_path}")
    print(f"Wrote {jsonl_path}")


def summarize_dataset_dict(name: str, dataset: DatasetDict) -> dict[str, Any]:
    return {
        "name": name,
        "splits": {
            split_name: {
                "rows": len(split),
                "columns": list(split.column_names),
                "features": {key: str(value) for key, value in split.features.items()},
            }
            for split_name, split in dataset.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = load_config(args.config)
    periad_config = config["datasets"]["periad"]
    periad = load_dataset_dict(periad_config["name"])

    validation_report = validate_periad(periad, config)
    report_dir = phase_artifact_dir("phase1", "reports")
    write_reports(validation_report)
    if validation_report["status"] == "failed":
        raise ValueError(f"PERIAD validation failed: {validation_report['errors']}")
    for warning in validation_report.get("warnings", []):
        print(f"WARNING: {warning}")

    train_split = periad_config["train_split"]
    test_split = periad_config["test_split"]
    train_columns = list(periad[train_split].column_names)
    text_column = detect_column(train_columns, TEXT_CANDIDATES)
    label_column = detect_column(train_columns, AUTHOR_CANDIDATES)
    if text_column is None or label_column is None:
        raise ValueError(f"Could not detect required PERIAD columns from {train_columns}")

    canonical_authors = list(periad_config["canonical_authors"])
    label_map = {author: index for index, author in enumerate(canonical_authors)}
    output_dir = dataset_dir("processed", "periad")
    train_frame = standardize_split(periad, train_split, text_column, label_column, label_map, canonical_authors)
    test_frame = standardize_split(periad, test_split, text_column, label_column, label_map, canonical_authors)
    write_frame(train_frame, output_dir / "train.csv", output_dir / "train.jsonl")
    write_frame(test_frame, output_dir / "test.csv", output_dir / "test.jsonl")

    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    dataset_card = {
        "name": periad_config["name"],
        "task": "authorship_attribution",
        "source": "huggingface",
        "splits": {
            train_split: len(train_frame),
            test_split: len(test_frame),
            "total": len(train_frame) + len(test_frame),
        },
        "authors": canonical_authors,
        "standardized_columns": list(train_frame.columns),
        "source_columns": {"text": text_column, "label_or_author": label_column},
        "validation_status": validation_report["status"],
        "validation_warnings": validation_report.get("warnings", []),
        "training": "Phase 1 does not train or fine-tune any model.",
    }
    (output_dir / "dataset_card.json").write_text(
        json.dumps(dataset_card, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'label_map.json'}")
    print(f"Wrote {output_dir / 'dataset_card.json'}")

    veaa_dir = dataset_dir("processed", "veaa")
    veaa_name = config["datasets"]["veaa"]["name"]
    try:
        veaa = load_dataset_dict(veaa_name)
        veaa_summary = summarize_dataset_dict(veaa_name, veaa)
        veaa_summary["status"] = "loaded"
    except Exception as exc:
        message = str(exc).replace("\n", " ")
        print(f"WARNING: Could not load VEAA dataset metadata: {message}")
        veaa_summary = {
            "name": veaa_name,
            "status": "unavailable",
            "warning": message,
            "splits": {},
        }
    (veaa_dir / "metadata_summary.json").write_text(
        json.dumps(veaa_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {veaa_dir / 'metadata_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
