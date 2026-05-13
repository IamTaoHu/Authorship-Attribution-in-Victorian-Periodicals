"""Prepare project Hugging Face datasets into the external processed layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml
from datasets import DatasetDict, load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.prepare_phase1_dataset import (  # noqa: E402
    AUTHOR_CANDIDATES,
    TEXT_CANDIDATES,
    load_config,
    standardize_split,
    summarize_dataset_dict,
    write_frame,
)
from src.data.validate_periad import detect_column, validate_periad  # noqa: E402
from src.utils.paths import dataset_dir  # noqa: E402


DEFAULT_DATASETS_CONFIG = REPO_ROOT / "configs" / "datasets.yaml"
DEFAULT_PHASE1_CONFIG = REPO_ROOT / "configs" / "phase1" / "dataset_pipeline.yaml"
PERIAD_REQUIRED = ("train.csv", "test.csv", "label_map.json")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def configured_dataset_id(datasets_config: dict[str, Any], dataset_key: str) -> str:
    datasets = datasets_config.get("datasets")
    if not isinstance(datasets, dict) or dataset_key not in datasets:
        raise ValueError(f"Missing datasets.{dataset_key} in {DEFAULT_DATASETS_CONFIG}")
    dataset_id = str(datasets[dataset_key].get("name", "")).strip()
    if not dataset_id:
        raise ValueError(f"Missing Hugging Face dataset id at datasets.{dataset_key}.name")
    return dataset_id


def load_dataset_dict(dataset_id: str) -> DatasetDict:
    loaded = load_dataset(dataset_id, cache_dir=str(dataset_dir("cache")))
    if isinstance(loaded, DatasetDict):
        return loaded
    return DatasetDict({"train": loaded})


def remove_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)


def existing_periad_status(output_dir: Path) -> tuple[bool, list[str]]:
    missing = [name for name in PERIAD_REQUIRED if not (output_dir / name).exists()]
    return not missing, missing


def prepare_periad(
    datasets_config: dict[str, Any],
    phase1_config: dict[str, Any],
    overwrite: bool,
) -> None:
    dataset_id = configured_dataset_id(datasets_config, "periad")
    periad_config = phase1_config["datasets"]["periad"]
    periad_config["name"] = dataset_id

    configured_authors = datasets_config["datasets"]["periad"].get("authors")
    canonical_authors = list(periad_config["canonical_authors"])
    if configured_authors and list(configured_authors) != canonical_authors:
        raise ValueError("PERIAD author order differs between configs/datasets.yaml and Phase 1 config.")

    output_dir = dataset_dir("processed", "periad")
    complete, missing = existing_periad_status(output_dir)
    if complete and not overwrite:
        print(f"PERIAD already prepared at {output_dir}; use --overwrite to regenerate.")
        return
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"PERIAD output directory exists but is incomplete; missing {missing}. "
            "Use --overwrite to regenerate it explicitly."
        )
    remove_output_dir(output_dir, overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    periad = load_dataset_dict(dataset_id)
    validation_report = validate_periad(periad, phase1_config)
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

    label_map = {author: index for index, author in enumerate(canonical_authors)}
    train_frame = standardize_split(periad, train_split, text_column, label_column, label_map, canonical_authors)
    test_frame = standardize_split(periad, test_split, text_column, label_column, label_map, canonical_authors)
    write_frame(train_frame, output_dir / "train.csv", output_dir / "train.jsonl")
    write_frame(test_frame, output_dir / "test.csv", output_dir / "test.jsonl")
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    dataset_card = {
        "name": dataset_id,
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
        "training": "Phase 1 dataset preparation does not train or fine-tune any model.",
    }
    (output_dir / "dataset_card.json").write_text(
        json.dumps(dataset_card, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'label_map.json'}")
    print(f"Wrote {output_dir / 'dataset_card.json'}")


def prepare_veaa(datasets_config: dict[str, Any], overwrite: bool) -> None:
    dataset_id = configured_dataset_id(datasets_config, "veaa")
    output_dir = dataset_dir("processed", "veaa")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        print(f"VEAA already has processed outputs at {output_dir}; use --overwrite to regenerate.")
        return
    remove_output_dir(output_dir, overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        veaa = load_dataset_dict(dataset_id)
        summary = summarize_dataset_dict(dataset_id, veaa)
        summary["status"] = "loaded"
        for split_name, split in veaa.items():
            frame = split.to_pandas()
            csv_path = output_dir / f"{split_name}.csv"
            jsonl_path = output_dir / f"{split_name}.jsonl"
            frame.to_csv(csv_path, index=False)
            frame.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
            print(f"Wrote {csv_path}")
            print(f"Wrote {jsonl_path}")
    except Exception as exc:
        message = str(exc).replace("\n", " ")
        print(f"WARNING: Could not load VEAA dataset: {message}")
        summary = {
            "name": dataset_id,
            "status": "unavailable",
            "warning": message,
            "splits": {},
        }
    (output_dir / "metadata_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'metadata_summary.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("periad", "veaa", "all"), required=True)
    parser.add_argument("--datasets_config", default=str(DEFAULT_DATASETS_CONFIG))
    parser.add_argument("--phase1_config", default=str(DEFAULT_PHASE1_CONFIG))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets_config = load_yaml(args.datasets_config)
    phase1_config = load_config(args.phase1_config)

    if args.dataset in ("periad", "all"):
        prepare_periad(datasets_config, phase1_config, overwrite=args.overwrite)
    if args.dataset in ("veaa", "all"):
        prepare_veaa(datasets_config, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
