"""Load and summarize Phase 1 Hugging Face datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml
from datasets import DatasetDict, load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import dataset_dir  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase1" / "dataset_pipeline.yaml"


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def _load_dataset(name: str, cache_dir: Path | None) -> DatasetDict:
    kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    loaded = load_dataset(name, **kwargs)
    if not isinstance(loaded, DatasetDict):
        return DatasetDict({"train": loaded})
    return loaded


def summarize_dataset(alias: str, dataset: DatasetDict) -> None:
    print(f"\n{alias}")
    print("-" * len(alias))
    for split_name, split in dataset.items():
        print(f"split={split_name} rows={len(split)} columns={list(split.column_names)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--no_cache_dir",
        action="store_true",
        help="Use the default Hugging Face cache instead of dataset_dir('cache').",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir = None if args.no_cache_dir else dataset_dir("cache")

    periad_name = config["datasets"]["periad"]["name"]
    veaa_name = config["datasets"]["veaa"]["name"]

    periad = _load_dataset(periad_name, cache_dir)
    veaa = _load_dataset(veaa_name, cache_dir)

    summarize_dataset(periad_name, periad)
    summarize_dataset(veaa_name, veaa)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
