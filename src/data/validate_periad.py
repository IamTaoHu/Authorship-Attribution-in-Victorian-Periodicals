"""Validate the PERIAD dataset used by Phase 1."""

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


def load_periad(config: dict[str, Any]) -> DatasetDict:
    cache_dir = dataset_dir("cache")
    loaded = load_dataset(config["datasets"]["periad"]["name"], cache_dir=str(cache_dir))
    if not isinstance(loaded, DatasetDict):
        raise ValueError("PERIAD must load as a DatasetDict with train/test splits.")
    return loaded


def detect_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for candidate in candidates:
        for column in columns:
            if candidate in column.lower():
                return column
    return None


def _label_names_from_feature(feature: Any) -> list[str] | None:
    names = getattr(feature, "names", None)
    if names:
        return [str(name) for name in names]
    feature = getattr(feature, "feature", None)
    names = getattr(feature, "names", None)
    if names:
        return [str(name) for name in names]
    return None


def values_as_authors(
    frame: pd.DataFrame,
    label_column: str,
    feature: Any,
    canonical_authors: list[str] | None = None,
) -> pd.Series:
    label_names = _label_names_from_feature(feature)
    values = frame[label_column]
    if label_names and pd.api.types.is_integer_dtype(values):
        return values.map(lambda value: label_names[int(value)] if pd.notna(value) and int(value) < len(label_names) else None)
    if canonical_authors and pd.api.types.is_numeric_dtype(values):
        unique_values = sorted(int(value) for value in values.dropna().unique())
        expected_values = list(range(len(canonical_authors)))
        if unique_values == expected_values:
            return values.map(lambda value: canonical_authors[int(value)] if pd.notna(value) else None)
    return values.astype("string")


def validate_periad(dataset: DatasetDict, config: dict[str, Any]) -> dict[str, Any]:
    periad_config = config["datasets"]["periad"]
    train_split = periad_config["train_split"]
    test_split = periad_config["test_split"]
    canonical_authors = list(periad_config["canonical_authors"])

    warnings: list[str] = []
    errors: list[str] = []
    split_counts = {split: len(dataset[split]) for split in dataset.keys()}

    for required_split in (train_split, test_split):
        if required_split not in dataset:
            errors.append(f"Missing required split: {required_split}")

    train_rows = split_counts.get(train_split, 0)
    test_rows = split_counts.get(test_split, 0)
    total_rows = train_rows + test_rows
    expected = {
        train_split: int(periad_config["expected_train_rows"]),
        test_split: int(periad_config["expected_test_rows"]),
        "total": int(periad_config["expected_total_rows"]),
    }
    if train_rows != expected[train_split]:
        warnings.append(f"Expected {expected[train_split]} train rows, found {train_rows}.")
    if test_rows != expected[test_split]:
        warnings.append(f"Expected {expected[test_split]} test rows, found {test_rows}.")
    if total_rows != expected["total"]:
        warnings.append(f"Expected {expected['total']} total rows, found {total_rows}.")

    if errors:
        return {
            "status": "failed",
            "errors": errors,
            "warnings": warnings,
            "split_counts": split_counts,
        }

    frames = []
    for split_name in (train_split, test_split):
        frame = dataset[split_name].to_pandas()
        frame["_split"] = split_name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    columns = [column for column in combined.columns if column != "_split"]

    text_column = detect_column(columns, TEXT_CANDIDATES)
    label_column = detect_column(columns, AUTHOR_CANDIDATES)
    if text_column is None:
        errors.append(f"Could not detect text column from columns: {columns}")
    if label_column is None:
        errors.append(f"Could not detect author/label column from columns: {columns}")
    if errors:
        return {
            "status": "failed",
            "errors": errors,
            "warnings": warnings,
            "split_counts": split_counts,
            "columns": columns,
        }

    feature = dataset[train_split].features.get(label_column)
    authors = values_as_authors(combined, label_column, feature, canonical_authors)
    observed_authors = sorted(author for author in authors.dropna().astype(str).unique())
    missing_authors = sorted(set(canonical_authors) - set(observed_authors))
    extra_authors = sorted(set(observed_authors) - set(canonical_authors))
    if missing_authors:
        warnings.append(f"Missing canonical authors: {missing_authors}")
    if extra_authors:
        warnings.append(f"Found non-canonical authors: {extra_authors}")
    if len(observed_authors) != len(canonical_authors):
        warnings.append(f"Expected {len(canonical_authors)} authors, found {len(observed_authors)}.")

    text = combined[text_column].astype("string")
    null_text = int(text.isna().sum())
    empty_text = int(text.fillna("").str.strip().eq("").sum())
    duplicate_rows = int(combined.duplicated(subset=[text_column, label_column]).sum())
    if null_text:
        warnings.append(f"Found {null_text} rows with null text.")
    if empty_text:
        warnings.append(f"Found {empty_text} rows with empty text.")
    if duplicate_rows:
        warnings.append(f"Found {duplicate_rows} duplicate text/label rows.")

    return {
        "status": "failed" if errors else "passed_with_warnings" if warnings else "passed",
        "errors": errors,
        "warnings": warnings,
        "split_counts": split_counts,
        "expected_counts": expected,
        "detected_columns": {"text": text_column, "label_or_author": label_column},
        "observed_authors": observed_authors,
        "canonical_authors": canonical_authors,
        "quality_checks": {
            "null_text_rows": null_text,
            "empty_text_rows": empty_text,
            "duplicate_text_label_rows": duplicate_rows,
        },
    }


def write_reports(report: dict[str, Any]) -> None:
    report_dir = phase_artifact_dir("phase1", "reports")
    json_path = report_dir / "periad_validation_report.json"
    md_path = report_dir / "periad_validation_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# PERIAD Validation Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Split Counts",
        "",
    ]
    for split, count in report.get("split_counts", {}).items():
        lines.append(f"- {split}: {count}")
    lines.extend(["", "## Detected Columns", ""])
    for key, value in report.get("detected_columns", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Authors", ""])
    for author in report.get("observed_authors", []):
        lines.append(f"- {author}")
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings", [])
    lines.extend([f"- {warning}" for warning in warnings] or ["- None"])
    lines.extend(["", "## Errors", ""])
    errors = report.get("errors", [])
    lines.extend([f"- {error}" for error in errors] or ["- None"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = load_config(args.config)
    report = validate_periad(load_periad(config), config)
    write_reports(report)
    for warning in report.get("warnings", []):
        print(f"WARNING: {warning}")
    for error in report.get("errors", []):
        print(f"ERROR: {error}")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
