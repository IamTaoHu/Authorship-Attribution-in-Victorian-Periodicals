"""Generate Phase 1 dataset statistics tables and report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml


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


def read_periad() -> pd.DataFrame:
    periad_dir = dataset_dir("processed", "periad")
    train = pd.read_csv(periad_dir / "train.csv")
    test = pd.read_csv(periad_dir / "test.csv")
    return pd.concat([train, test], ignore_index=True)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    display = display.round(4)
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(lines)


def write_report(
    split_summary: pd.DataFrame,
    author_distribution: pd.DataFrame,
    length_statistics: pd.DataFrame,
    veaa_summary: dict[str, Any] | None,
) -> None:
    report_dir = phase_artifact_dir("phase1", "reports")
    report_path = report_dir / "phase1_dataset_report.md"
    lines = [
        "# Phase 1 Dataset Report",
        "",
        "Phase 1 is a dataset-only pipeline. It does not train or fine-tune any model.",
        "",
        "## PERIAD Split Summary",
        "",
        markdown_table(split_summary),
        "",
        "## PERIAD Author Distribution",
        "",
        markdown_table(author_distribution),
        "",
        "## PERIAD Length Statistics",
        "",
        markdown_table(length_statistics),
        "",
        "## ModifiedVEAA Summary",
        "",
    ]
    if veaa_summary:
        for split_name, split in veaa_summary.get("splits", {}).items():
            lines.append(f"- {split_name}: {split.get('rows')} rows; columns={split.get('columns')}")
    else:
        lines.append("- VEAA metadata summary not found.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.parse_args()

    frame = read_periad()
    table_dir = phase_artifact_dir("phase1", "tables")

    split_summary = (
        frame.groupby("split", dropna=False)
        .agg(rows=("sample_id", "count"), authors=("author", "nunique"))
        .reset_index()
    )
    split_summary["percent"] = split_summary["rows"] / split_summary["rows"].sum() * 100

    author_distribution = (
        frame.groupby(["split", "author"], dropna=False)
        .agg(rows=("sample_id", "count"))
        .reset_index()
    )
    totals = author_distribution.groupby("split")["rows"].transform("sum")
    author_distribution["percent_within_split"] = author_distribution["rows"] / totals * 100

    length_statistics = (
        frame.groupby("split", dropna=False)
        .agg(
            rows=("sample_id", "count"),
            chars_mean=("text_length_chars", "mean"),
            chars_median=("text_length_chars", "median"),
            chars_min=("text_length_chars", "min"),
            chars_max=("text_length_chars", "max"),
            words_mean=("text_length_words", "mean"),
            words_median=("text_length_words", "median"),
            words_min=("text_length_words", "min"),
            words_max=("text_length_words", "max"),
        )
        .reset_index()
    )

    veaa_path = dataset_dir("processed", "veaa") / "metadata_summary.json"
    veaa_summary = json.loads(veaa_path.read_text(encoding="utf-8")) if veaa_path.exists() else None
    veaa_table = pd.DataFrame(
        [
            {
                "dataset": veaa_summary.get("name") if veaa_summary else "NicholasSynovic/ModifiedVEAA",
                "split": split_name,
                "rows": split.get("rows"),
                "columns": "|".join(split.get("columns", [])),
            }
            for split_name, split in (veaa_summary or {}).get("splits", {}).items()
        ]
    )

    outputs = {
        "periad_split_summary.csv": split_summary,
        "periad_author_distribution.csv": author_distribution,
        "periad_length_statistics.csv": length_statistics,
        "veaa_summary.csv": veaa_table,
    }
    for filename, table in outputs.items():
        path = table_dir / filename
        table.to_csv(path, index=False)
        print(f"Wrote {path}")

    write_report(split_summary, author_distribution, length_statistics, veaa_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
