"""Aggregate completed Phase 2 encoder baseline runs."""

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

from src.utils.paths import phase_artifact_dir  # noqa: E402


REQUIRED_RUN_FILES = (
    "predictions.csv",
    "eval_metrics.json",
    "classification_report.csv",
    "confusion_matrix.csv",
    "run_config_resolved.yaml",
)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No completed Phase 2 runs found._"
    display = frame.copy()
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].round(4)
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(lines)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_row(run_dir: Path) -> dict[str, Any] | None:
    missing = [filename for filename in REQUIRED_RUN_FILES if not (run_dir / filename).exists()]
    if missing:
        print(f"WARNING: Skipping incomplete run {run_dir.name}; missing {', '.join(missing)}")
        return None
    try:
        metrics = read_json(run_dir / "eval_metrics.json")
        runtime_path = run_dir / "runtime.json"
        runtime = read_json(runtime_path) if runtime_path.exists() else {}
        resolved = yaml.safe_load((run_dir / "run_config_resolved.yaml").read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"WARNING: Skipping malformed run {run_dir.name}: {exc}")
        return None
    return {
        "experiment_name": resolved.get("experiment_name", run_dir.name),
        "model_name": resolved.get("model_name"),
        "accuracy": metrics.get("accuracy", metrics.get("eval_accuracy")),
        "macro_f1": metrics.get("macro_f1", metrics.get("eval_macro_f1")),
        "weighted_f1": metrics.get("weighted_f1", metrics.get("eval_weighted_f1")),
        "precision_macro": metrics.get("precision_macro", metrics.get("eval_precision_macro")),
        "recall_macro": metrics.get("recall_macro", metrics.get("eval_recall_macro")),
        "train_runtime": runtime.get("train_runtime"),
        "eval_runtime": runtime.get("eval_runtime", metrics.get("eval_runtime")),
        "checkpoint_dir": resolved.get("checkpoint_dir"),
        "predictions_path": str(run_dir / "predictions.csv"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    runs_dir = phase_artifact_dir("phase2", "runs")
    table_dir = phase_artifact_dir("phase2", "tables")
    report_dir = phase_artifact_dir("phase2", "reports")
    rows = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        row = build_row(run_dir)
        if row is not None:
            rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["macro_f1", "accuracy"], ascending=False, na_position="last")
    csv_path = table_dir / "encoder_results.csv"
    md_path = table_dir / "encoder_results.md"
    report_path = report_dir / "phase2_encoder_baseline_report.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(frame) + "\n", encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Phase 2 Encoder Baseline Report",
                "",
                "This report aggregates completed Phase 2 encoder baseline runs.",
                "Full benchmark claims should only be made after Colab training is complete.",
                "",
                markdown_table(frame),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
