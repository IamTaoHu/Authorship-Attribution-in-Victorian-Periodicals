"""Aggregate completed Phase 4 decoder prompting runs."""

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

from src.prompting.phase4_parser import CANONICAL_AUTHORS  # noqa: E402
from src.utils.paths import phase_artifact_dir  # noqa: E402


REQUIRED_RUN_FILES = ("predictions.csv", "metrics.json", "classification_report.csv", "confusion_matrix.csv", "run_config_resolved.yaml")
PHASE2_ACCURACY = 0.9270
PHASE2_MACRO_F1 = 0.9191


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No completed Phase 4 runs found._"
    display = frame.copy()
    numeric = display.select_dtypes(include="number").columns
    display[numeric] = display[numeric].round(4)
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
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
        metrics = read_json(run_dir / "metrics.json")
        resolved = yaml.safe_load((run_dir / "run_config_resolved.yaml").read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"WARNING: Skipping malformed run {run_dir.name}: {exc}")
        return None
    return {
        "run_name": metrics.get("run_name", run_dir.name),
        "model_name": metrics.get("model_name", resolved.get("model_name")),
        "prompt_type": metrics.get("prompt_type", resolved.get("prompt_type")),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "invalid_output_rate": metrics.get("invalid_output_rate"),
        "n_samples": metrics.get("n_samples"),
        "n_valid_predictions": metrics.get("n_valid_predictions"),
        "avg_input_tokens": metrics.get("avg_input_tokens"),
        "avg_output_tokens": metrics.get("avg_output_tokens"),
        "avg_inference_time_sec": metrics.get("avg_inference_time_sec"),
        "total_inference_time_sec": metrics.get("total_inference_time_sec"),
        "valid_run": metrics.get("valid_run", True),
        "invalid_reason": metrics.get("invalid_reason", ""),
        "accuracy_delta_vs_phase2_roberta_large": None if metrics.get("accuracy") is None else metrics.get("accuracy") - PHASE2_ACCURACY,
        "macro_f1_delta_vs_phase2_roberta_large": None if metrics.get("macro_f1") is None else metrics.get("macro_f1") - PHASE2_MACRO_F1,
        "predictions_path": str(run_dir / "predictions.csv"),
    }


def read_per_author_reports(results: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, result in results.iterrows():
        run_name = str(result["run_name"])
        report_path = runs_dir / run_name / "classification_report.csv"
        if not report_path.exists():
            continue
        report = pd.read_csv(report_path)
        if "label" not in report.columns:
            continue
        report = report.loc[report["label"].astype(str).isin(CANONICAL_AUTHORS)].copy()
        report["run_name"] = run_name
        report["model_name"] = result.get("model_name", "")
        report["prompt_type"] = result.get("prompt_type", "")
        rows.append(report)
    if not rows:
        return pd.DataFrame(columns=["run_name", "model_name", "prompt_type", "label", "precision", "recall", "f1-score", "support"])
    return pd.concat(rows, ignore_index=True)


def write_report(results: pd.DataFrame, per_author: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Phase 4 Decoder Prompting Report",
        "",
        "Phase 4 evaluates instruction-tuned decoder prompting baselines on the fixed PERIAD test split.",
        "",
        "## Phase 2 Reference",
        "",
        f"- RoBERTa-large accuracy: {PHASE2_ACCURACY:.4f}",
        f"- RoBERTa-large macro F1: {PHASE2_MACRO_F1:.4f}",
        "",
        "## Decoder Runs",
        "",
        markdown_table(results),
        "",
    ]
    if not per_author.empty:
        lines.extend(["## Per-Author F1", "", markdown_table(per_author[["run_name", "label", "f1-score", "support"]]), ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    runs_dir = phase_artifact_dir("phase4", "runs")
    tables_dir = phase_artifact_dir("phase4", "tables")
    reports_dir = phase_artifact_dir("phase4", "reports")
    rows = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        row = build_row(run_dir)
        if row is not None:
            rows.append(row)
    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["macro_f1", "accuracy"], ascending=False, na_position="last")
    results_csv = tables_dir / "decoder_prompting_results.csv"
    results_md = tables_dir / "decoder_prompting_results.md"
    results.to_csv(results_csv, index=False)
    results_md.write_text(markdown_table(results) + "\n", encoding="utf-8")
    print(f"Wrote {results_csv}")
    print(f"Wrote {results_md}")
    per_author = read_per_author_reports(results, runs_dir)
    per_author_path = tables_dir / "phase4_per_author_f1.csv"
    per_author.to_csv(per_author_path, index=False)
    print(f"Wrote {per_author_path}")
    write_report(results, per_author, reports_dir / "phase4_decoder_prompting_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
