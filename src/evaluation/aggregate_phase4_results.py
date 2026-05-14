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
FINAL_RUN_NAMES = (
    "mistral_zero_shot",
    "mistral_few_shot",
    "llama3_zero_shot",
    "llama3_few_shot",
    "gemma2_zero_shot",
    "gemma2_few_shot",
)
FINAL_N_SAMPLES = 3549


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


def is_diagnostic_path(run_dir: Path, runs_dir: Path) -> bool:
    try:
        relative_parts = run_dir.relative_to(runs_dir).parts
    except ValueError:
        relative_parts = run_dir.parts
    lowered = {part.lower() for part in relative_parts}
    return bool(lowered.intersection({"diagnostic", "test"}))


def include_final_row(row: dict[str, Any], run_dir: Path, runs_dir: Path) -> bool:
    run_name = str(row.get("run_name", run_dir.name))
    if is_diagnostic_path(run_dir, runs_dir):
        print(f"WARNING: Excluding diagnostic run directory from final report: {run_dir}")
        return False
    if run_name not in FINAL_RUN_NAMES:
        print(f"WARNING: Excluding non-final Phase 4 run from final report: {run_name}")
        return False
    if bool(row.get("valid_run")) is not True:
        print(f"WARNING: Excluding invalid final run {run_name}; valid_run={row.get('valid_run')!r}")
        return False
    n_samples = row.get("n_samples")
    if n_samples is None or n_samples <= 0:
        print(f"WARNING: Excluding final run {run_name}; n_samples={n_samples!r}")
        return False
    if str(row.get("invalid_reason") or "").strip():
        print(f"WARNING: Excluding final run {run_name}; invalid_reason is not empty")
        return False
    return True


def warn_final_run_health(results: pd.DataFrame, discovered: set[str]) -> None:
    for run_name in FINAL_RUN_NAMES:
        if run_name not in discovered:
            print(f"WARNING: Expected final run is missing: {run_name}")
            continue
        row = results.loc[results["run_name"].astype(str) == run_name]
        if row.empty:
            print(f"WARNING: Expected final run was not included in final-only outputs: {run_name}")
            continue
        n_samples = int(row.iloc[0].get("n_samples", 0))
        if n_samples != FINAL_N_SAMPLES:
            print(f"WARNING: Expected final run {run_name} has n_samples={n_samples}, expected {FINAL_N_SAMPLES}")
        if bool(row.iloc[0].get("valid_run")) is not True:
            print(f"WARNING: Expected final run {run_name} has valid_run={row.iloc[0].get('valid_run')!r}")


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


def write_report(results: pd.DataFrame, per_author: pd.DataFrame, path: Path, *, final_only: bool = False) -> None:
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
    ]
    if final_only:
        lines.extend(["## Reporting Scope", "", "Final-only report excludes diagnostic TinyLlama smoke runs.", ""])
    lines.extend(["## Decoder Runs", "", markdown_table(results), ""])
    if not per_author.empty:
        lines.extend(["## Per-Author F1", "", markdown_table(per_author[["run_name", "label", "f1-score", "support"]]), ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-only", action="store_true", help="Report only the six final Phase 4 decoder benchmark runs.")
    args = parser.parse_args()
    runs_dir = phase_artifact_dir("phase4", "runs")
    tables_dir = phase_artifact_dir("phase4", "tables")
    reports_dir = phase_artifact_dir("phase4", "reports")
    rows = []
    discovered_final_runs: set[str] = set()
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        if args.final_only and is_diagnostic_path(run_dir, runs_dir):
            print(f"WARNING: Excluding diagnostic run directory from final report: {run_dir}")
            continue
        row = build_row(run_dir)
        if row is not None:
            run_name = str(row.get("run_name", run_dir.name))
            if run_name in FINAL_RUN_NAMES:
                discovered_final_runs.add(run_name)
            if args.final_only and not include_final_row(row, run_dir, runs_dir):
                continue
            rows.append(row)
    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["macro_f1", "accuracy"], ascending=False, na_position="last")
    if args.final_only:
        warn_final_run_health(results, discovered_final_runs)
    results_csv = tables_dir / "decoder_prompting_results.csv"
    results_md = tables_dir / "decoder_prompting_results.md"
    results.to_csv(results_csv, index=False)
    results_md.write_text(markdown_table(results) + "\n", encoding="utf-8")
    print(f"Wrote {results_csv}")
    print(f"Wrote {results_md}")
    per_author = read_per_author_reports(results, runs_dir)
    if args.final_only and len(per_author) != len(FINAL_RUN_NAMES) * len(CANONICAL_AUTHORS):
        print(
            "WARNING: final-only per-author table has "
            f"{len(per_author)} rows; expected {len(FINAL_RUN_NAMES) * len(CANONICAL_AUTHORS)}"
        )
    per_author_path = tables_dir / "phase4_per_author_f1.csv"
    per_author.to_csv(per_author_path, index=False)
    print(f"Wrote {per_author_path}")
    write_report(results, per_author, reports_dir / "phase4_decoder_prompting_report.md", final_only=args.final_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
