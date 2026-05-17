"""Aggregate completed Phase 5 full QLoRA run outputs without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FULL_RUNS = ("mistral_qlora", "llama3_qlora", "gemma2_qlora")
RESULT_COLUMNS = (
    "run_name",
    "model_name",
    "n_samples",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "invalid_output_rate",
    "avg_output_length_chars",
    "avg_inference_time_sec",
    "total_inference_time_sec",
    "predictions_path",
    "metrics_path",
)
PER_AUTHOR_COLUMNS = ("run_name", "author", "precision", "recall", "f1", "support")
AUTHOR_COLUMNS = ("true_author", "predicted_author")


def resolve_phase5_dir(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return ""


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def completed_run_dirs(phase5_dir: Path) -> list[Path]:
    run_root = phase5_dir / "runs"
    dirs: list[Path] = []
    for run_name in FULL_RUNS:
        run_dir = run_root / run_name
        if (run_dir / "predictions.csv").exists() and (run_dir / "metrics.json").exists():
            dirs.append(run_dir)
    return dirs


def build_results_row(run_dir: Path, phase5_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.csv"
    metrics = load_json(metrics_path)
    avg_output_length = metric_value(metrics, "avg_output_length_chars")
    if avg_output_length == "":
        predictions = pd.read_csv(predictions_path, usecols=lambda column: column in {"raw_output", "generated_length"})
        if "raw_output" in predictions:
            avg_output_length = predictions["raw_output"].fillna("").astype(str).str.len().mean()
        elif "generated_length" in predictions:
            avg_output_length = pd.to_numeric(predictions["generated_length"], errors="coerce").mean()
    return {
        "run_name": metric_value(metrics, "run_name") or run_dir.name,
        "model_name": metric_value(metrics, "model_name"),
        "n_samples": metric_value(metrics, "n_samples"),
        "accuracy": metric_value(metrics, "accuracy"),
        "macro_f1": metric_value(metrics, "macro_f1"),
        "weighted_f1": metric_value(metrics, "weighted_f1"),
        "invalid_output_rate": metric_value(metrics, "invalid_output_rate"),
        "avg_output_length_chars": avg_output_length,
        "avg_inference_time_sec": metric_value(metrics, "avg_inference_time_sec", "average_inference_time_sec"),
        "total_inference_time_sec": metric_value(metrics, "total_inference_time_sec"),
        "predictions_path": relative_or_absolute(predictions_path, phase5_dir),
        "metrics_path": relative_or_absolute(metrics_path, phase5_dir),
    }


def per_author_from_predictions(run_dir: Path) -> pd.DataFrame:
    predictions = pd.read_csv(run_dir / "predictions.csv", usecols=lambda column: column in AUTHOR_COLUMNS)
    missing = [column for column in AUTHOR_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(f"{run_dir / 'predictions.csv'} missing columns: {', '.join(missing)}")
    labels = sorted(set(predictions["true_author"].dropna().astype(str)))
    rows: list[dict[str, Any]] = []
    for author in labels:
        true_author = predictions["true_author"].astype(str) == author
        predicted_author = predictions["predicted_author"].astype(str) == author
        tp = int((true_author & predicted_author).sum())
        fp = int((~true_author & predicted_author).sum())
        fn = int((true_author & ~predicted_author).sum())
        support = int(true_author.sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "run_name": run_dir.name,
                "author": author,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return pd.DataFrame(rows, columns=PER_AUTHOR_COLUMNS)


def write_report(phase5_dir: Path, results: pd.DataFrame, per_author_path: Path) -> Path:
    report_path = phase5_dir / "reports" / "phase5_qlora_decoder_finetuning_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 5 QLoRA Decoder Fine-Tuning Report",
        "",
        "Phase 5 full QLoRA runs are now completed for Mistral, Llama 3, and Gemma 2.",
        "",
        "Diagnostic artifacts may be absent in Colab Drive full-run environments; the local diagnostic run previously validated the code path.",
        "",
        "## Completed Models",
        "",
    ]
    for _, row in results.iterrows():
        lines.append(f"- `{row['run_name']}`: `{row['model_name']}`")
    lines.extend(
        [
            "",
            "## Output Paths",
            "",
            f"- Aggregate CSV: `{relative_or_absolute(phase5_dir / 'tables' / 'decoder_qlora_results.csv', phase5_dir)}`",
            f"- Aggregate Markdown: `{relative_or_absolute(phase5_dir / 'tables' / 'decoder_qlora_results.md', phase5_dir)}`",
            f"- Per-author F1 CSV: `{relative_or_absolute(per_author_path, phase5_dir)}`",
            "",
            "## Aggregate Metrics",
            "",
            results.to_markdown(index=False),
            "",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    return report_path


def aggregate_phase5_results(phase5_dir: str | Path) -> dict[str, Path]:
    phase5_dir = resolve_phase5_dir(phase5_dir)
    run_dirs = completed_run_dirs(phase5_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No completed full Phase 5 run directories found under {phase5_dir / 'runs'}")
    tables_dir = phase5_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame([build_results_row(run_dir, phase5_dir) for run_dir in run_dirs], columns=RESULT_COLUMNS)
    results_csv = tables_dir / "decoder_qlora_results.csv"
    results_md = tables_dir / "decoder_qlora_results.md"
    results.to_csv(results_csv, index=False)
    results_md.write_text(results.to_markdown(index=False) + "\n", encoding="utf-8")
    print(f"Wrote {results_csv}")
    print(f"Wrote {results_md}")

    per_author = pd.concat([per_author_from_predictions(run_dir) for run_dir in run_dirs], ignore_index=True)
    per_author_csv = tables_dir / "phase5_per_author_f1.csv"
    per_author.to_csv(per_author_csv, index=False)
    print(f"Wrote {per_author_csv}")
    report = write_report(phase5_dir, results, per_author_csv)
    return {
        "results_csv": results_csv,
        "results_md": results_md,
        "per_author_csv": per_author_csv,
        "report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5_dir", default="outputs/phase5")
    args = parser.parse_args()
    aggregate_phase5_results(args.phase5_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
