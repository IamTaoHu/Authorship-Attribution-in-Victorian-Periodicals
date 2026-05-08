"""Aggregate Phase 3 encoder experiment outputs into comparison tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULT_COLUMNS = [
    "experiment_name",
    "model_name",
    "max_length",
    "use_lora",
    "lora_r",
    "accuracy",
    "macro_f1",
    "precision_macro",
    "recall_macro",
    "weighted_f1",
    "train_runtime",
    "gpu_max_allocated_gb",
    "trainable_params",
    "total_params",
    "trainable_param_pct",
]

SKIPPED_COLUMNS = [
    "experiment_name",
    "model_name",
    "metrics_path",
    "run_status",
    "macro_f1",
    "validation_loss",
    "skip_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase 3 encoder metrics.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/phase3"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = resolve_repo_path(args.outputs_dir)
    rows, skipped_rows = load_metric_rows(outputs_dir)
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(skipped_rows, columns=SKIPPED_COLUMNS).to_csv(
        tables_dir / "skipped_invalid_runs.csv",
        index=False,
    )

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=RESULT_COLUMNS)
    frame = normalize_columns(frame)
    frame = frame.sort_values("macro_f1", ascending=False, na_position="last")

    save_table(frame, tables_dir / "encoder_results.csv", tables_dir / "encoder_results.md")
    frame.to_csv(tables_dir / "final_encoder_results_table.csv", index=False)

    lora = frame[frame["use_lora"].fillna(False).astype(bool)].copy()
    lora.to_csv(tables_dir / "lora_rank_ablation_table.csv", index=False)

    modernbert = frame[frame["model_name"].fillna("").str.contains("ModernBERT", case=False, regex=False)].copy()
    modernbert.to_csv(tables_dir / "modernbert_context_ablation_table.csv", index=False)

    print(f"Wrote Phase 3 tables to {tables_dir}")


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_metric_rows(outputs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for metrics_path in outputs_dir.glob("*/metrics.json"):
        with metrics_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        skip_reason = invalid_run_reason(payload)
        if skip_reason is not None:
            skipped_rows.append(
                {
                    "experiment_name": payload.get("experiment_name") or metrics_path.parent.name,
                    "model_name": payload.get("model_name"),
                    "metrics_path": str(metrics_path),
                    "run_status": payload.get("run_status"),
                    "macro_f1": payload.get("macro_f1"),
                    "validation_loss": validation_loss(payload),
                    "skip_reason": skip_reason,
                }
            )
            continue
        row = {column: payload.get(column) for column in RESULT_COLUMNS}
        row["experiment_name"] = payload.get("experiment_name") or metrics_path.parent.name
        row["metrics_path"] = str(metrics_path)
        rows.append(row)
    return rows, skipped_rows


def invalid_run_reason(payload: dict[str, Any]) -> str | None:
    if payload.get("run_status") == "failed_nan":
        return "run_status_failed_nan"
    macro_f1 = payload.get("macro_f1")
    if is_nan_or_inf(macro_f1):
        return "macro_f1_nan_or_inf"
    loss = validation_loss(payload)
    if is_nan_or_inf(loss):
        return "validation_loss_nan_or_inf"
    return None


def validation_loss(payload: dict[str, Any]) -> Any:
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        return None
    return validation.get("validation_loss", validation.get("eval_loss"))


def is_nan_or_inf(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    numeric_columns = [
        "max_length",
        "lora_r",
        "accuracy",
        "macro_f1",
        "precision_macro",
        "recall_macro",
        "weighted_f1",
        "train_runtime",
        "gpu_max_allocated_gb",
        "trainable_params",
        "total_params",
        "trainable_param_pct",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[RESULT_COLUMNS]


def save_table(frame: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    frame.to_csv(csv_path, index=False)
    markdown = dataframe_to_markdown(frame) if not frame.empty else "| No Phase 3 metrics found |\n| --- |"
    md_path.write_text(markdown + "\n", encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = [format_markdown_value(row[column]) for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
