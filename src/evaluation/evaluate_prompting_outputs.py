"""Aggregate Phase 4 decoder prompting predictions into metrics tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CANONICAL_AUTHORS = [
    "Leslie Stephen",
    "John Morley",
    "Eliza Lynn Linton",
    "George Henry Lewes",
    "Anne Mozley",
    "James Fitzjames Stephen",
]
INVALID_LABEL = "__INVALID__"
PREDICTION_LABELS = CANONICAL_AUTHORS + [INVALID_LABEL]
REQUIRED_COLUMNS = {
    "sample_id",
    "true_author",
    "predicted_author",
    "raw_output",
    "parsed_status",
    "prompt_type",
    "model_name",
}
VALID_PROMPT_TYPES = {"zero_shot", "cot_zero_shot", "few_shot"}
DECODER_RESULT_COLUMNS = [
    "run_name",
    "model_name",
    "prompt_type",
    "n_samples",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "invalid_output_rate",
    "avg_output_length_chars",
    "avg_inference_time_sec",
    "total_inference_time_sec",
    "avg_input_tokens",
    "avg_output_tokens",
    "tokens_per_sec",
    "predictions_path",
]
DECODER_COMPARISON_COLUMNS = [
    "model_name",
    "zero_shot_accuracy",
    "zero_shot_macro_f1",
    "cot_zero_shot_accuracy",
    "cot_zero_shot_macro_f1",
    "few_shot_accuracy",
    "few_shot_macro_f1",
    "best_prompt_type",
    "best_accuracy",
    "best_macro_f1",
]
PROMPT_TYPE_SUMMARY_COLUMNS = [
    "prompt_type",
    "accuracy_mean",
    "accuracy_std",
    "accuracy_max",
    "macro_f1_mean",
    "macro_f1_std",
    "macro_f1_max",
    "invalid_output_rate_mean",
    "invalid_output_rate_std",
    "invalid_output_rate_max",
]
MODEL_SUMMARY_COLUMNS = [
    "model_name",
    "best_accuracy",
    "best_macro_f1",
    "mean_accuracy",
    "mean_macro_f1",
    "best_prompt_type",
]
SKIPPED_COLUMNS = [
    "run_name",
    "predictions_path",
    "skip_reason",
]
ENCODER_DECODER_COLUMNS = [
    "model_family",
    "model_name",
    "prompt_type",
    "accuracy",
    "macro_f1",
    "source_phase",
    "run_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase 4 decoder prompting outputs.")
    parser.add_argument("--phase4_dir", type=Path, default=Path("outputs/phase4"))
    parser.add_argument(
        "--phase3_table",
        type=Path,
        default=Path("outputs/phase3/tables/final_encoder_results_table.csv"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/phase4/tables"))
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def discover_prediction_files(phase4_dir: Path) -> list[Path]:
    """Return existing Phase 4 prediction files, ignoring folders without predictions."""
    if not phase4_dir.exists():
        warnings.warn(f"Phase 4 directory does not exist: {phase4_dir}", RuntimeWarning, stacklevel=2)
        return []
    return sorted(phase4_dir.glob("*/predictions.csv"))


def validate_predictions_df(df: pd.DataFrame, predictions_path: Path) -> None:
    """Validate required columns and canonical true-author labels."""
    if df.empty:
        raise ValueError("predictions.csv is empty")
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"predictions.csv is missing required columns: {missing}")
    true_authors = df["true_author"].map(normalize_author)
    invalid_true = sorted(set(true_authors) - set(CANONICAL_AUTHORS))
    if invalid_true:
        raise ValueError(f"predictions.csv has non-canonical true_author values: {invalid_true}")
    model_names = unique_nonempty(df["model_name"])
    prompt_types = unique_nonempty(df["prompt_type"])
    if len(model_names) != 1 or len(prompt_types) != 1:
        raise ValueError("predictions.csv has missing model_name or prompt_type metadata")
    if prompt_types[0] not in VALID_PROMPT_TYPES:
        raise ValueError(f"predictions.csv has unknown prompt_type: {prompt_types[0]!r}")


def infer_run_metadata(df: pd.DataFrame, predictions_path: Path) -> dict[str, str]:
    """Infer stable run metadata from a predictions dataframe."""
    return {
        "run_name": predictions_path.parent.name,
        "model_name": first_unique_nonempty(df["model_name"]) if "model_name" in df else "",
        "prompt_type": first_unique_nonempty(df["prompt_type"]) if "prompt_type" in df else "",
    }


def first_unique_nonempty(series: pd.Series) -> str:
    values = unique_nonempty(series)
    return values[0] if values else ""


def unique_nonempty(series: pd.Series) -> list[str]:
    return [str(value).strip() for value in series.dropna().unique() if str(value).strip()]


def normalize_author(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def normalize_prediction(value: Any) -> str:
    author = normalize_author(value)
    return author if author in CANONICAL_AUTHORS else INVALID_LABEL


def compute_run_metrics(df: pd.DataFrame, predictions_path: Path) -> dict[str, Any]:
    """Compute aggregate, per-author, parser, and runtime metrics for one run."""
    metadata = infer_run_metadata(df, predictions_path)
    y_true = df["true_author"].map(normalize_author).to_numpy()
    y_pred = df["predicted_author"].map(normalize_prediction).to_numpy()
    n_samples = int(len(df))

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CANONICAL_AUTHORS,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CANONICAL_AUTHORS,
        average="weighted",
        zero_division=0,
    )
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CANONICAL_AUTHORS,
        zero_division=0,
    )

    per_author = [
        {
            "author": author,
            "precision": float(per_precision[index]),
            "recall": float(per_recall[index]),
            "f1": float(per_f1[index]),
            "support": int(per_support[index]),
        }
        for index, author in enumerate(CANONICAL_AUTHORS)
    ]
    matrix = confusion_matrix(y_true, y_pred, labels=PREDICTION_LABELS)[: len(CANONICAL_AUTHORS), :]
    invalid_mask = pd.Series(y_pred).eq(INVALID_LABEL)
    output_lengths = df["raw_output"].fillna("").astype(str).str.len()
    input_tokens = numeric_column(df, "input_tokens")
    output_tokens = numeric_column(df, "output_tokens")
    inference_time = numeric_column(df, "inference_time_sec")
    total_inference_time = optional_sum(inference_time)
    total_output_tokens = optional_sum(output_tokens)

    return {
        **metadata,
        "predictions_path": str(predictions_path),
        "n_samples": n_samples,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_author": per_author,
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_rows": CANONICAL_AUTHORS,
        "confusion_matrix_columns": PREDICTION_LABELS,
        "parse_status_counts": df["parsed_status"].fillna("").astype(str).value_counts(dropna=False).to_dict(),
        "invalid_output_count": int(invalid_mask.sum()),
        "invalid_output_rate": float(invalid_mask.mean()) if n_samples else np.nan,
        "avg_output_length_chars": optional_mean(output_lengths),
        "avg_inference_time_sec": optional_mean(inference_time),
        "total_inference_time_sec": total_inference_time,
        "avg_input_tokens": optional_mean(input_tokens),
        "avg_output_tokens": optional_mean(output_tokens),
        "tokens_per_sec": (
            float(total_output_tokens / total_inference_time)
            if total_output_tokens is not None and total_inference_time and total_inference_time > 0
            else np.nan
        ),
    }


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in df:
        return None
    values = pd.to_numeric(df[column], errors="coerce")
    return values.dropna()


def optional_mean(values: pd.Series | None) -> float:
    if values is None or values.empty:
        return np.nan
    return float(values.mean())


def optional_sum(values: pd.Series | None) -> float | None:
    if values is None or values.empty:
        return None
    return float(values.sum())


def write_per_run_outputs(metrics: dict[str, Any], run_dir: Path) -> None:
    """Write per-run JSON metrics, per-author report, and confusion matrix."""
    run_dir.mkdir(parents=True, exist_ok=True)
    json_payload = json_ready(metrics)
    (run_dir / "metrics.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    pd.DataFrame(metrics["per_author"]).to_csv(run_dir / "classification_report.csv", index=False)
    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["confusion_matrix_rows"],
        columns=metrics["confusion_matrix_columns"],
    )
    matrix.index.name = "true_author"
    matrix.columns.name = "predicted_author"
    matrix.to_csv(run_dir / "confusion_matrix.csv")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def aggregate_decoder_results(metrics_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build one decoder result row per valid run."""
    rows = []
    for metrics in metrics_rows:
        rows.append({column: metrics.get(column) for column in DECODER_RESULT_COLUMNS})
    frame = pd.DataFrame(rows, columns=DECODER_RESULT_COLUMNS)
    return sort_by_metric(frame, "macro_f1")


def build_decoder_comparison_table(decoder_results: pd.DataFrame) -> pd.DataFrame:
    """Build report-ready model by prompt-type comparison table."""
    if decoder_results.empty:
        return pd.DataFrame(columns=DECODER_COMPARISON_COLUMNS)
    rows: list[dict[str, Any]] = []
    for model_name, group in decoder_results.groupby("model_name", dropna=False):
        row: dict[str, Any] = {column: np.nan for column in DECODER_COMPARISON_COLUMNS}
        row["model_name"] = model_name
        for prompt_type in ("zero_shot", "cot_zero_shot", "few_shot"):
            prompt_rows = group[group["prompt_type"] == prompt_type]
            if prompt_rows.empty:
                continue
            best = prompt_rows.sort_values("macro_f1", ascending=False, na_position="last").iloc[0]
            row[f"{prompt_type}_accuracy"] = best["accuracy"]
            row[f"{prompt_type}_macro_f1"] = best["macro_f1"]
        best_overall = group.sort_values("macro_f1", ascending=False, na_position="last").iloc[0]
        row["best_prompt_type"] = best_overall["prompt_type"]
        row["best_accuracy"] = best_overall["accuracy"]
        row["best_macro_f1"] = best_overall["macro_f1"]
        rows.append(row)
    return sort_by_metric(pd.DataFrame(rows, columns=DECODER_COMPARISON_COLUMNS), "best_macro_f1")


def build_prompt_type_summary(decoder_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize decoder results by prompt type."""
    if decoder_results.empty:
        return pd.DataFrame(columns=PROMPT_TYPE_SUMMARY_COLUMNS)
    summary = (
        decoder_results.groupby("prompt_type", dropna=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            accuracy_max=("accuracy", "max"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_max=("macro_f1", "max"),
            invalid_output_rate_mean=("invalid_output_rate", "mean"),
            invalid_output_rate_std=("invalid_output_rate", "std"),
            invalid_output_rate_max=("invalid_output_rate", "max"),
        )
        .reset_index()
    )
    return summary[PROMPT_TYPE_SUMMARY_COLUMNS]


def build_model_summary(decoder_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize decoder results by model."""
    if decoder_results.empty:
        return pd.DataFrame(columns=MODEL_SUMMARY_COLUMNS)
    rows = []
    for model_name, group in decoder_results.groupby("model_name", dropna=False):
        best = group.sort_values("macro_f1", ascending=False, na_position="last").iloc[0]
        rows.append(
            {
                "model_name": model_name,
                "best_accuracy": best["accuracy"],
                "best_macro_f1": best["macro_f1"],
                "mean_accuracy": group["accuracy"].mean(),
                "mean_macro_f1": group["macro_f1"].mean(),
                "best_prompt_type": best["prompt_type"],
            }
        )
    return sort_by_metric(pd.DataFrame(rows, columns=MODEL_SUMMARY_COLUMNS), "best_macro_f1")


def load_encoder_baseline_table(phase3_table: Path) -> pd.DataFrame:
    """Load the Phase 3 encoder baseline table, returning an empty frame on failure."""
    if not phase3_table.exists():
        warnings.warn(f"Phase 3 encoder table not found: {phase3_table}", RuntimeWarning, stacklevel=2)
        return pd.DataFrame(columns=ENCODER_DECODER_COLUMNS)
    try:
        frame = pd.read_csv(phase3_table)
    except Exception as exc:
        warnings.warn(f"Could not read Phase 3 encoder table {phase3_table}: {exc}", RuntimeWarning, stacklevel=2)
        return pd.DataFrame(columns=ENCODER_DECODER_COLUMNS)
    required = {"experiment_name", "model_name", "accuracy", "macro_f1"}
    missing = sorted(required - set(frame.columns))
    if missing:
        warnings.warn(f"Phase 3 encoder table missing columns {missing}: {phase3_table}", RuntimeWarning, stacklevel=2)
        return pd.DataFrame(columns=ENCODER_DECODER_COLUMNS)
    mapped = pd.DataFrame(
        {
            "model_family": "encoder",
            "model_name": frame["model_name"],
            "prompt_type": "supervised_finetune",
            "accuracy": pd.to_numeric(frame["accuracy"], errors="coerce"),
            "macro_f1": pd.to_numeric(frame["macro_f1"], errors="coerce"),
            "source_phase": "phase3",
            "run_name": frame["experiment_name"],
        }
    )
    return mapped[ENCODER_DECODER_COLUMNS]


def build_encoder_vs_decoder_table(encoder_rows: pd.DataFrame, decoder_results: pd.DataFrame) -> pd.DataFrame:
    """Combine Phase 3 encoder rows and Phase 4 decoder prompting rows."""
    decoder_rows = pd.DataFrame(columns=ENCODER_DECODER_COLUMNS)
    if not decoder_results.empty:
        decoder_rows = pd.DataFrame(
            {
                "model_family": "decoder_prompting",
                "model_name": decoder_results["model_name"],
                "prompt_type": decoder_results["prompt_type"],
                "accuracy": decoder_results["accuracy"],
                "macro_f1": decoder_results["macro_f1"],
                "source_phase": "phase4",
                "run_name": decoder_results["run_name"],
            }
        )
    combined = pd.concat([encoder_rows, decoder_rows], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=ENCODER_DECODER_COLUMNS)
    return sort_by_metric(combined[ENCODER_DECODER_COLUMNS], "macro_f1")


def sort_by_metric(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame:
        return frame
    sorted_frame = frame.copy()
    sorted_frame[column] = pd.to_numeric(sorted_frame[column], errors="coerce")
    return sorted_frame.sort_values(column, ascending=False, na_position="last")


def save_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        for column in columns:
            if column not in frame:
                frame[column] = np.nan
        frame = frame[columns]
    frame.to_csv(path, index=False)


def save_markdown(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        markdown = frame.to_markdown(index=False)
    except ImportError:
        warnings.warn(
            f"tabulate is not available; writing CSV-style text fallback to {path}",
            RuntimeWarning,
            stacklevel=2,
        )
        markdown = frame.to_csv(index=False)
    path.write_text(markdown + "\n", encoding="utf-8")


def skipped_row(predictions_path: Path, reason: str) -> dict[str, str]:
    return {
        "run_name": predictions_path.parent.name,
        "predictions_path": str(predictions_path),
        "skip_reason": reason,
    }


def process_prediction_file(predictions_path: Path, strict: bool) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    try:
        df = pd.read_csv(predictions_path)
        validate_predictions_df(df, predictions_path)
    except Exception as exc:
        if strict:
            raise
        reason = str(exc)
        warnings.warn(f"Skipping {predictions_path}: {reason}", RuntimeWarning, stacklevel=2)
        return None, skipped_row(predictions_path, reason)
    metrics = compute_run_metrics(df, predictions_path)
    write_per_run_outputs(metrics, predictions_path.parent)
    return metrics, None


def main() -> None:
    args = parse_args()
    phase4_dir = resolve_repo_path(args.phase4_dir)
    phase3_table = resolve_repo_path(args.phase3_table)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []
    for predictions_path in discover_prediction_files(phase4_dir):
        metrics, skipped = process_prediction_file(predictions_path, strict=args.strict)
        if metrics is not None:
            metrics_rows.append(metrics)
        if skipped is not None:
            skipped_rows.append(skipped)

    decoder_results = aggregate_decoder_results(metrics_rows)
    decoder_comparison = build_decoder_comparison_table(decoder_results)
    prompt_summary = build_prompt_type_summary(decoder_results)
    model_summary = build_model_summary(decoder_results)
    skipped = pd.DataFrame(skipped_rows, columns=SKIPPED_COLUMNS)
    encoder_rows = load_encoder_baseline_table(phase3_table)
    encoder_decoder = build_encoder_vs_decoder_table(encoder_rows, decoder_results)

    save_csv(decoder_results, output_dir / "decoder_results.csv", DECODER_RESULT_COLUMNS)
    save_markdown(decoder_results, output_dir / "decoder_results.md")
    save_csv(decoder_comparison, output_dir / "decoder_comparison_table.csv", DECODER_COMPARISON_COLUMNS)
    save_csv(prompt_summary, output_dir / "decoder_prompt_type_summary.csv", PROMPT_TYPE_SUMMARY_COLUMNS)
    save_csv(model_summary, output_dir / "decoder_model_summary.csv", MODEL_SUMMARY_COLUMNS)
    save_csv(skipped, output_dir / "skipped_invalid_runs.csv", SKIPPED_COLUMNS)
    save_csv(encoder_decoder, output_dir / "encoder_vs_decoder_comparison.csv", ENCODER_DECODER_COLUMNS)
    save_markdown(encoder_decoder, output_dir / "encoder_vs_decoder_comparison.md")

    print(f"valid_runs={len(metrics_rows)}")
    print(f"skipped_runs={len(skipped_rows)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
