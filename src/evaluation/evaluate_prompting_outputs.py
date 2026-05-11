"""Aggregate Phase 4 decoder prompting runs into consolidated result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VALID_AUTHORS = [
    "Leslie Stephen",
    "John Morley",
    "Eliza Lynn Linton",
    "George Henry Lewes",
    "Anne Mozley",
    "James Fitzjames Stephen",
]
INVALID_PREDICTION = "__INVALID__"
BENCHMARK_COLUMNS = [
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
]
PARSED_STATUS_COLUMNS = [
    "run_name",
    "model_name",
    "prompt_type",
    "parsed_status",
    "count",
    "rate",
]
PREDICTION_DISTRIBUTION_COLUMNS = [
    "run_name",
    "model_name",
    "prompt_type",
    "predicted_author",
    "count",
    "rate",
]
SKIPPED_COLUMNS = ["run_name", "run_dir", "skip_reason"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase 4 decoder prompting outputs.")
    parser.add_argument("--phase4_dir", type=Path, default=Path("outputs/phase4"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/phase4/tables"))
    parser.add_argument("--strict", action="store_true", help="Raise instead of skipping malformed run directories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase4_dir = resolve_repo_path(args.phase4_dir)
    output_dir = resolve_repo_path(args.output_dir)

    rows: list[dict[str, Any]] = []
    parsed_status_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []

    candidates = discover_run_dirs(phase4_dir, output_dir)
    print_discovered_runs(candidates)

    for run_dir in candidates:
        try:
            run_result = process_run_dir(run_dir)
        except Exception as exc:
            if args.strict:
                raise
            skipped_rows.append(skipped_row(run_dir, str(exc)))
            continue
        rows.append(run_result.benchmark_row)
        parsed_status_rows.extend(run_result.parsed_status_rows)
        prediction_rows.extend(run_result.prediction_distribution_rows)

    benchmark = sort_benchmark(pd.DataFrame(rows, columns=BENCHMARK_COLUMNS))
    parsed_status = sort_detail_table(pd.DataFrame(parsed_status_rows, columns=PARSED_STATUS_COLUMNS))
    prediction_distribution = sort_detail_table(
        pd.DataFrame(prediction_rows, columns=PREDICTION_DISTRIBUTION_COLUMNS)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(output_dir / "benchmark_summary.csv", index=False)
    dataframe_to_markdown(benchmark).write_text_to(output_dir / "benchmark_summary.md")
    parsed_status.to_csv(output_dir / "parsed_status_summary.csv", index=False)
    prediction_distribution.to_csv(output_dir / "prediction_distribution.csv", index=False)
    benchmark.to_csv(output_dir / "final_decoder_results_table.csv", index=False)

    print_skipped_runs(skipped_rows)
    print_leaderboard(benchmark)
    print(f"\nWrote Phase 4 tables to {output_dir}")


class MarkdownTable(str):
    def write_text_to(self, path: Path) -> None:
        path.write_text(str(self) + "\n", encoding="utf-8")


class RunResult:
    def __init__(
        self,
        benchmark_row: dict[str, Any],
        parsed_status_rows: list[dict[str, Any]],
        prediction_distribution_rows: list[dict[str, Any]],
    ) -> None:
        self.benchmark_row = benchmark_row
        self.parsed_status_rows = parsed_status_rows
        self.prediction_distribution_rows = prediction_distribution_rows


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def discover_run_dirs(phase4_dir: Path, output_dir: Path) -> list[Path]:
    if not phase4_dir.exists():
        warnings.warn(f"Phase 4 directory does not exist: {phase4_dir}", RuntimeWarning, stacklevel=2)
        return []
    resolved_output_dir = output_dir.resolve()
    return [
        path
        for path in sorted(phase4_dir.iterdir(), key=lambda item: item.name)
        if path.is_dir() and path.resolve() != resolved_output_dir
    ]


def process_run_dir(run_dir: Path) -> RunResult:
    predictions_path = run_dir / "predictions.csv"
    runtime_path = run_dir / "runtime.json"
    if not predictions_path.exists():
        raise ValueError("missing predictions.csv")
    if not runtime_path.exists():
        raise ValueError("missing runtime.json")

    df = read_predictions(predictions_path)
    runtime = read_runtime(runtime_path)
    validate_predictions(df)
    metadata = run_metadata(run_dir, df, runtime)

    normalized_true = df["true_author"].map(normalize_author)
    normalized_pred = normalized_prediction_series(df)
    metric_pred = normalized_pred.where(normalized_pred.isin(VALID_AUTHORS), INVALID_PREDICTION)
    invalid_mask = invalid_output_mask(df, normalized_pred, run_dir)
    n_samples = int(len(df))

    _, _, macro_f1, _ = precision_recall_fscore_support(
        normalized_true,
        metric_pred,
        labels=VALID_AUTHORS,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        normalized_true,
        metric_pred,
        labels=VALID_AUTHORS,
        average="weighted",
        zero_division=0,
    )

    benchmark_row = {
        **metadata,
        "n_samples": n_samples,
        "accuracy": compute_accuracy(df, normalized_true, normalized_pred),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "invalid_output_rate": float(invalid_mask.mean()) if n_samples else np.nan,
        "avg_output_length_chars": mean_raw_output_length(df),
        "avg_inference_time_sec": average_inference_time(df, runtime),
        "total_inference_time_sec": total_inference_time(df, runtime),
    }
    return RunResult(
        benchmark_row=benchmark_row,
        parsed_status_rows=build_parsed_status_rows(df, metadata),
        prediction_distribution_rows=build_prediction_distribution_rows(normalized_pred, metadata),
    )


def read_predictions(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"could not read predictions.csv: {exc}") from exc
    if df.empty:
        raise ValueError("predictions.csv is empty")
    return df


def read_runtime(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as exc:
        raise ValueError(f"could not read runtime.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("runtime.json does not contain a JSON object")
    return payload


def validate_predictions(df: pd.DataFrame) -> None:
    if "true_author" not in df.columns:
        raise ValueError("predictions.csv is missing required column: true_author")
    normalized_true = df["true_author"].map(normalize_author)
    invalid_true = sorted(set(normalized_true) - set(VALID_AUTHORS))
    if invalid_true:
        raise ValueError(f"predictions.csv has non-canonical true_author values: {invalid_true}")
    if "predicted_author" not in df.columns:
        warnings.warn(
            "predictions.csv is missing predicted_author; all predictions will be treated as invalid",
            RuntimeWarning,
            stacklevel=2,
        )
    if "raw_output" not in df.columns:
        warnings.warn(
            "predictions.csv is missing raw_output; output length will be blank",
            RuntimeWarning,
            stacklevel=2,
        )


def run_metadata(run_dir: Path, df: pd.DataFrame, runtime: dict[str, Any]) -> dict[str, str]:
    return {
        "run_name": str(runtime.get("run_name") or run_dir.name),
        "model_name": str(runtime.get("model_name") or first_unique(df, "model_name") or ""),
        "prompt_type": str(runtime.get("prompt_type") or first_unique(df, "prompt_type") or ""),
    }


def first_unique(df: pd.DataFrame, column: str) -> str | None:
    if column not in df.columns:
        return None
    values = [str(value).strip() for value in df[column].dropna().unique() if str(value).strip()]
    return values[0] if values else None


def normalize_author(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def normalized_prediction_series(df: pd.DataFrame) -> pd.Series:
    if "predicted_author" not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df["predicted_author"].map(normalize_author)


def invalid_output_mask(df: pd.DataFrame, normalized_pred: pd.Series, run_dir: Path) -> pd.Series:
    invalid_prediction = normalized_pred.eq("") | ~normalized_pred.isin(VALID_AUTHORS)
    if "parsed_status" not in df.columns:
        warnings.warn(
            f"{run_dir.name}: parsed_status column is missing; invalid rate uses canonical prediction validity only",
            RuntimeWarning,
            stacklevel=2,
        )
        return invalid_prediction
    parsed_status = df["parsed_status"].map(normalize_status)
    invalid_status = parsed_status.eq("") | ~parsed_status.str.startswith("ok", na=False)
    return invalid_prediction | invalid_status


def normalize_status(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def compute_accuracy(df: pd.DataFrame, normalized_true: pd.Series, normalized_pred: pd.Series) -> float:
    if "correct" in df.columns:
        correct = df["correct"].map(parse_bool)
        if correct.notna().any():
            return float(correct.fillna(False).mean())
    return float(normalized_true.eq(normalized_pred).mean())


def parse_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def mean_raw_output_length(df: pd.DataFrame) -> float:
    if "raw_output" not in df.columns:
        return np.nan
    return float(df["raw_output"].fillna("").astype(str).str.len().mean())


def average_inference_time(df: pd.DataFrame, runtime: dict[str, Any]) -> float:
    prediction_times = numeric_column(df, "inference_time_sec")
    if not prediction_times.empty:
        return float(prediction_times.mean())
    return numeric_value(runtime.get("avg_time_sec"))


def total_inference_time(df: pd.DataFrame, runtime: dict[str, Any]) -> float:
    prediction_times = numeric_column(df, "inference_time_sec")
    if not prediction_times.empty:
        return float(prediction_times.sum())
    return numeric_value(runtime.get("total_time_sec"))


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").dropna()


def numeric_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def build_parsed_status_rows(df: pd.DataFrame, metadata: dict[str, str]) -> list[dict[str, Any]]:
    n_samples = len(df)
    if "parsed_status" in df.columns:
        statuses = df["parsed_status"].map(status_for_table)
    else:
        statuses = pd.Series(["__MISSING_COLUMN__"] * n_samples, index=df.index)
    counts = statuses.value_counts(dropna=False).sort_index()
    return [
        {
            **metadata,
            "parsed_status": status,
            "count": int(count),
            "rate": float(count / n_samples) if n_samples else np.nan,
        }
        for status, count in counts.items()
    ]


def status_for_table(value: Any) -> str:
    status = normalize_status(value)
    return status if status else "__BLANK__"


def build_prediction_distribution_rows(
    normalized_pred: pd.Series,
    metadata: dict[str, str],
) -> list[dict[str, Any]]:
    n_samples = len(normalized_pred)
    predictions = normalized_pred.where(normalized_pred.isin(VALID_AUTHORS), INVALID_PREDICTION)
    counts = predictions.value_counts(dropna=False).sort_index()
    return [
        {
            **metadata,
            "predicted_author": predicted_author,
            "count": int(count),
            "rate": float(count / n_samples) if n_samples else np.nan,
        }
        for predicted_author, count in counts.items()
    ]


def sort_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=BENCHMARK_COLUMNS)
    sorted_frame = frame.copy()
    sorted_frame["accuracy"] = pd.to_numeric(sorted_frame["accuracy"], errors="coerce")
    sorted_frame["macro_f1"] = pd.to_numeric(sorted_frame["macro_f1"], errors="coerce")
    return sorted_frame.sort_values(
        ["accuracy", "macro_f1", "run_name"],
        ascending=[False, False, True],
        na_position="last",
    )[BENCHMARK_COLUMNS]


def sort_detail_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_columns = [column for column in ["run_name", "model_name", "prompt_type"] if column in frame.columns]
    ascending = [True] * len(sort_columns)
    if "count" in frame.columns:
        frame = frame.copy()
        frame["count"] = pd.to_numeric(frame["count"], errors="coerce")
        sort_columns.extend(["count"])
        ascending.append(False)
    for detail_column in ["parsed_status", "predicted_author"]:
        if detail_column in frame.columns:
            sort_columns.append(detail_column)
            ascending.append(True)
            break
    return frame.sort_values(sort_columns, ascending=ascending, na_position="last")


def dataframe_to_markdown(frame: pd.DataFrame) -> MarkdownTable:
    if frame.empty:
        return MarkdownTable("| No Phase 4 runs found |\n| --- |")
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = [format_markdown_value(row[column]) for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return MarkdownTable("\n".join(lines))


def format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def skipped_row(run_dir: Path, reason: str) -> dict[str, str]:
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "skip_reason": reason,
    }


def print_discovered_runs(candidates: list[Path]) -> None:
    print("Discovered Phase 4 run directories:")
    if not candidates:
        print("  none")
        return
    for path in candidates:
        print(f"  {path.name}")


def print_skipped_runs(skipped_rows: list[dict[str, str]]) -> None:
    print("\nSkipped run directories:")
    if not skipped_rows:
        print("  none")
        return
    skipped = pd.DataFrame(skipped_rows, columns=SKIPPED_COLUMNS)
    print(skipped.to_string(index=False))


def print_leaderboard(benchmark: pd.DataFrame) -> None:
    print("\nFinal ranked leaderboard:")
    if benchmark.empty:
        print("  no valid runs")
        return
    columns = ["run_name", "model_name", "prompt_type", "accuracy", "macro_f1", "invalid_output_rate"]
    print(benchmark[columns].to_string(index=False))


if __name__ == "__main__":
    main()
