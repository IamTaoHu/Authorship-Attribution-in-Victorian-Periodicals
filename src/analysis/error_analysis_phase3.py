"""Focused error analysis for Phase 3 encoder predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FOCUS_AUTHORS = ["James Fitzjames Stephen", "Eliza Lynn Linton"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 error analysis.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/phase3"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = resolve_repo_path(args.outputs_dir)
    for experiment_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        if experiment_dir.name in {"tables", "plots"}:
            continue
        predictions_path = experiment_dir / "predictions.csv"
        if predictions_path.exists():
            print(f"[INFO] Processing experiment: {experiment_dir.name}")
            try:
                analyze_experiment(experiment_dir, predictions_path)
            except Exception:
                print(f"[ERROR] Failed error analysis for {experiment_dir.name}")
                traceback.print_exc()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def analyze_experiment(experiment_dir: Path, predictions_path: Path) -> None:
    output_dir = experiment_dir / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(predictions_path)
    print(f"[DEBUG] Loaded dataframe shape: {predictions.shape}")
    print(f"[DEBUG] Available columns: {list(predictions.columns)}")
    if predictions.empty:
        return
    predictions = add_analysis_columns(predictions)
    print(f"[DEBUG] predicted_probability NaN count: {int(predictions['predicted_probability'].isna().sum())}")

    errors = predictions[~predictions["correct"].astype(bool)].copy()
    errors = errors.copy()
    if "predicted_probability" not in errors.columns:
        raise ValueError("predicted_probability column missing")
    if errors["predicted_probability"].isna().all():
        raise ValueError("predicted_probability is entirely NaN")

    false_positives = errors.sort_values("predicted_probability", ascending=False)
    false_negatives = errors.sort_values(["true_label_name", "predicted_probability"], ascending=[True, False])
    hard_pairs = build_hard_pairs(errors)

    false_positives.to_csv(output_dir / "false_positives.csv", index=False)
    false_negatives.to_csv(output_dir / "false_negatives.csv", index=False)
    hard_pairs.to_csv(output_dir / "hard_author_pairs.csv", index=False)
    write_summary(predictions, errors, hard_pairs, output_dir / "error_summary.md", experiment_dir.name)
    print(f"[OK] Wrote error analysis to {output_dir}")


def add_analysis_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    if "correct" not in frame:
        frame["correct"] = frame["true_label"].astype(int) == frame["pred_label"].astype(int)
    probability_column = "probability_pred" if "probability_pred" in frame else "confidence"
    frame["predicted_probability"] = (
        pd.to_numeric(frame[probability_column], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    frame["text_length_chars"] = frame["text"].fillna("").astype(str).str.len()
    frame["text_length_words"] = frame["text"].fillna("").astype(str).str.split().str.len()
    frame["confidence_bucket"] = pd.cut(
        frame["predicted_probability"],
        bins=[-0.01, 0.50, 0.70, 0.90, 1.01],
        labels=["<=0.50", "0.50-0.70", "0.70-0.90", ">0.90"],
    )
    return frame


def build_hard_pairs(errors: pd.DataFrame) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame(columns=["true_label_name", "pred_label_name", "count", "avg_predicted_probability"])
    return (
        errors.groupby(["true_label_name", "pred_label_name"], dropna=False)
        .agg(count=("row_id", "count"), avg_predicted_probability=("predicted_probability", "mean"))
        .reset_index()
        .sort_values(["count", "avg_predicted_probability"], ascending=[False, False])
    )


def write_summary(predictions: pd.DataFrame, errors: pd.DataFrame, hard_pairs: pd.DataFrame, output_path: Path, experiment_name: str) -> None:
    total = len(predictions)
    error_count = len(errors)
    high_conf_wrong = errors[errors["predicted_probability"] >= 0.90].sort_values("predicted_probability", ascending=False)
    low_conf_correct = predictions[
        predictions["correct"].astype(bool) & (predictions["predicted_probability"] <= 0.70)
    ].sort_values("predicted_probability")
    short_errors = errors.sort_values("text_length_words").head(20)
    lines = [
        f"# Error Summary: {experiment_name}",
        "",
        f"- Total predictions: {total}",
        f"- Errors: {error_count}",
        f"- Error rate: {(error_count / total if total else 0.0):.4f}",
        f"- High-confidence wrong predictions (>=0.90): {len(high_conf_wrong)}",
        f"- Low-confidence correct predictions (<=0.70): {len(low_conf_correct)}",
        "",
        "## Most Confused Author Pairs",
        "",
        table_to_markdown(hard_pairs.head(10)),
        "",
        "## Focus Authors",
        "",
    ]
    for author in FOCUS_AUTHORS:
        involved = errors[(errors["true_label_name"] == author) | (errors["pred_label_name"] == author)]
        lines.append(f"- {author}: {len(involved)} errors involving this author")
    lines.extend(
        [
            "",
            "## Short Error Paragraphs",
            "",
            table_to_markdown(short_errors[["row_id", "true_label_name", "pred_label_name", "text_length_words", "predicted_probability"]].head(10)),
            "",
            "## High-Confidence Wrong Predictions",
            "",
            table_to_markdown(high_conf_wrong[["row_id", "true_label_name", "pred_label_name", "predicted_probability"]].head(10)),
            "",
            "## Low-Confidence Correct Predictions",
            "",
            table_to_markdown(low_conf_correct[["row_id", "true_label_name", "pred_label_name", "predicted_probability"]].head(10)),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def table_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = [format_markdown_value(row[column]) for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_markdown_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
