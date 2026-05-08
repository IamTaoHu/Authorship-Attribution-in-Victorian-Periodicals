"""Report writers for Phase 2 baseline reproduction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PAPER_ACCURACY = 0.93
PAPER_MACRO_F1 = 0.91


def save_json(payload: dict[str, Any], output_path: str | Path) -> None:
    """Save JSON with stable indentation."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def classification_report_to_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Convert sklearn classification_report output to a table."""
    rows: list[dict[str, Any]] = []
    for label, values in report.items():
        if label == "accuracy":
            rows.append(
                {
                    "label": "accuracy",
                    "precision": "",
                    "recall": "",
                    "f1-score": float(values),
                    "support": "",
                }
            )
            continue
        if isinstance(values, dict):
            rows.append(
                {
                    "label": label,
                    "precision": float(values.get("precision", 0.0)),
                    "recall": float(values.get("recall", 0.0)),
                    "f1-score": float(values.get("f1-score", 0.0)),
                    "support": int(values.get("support", 0)),
                }
            )
    return pd.DataFrame(rows)


def save_classification_report(report: dict[str, Any], json_path: str | Path, csv_path: str | Path) -> None:
    """Save classification report as JSON and CSV."""
    save_json(report, json_path)
    frame = classification_report_to_frame(report)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)


def save_confusion_matrix_csv(
    matrix,
    label_names: list[str],
    output_path: str | Path,
) -> None:
    """Save a confusion matrix with author-name axes."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(matrix, index=label_names, columns=label_names)
    frame.index.name = "true_label_name"
    frame.columns.name = "predicted_label_name"
    frame.to_csv(path)


def build_split_summary(split_labels: dict[str, list[int]], label_id_to_name: dict[int, str]) -> dict[str, Any]:
    """Build per-split counts for reproducibility reporting."""
    summary: dict[str, Any] = {}
    for split, labels in split_labels.items():
        counts: dict[str, dict[str, Any]] = {}
        for label_id in sorted(label_id_to_name):
            count = sum(1 for label in labels if int(label) == label_id)
            counts[str(label_id)] = {
                "label": label_id,
                "label_name": label_id_to_name[label_id],
                "count": count,
            }
        summary[split] = {
            "num_rows": len(labels),
            "class_counts": counts,
        }
    return summary


def build_baseline_comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compare a test run with the paper BERT-base baseline."""
    accuracy = float(metrics.get("accuracy", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    accuracy_diff = accuracy - PAPER_ACCURACY
    macro_f1_diff = macro_f1 - PAPER_MACRO_F1

    if abs(accuracy_diff) <= 0.03 and abs(macro_f1_diff) <= 0.03:
        interpretation = "close match"
    elif accuracy < PAPER_ACCURACY and macro_f1 < PAPER_MACRO_F1:
        interpretation = "underfitting or dataset/split mismatch possibility"
    elif accuracy > PAPER_ACCURACY and macro_f1 > PAPER_MACRO_F1:
        interpretation = "overfitting or easier dataset/split possibility"
    else:
        interpretation = "mixed result; inspect per-class metrics and split compatibility"

    return {
        "paper_accuracy": PAPER_ACCURACY,
        "paper_macro_f1": PAPER_MACRO_F1,
        "current_accuracy": accuracy,
        "current_macro_f1": macro_f1,
        "accuracy_absolute_difference": abs(accuracy_diff),
        "macro_f1_absolute_difference": abs(macro_f1_diff),
        "accuracy_signed_difference": accuracy_diff,
        "macro_f1_signed_difference": macro_f1_diff,
        "is_close_reproduction": abs(accuracy_diff) <= 0.03 and abs(macro_f1_diff) <= 0.03,
        "interpretation": interpretation,
    }


def save_baseline_comparison(comparison: dict[str, Any], json_path: str | Path, md_path: str | Path) -> None:
    """Save baseline comparison as JSON and Markdown."""
    save_json(comparison, json_path)
    markdown = "\n".join(
        [
            "# BERT-base Baseline Comparison",
            "",
            f"- Paper accuracy: {comparison['paper_accuracy']:.4f}",
            f"- Current test accuracy: {comparison['current_accuracy']:.4f}",
            f"- Accuracy absolute difference: {comparison['accuracy_absolute_difference']:.4f}",
            f"- Paper macro F1: {comparison['paper_macro_f1']:.4f}",
            f"- Current test macro F1: {comparison['current_macro_f1']:.4f}",
            f"- Macro F1 absolute difference: {comparison['macro_f1_absolute_difference']:.4f}",
            f"- Close reproduction: {comparison['is_close_reproduction']}",
            f"- Interpretation: {comparison['interpretation']}",
            "",
        ]
    )
    path = Path(md_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
