"""Reusable classification metrics for authorship attribution."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    label_ids: list[int],
    label_names: list[str],
) -> dict[str, Any]:
    """Compute aggregate and per-class classification metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    per_class = {}
    for index, label_id in enumerate(label_ids):
        per_class[str(label_id)] = {
            "label": int(label_id),
            "label_name": label_names[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "macro_f1": float(macro_f1),
        "precision_weighted": float(weighted_precision),
        "recall_weighted": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
    }


def build_classification_report(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    label_ids: list[int],
    label_names: list[str],
) -> dict[str, Any]:
    """Build a sklearn classification report keyed by author names."""
    return classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )


def build_confusion_matrices(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    label_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw-count and row-normalized confusion matrices."""
    raw = confusion_matrix(y_true, y_pred, labels=label_ids)
    row_sums = raw.sum(axis=1, keepdims=True)
    normalized = np.divide(raw, row_sums, out=np.zeros_like(raw, dtype=float), where=row_sums != 0)
    return raw, normalized


def trainer_compute_metrics(label_ids: list[int], label_names: list[str]):
    """Create a Hugging Face Trainer compute_metrics callback."""

    def _compute(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metrics = compute_classification_metrics(labels, predictions, label_ids, label_names)
        return {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
        }

    return _compute
