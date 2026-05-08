"""Plotting helpers for Phase 2 evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    label_names: list[str],
    output_path: str | Path,
    title: str = "BERT-base Confusion Matrix",
) -> None:
    """Save a confusion matrix heatmap with author names on both axes."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted Author")
    ax.set_ylabel("True Author")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=35, ha="right")
    ax.set_yticklabels(label_names)

    threshold = float(matrix.max()) / 2.0 if matrix.size and matrix.max() > 0 else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = f"{value:.2f}" if np.issubdtype(matrix.dtype, np.floating) else str(int(value))
            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
