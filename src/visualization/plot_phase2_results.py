"""Plot Phase 2 encoder baseline results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import phase_artifact_dir  # noqa: E402


def plot_metric_bar(frame: pd.DataFrame, metric: str, output_path: Path) -> None:
    if frame.empty or metric not in frame.columns:
        print(f"WARNING: Cannot plot {metric}; no aggregate data available.")
        return
    plot_frame = frame.sort_values(metric, ascending=False)
    plt.figure(figsize=(8, 4.5))
    plt.bar(plot_frame["experiment_name"].astype(str), plot_frame[metric].astype(float), color="#4C78A8")
    plt.ylim(0, 1)
    plt.xlabel("Experiment")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def plot_confusion_matrices(runs_dir: Path, plots_dir: Path) -> None:
    for matrix_path in sorted(runs_dir.glob("*/confusion_matrix.csv")):
        experiment_name = matrix_path.parent.name
        matrix = pd.read_csv(matrix_path, index_col=0)
        plt.figure(figsize=(8, 6))
        values = matrix.to_numpy()
        image = plt.imshow(values, cmap="Blues")
        plt.colorbar(image)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                plt.text(column_index, row_index, str(values[row_index, column_index]), ha="center", va="center")
        plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
        plt.yticks(range(len(matrix.index)), matrix.index)
        plt.xlabel("Predicted author")
        plt.ylabel("True author")
        plt.title(experiment_name)
        plt.tight_layout()
        output_path = plots_dir / f"{experiment_name}_confusion_matrix.png"
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    table_dir = phase_artifact_dir("phase2", "tables")
    plots_dir = phase_artifact_dir("phase2", "plots")
    runs_dir = phase_artifact_dir("phase2", "runs")
    results_path = table_dir / "encoder_results.csv"
    if not results_path.exists():
        print(f"WARNING: Aggregate results not found: {results_path}")
        return 0
    frame = pd.read_csv(results_path)
    plot_metric_bar(frame, "accuracy", plots_dir / "encoder_accuracy_bar.png")
    plot_metric_bar(frame, "macro_f1", plots_dir / "encoder_macro_f1_bar.png")
    plot_confusion_matrices(runs_dir, plots_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
