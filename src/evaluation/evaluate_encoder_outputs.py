"""Recompute Phase 2 metrics from encoder prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import phase_artifact_dir  # noqa: E402


REQUIRED_COLUMNS = ("sample_id", "author", "label", "pred_label", "pred_author", "correct")
PROBABILITY_COLUMNS = [f"prob_{index}" for index in range(6)]


def find_prediction_paths(predictions: str | None, experiment_name: str | None) -> list[Path]:
    if predictions:
        return [Path(predictions)]
    runs_dir = phase_artifact_dir("phase2", "runs")
    if experiment_name:
        return [runs_dir / experiment_name / "predictions.csv"]
    return sorted(runs_dir.glob("*/predictions.csv"))


def load_label_names(run_dir: Path, frame: pd.DataFrame) -> list[str]:
    label_map_path = run_dir / "label_map.json"
    if label_map_path.exists():
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        return [author for author, _ in sorted(label_map.items(), key=lambda item: int(item[1]))]
    labels = sorted(frame["label"].astype(int).unique())
    return [str(label) for label in labels]


def evaluate_predictions(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    missing_probabilities = [column for column in PROBABILITY_COLUMNS if column not in frame.columns]
    if missing_probabilities:
        print(f"WARNING: {path} is missing probability columns: {', '.join(missing_probabilities)}")

    y_true = frame["label"].astype(int)
    y_pred = frame["pred_label"].astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    run_dir = path.parent
    experiment_name = run_dir.name
    label_names = load_label_names(run_dir, frame)
    label_ids = list(range(len(label_names)))

    table_dir = phase_artifact_dir("phase2", "tables")
    predictions_dir = phase_artifact_dir("phase2", "predictions")
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"}).to_csv(
        table_dir / f"{experiment_name}_classification_report.csv",
        index=False,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    matrix_frame = pd.DataFrame(matrix, index=label_names, columns=label_names)
    matrix_frame.index.name = "true_author"
    matrix_frame.to_csv(table_dir / f"{experiment_name}_confusion_matrix.csv")
    frame.to_csv(predictions_dir / f"{experiment_name}_predictions.csv", index=False)
    print(f"Evaluated {path}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--experiment_name", default=None)
    args = parser.parse_args()

    paths = find_prediction_paths(args.predictions, args.experiment_name)
    if not paths:
        raise FileNotFoundError("No Phase 2 predictions.csv files found.")
    for path in paths:
        metrics = evaluate_predictions(path)
        print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
