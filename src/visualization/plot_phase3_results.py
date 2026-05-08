"""Create Phase 3 comparison plots and per-run diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Phase 3 encoder results.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/phase3"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = resolve_repo_path(args.outputs_dir)
    tables_dir = outputs_dir / "tables"
    plots_dir = outputs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(tables_dir / "encoder_results.csv")
    if not results.empty:
        plot_bar(results, "macro_f1", plots_dir / "macro_f1_comparison.png", "Macro F1")
        plot_bar(results, "accuracy", plots_dir / "accuracy_comparison.png", "Accuracy")
        plot_lora(results, plots_dir / "deberta_lora_rank_ablation.png")
        plot_modernbert(results, plots_dir / "modernbert_context_length_ablation.png")

    for experiment_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        if experiment_dir.name in {"tables", "plots"}:
            continue
        plot_confusion_matrix(experiment_dir)
        plot_training_curves(experiment_dir)

    print(f"Wrote Phase 3 plots to {plots_dir}")


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_bar(frame: pd.DataFrame, metric: str, output_path: Path, label: str) -> None:
    data = frame.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if data.empty:
        return
    plt.figure(figsize=(12, max(5, 0.45 * len(data))))
    sns.barplot(data=data, y="experiment_name", x=metric, color="#3a6ea5")
    plt.xlabel(label)
    plt.ylabel("Experiment")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_lora(frame: pd.DataFrame, output_path: Path) -> None:
    data = frame[frame["use_lora"].fillna(False).astype(bool)].dropna(subset=["lora_r", "macro_f1"])
    if data.empty:
        return
    data = data.sort_values("lora_r")
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=data, x="lora_r", y="macro_f1", marker="o")
    plt.xlabel("LoRA rank")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_modernbert(frame: pd.DataFrame, output_path: Path) -> None:
    model_names = frame["model_name"].fillna("")
    data = frame[model_names.str.contains("ModernBERT", case=False, regex=False)].dropna(subset=["max_length", "macro_f1"])
    if data.empty:
        return
    data = data.sort_values("max_length")
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=data, x="max_length", y="macro_f1", marker="o")
    plt.xlabel("Max length")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(experiment_dir: Path) -> None:
    predictions_path = experiment_dir / "predictions.csv"
    if not predictions_path.exists():
        return
    predictions = pd.read_csv(predictions_path)
    if predictions.empty or "true_label" not in predictions or "pred_label" not in predictions:
        return
    labels = sorted(set(predictions["true_label"].dropna().astype(int)) | set(predictions["pred_label"].dropna().astype(int)))
    label_names = label_names_from_predictions(predictions, labels)
    matrix = confusion_matrix(predictions["true_label"].astype(int), predictions["pred_label"].astype(int), labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted author")
    plt.ylabel("True author")
    plt.title(f"{experiment_dir.name} confusion matrix")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(experiment_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def label_names_from_predictions(predictions: pd.DataFrame, labels: list[int]) -> list[str]:
    names: dict[int, str] = {}
    for _, row in predictions.iterrows():
        names[int(row["true_label"])] = str(row.get("true_label_name", row["true_label"]))
        names[int(row["pred_label"])] = str(row.get("pred_label_name", row["pred_label"]))
    return [names.get(label, f"label_{label}") for label in labels]


def plot_training_curves(experiment_dir: Path) -> None:
    state_path = experiment_dir / "trainer_state.json"
    if not state_path.exists():
        return
    with state_path.open("r", encoding="utf-8") as file:
        state = json.load(file)
    rows: list[dict[str, Any]] = state.get("log_history", [])
    if not rows:
        return
    frame = pd.DataFrame(rows)
    metric_columns = [column for column in ("loss", "eval_loss", "eval_macro_f1", "eval_accuracy") if column in frame]
    if not metric_columns:
        return
    plt.figure(figsize=(10, 6))
    for column in metric_columns:
        points = frame.dropna(subset=[column])
        x_column = "epoch" if "epoch" in points else "step"
        if not points.empty and x_column in points:
            plt.plot(points[x_column], points[column], marker="o", label=column)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(experiment_dir / "training_curves.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
