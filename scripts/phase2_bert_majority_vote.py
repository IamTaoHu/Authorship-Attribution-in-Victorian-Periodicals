"""Run multiple BERT-base seeds and aggregate predictions by majority vote."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.error_analysis import save_error_analysis  # noqa: E402
from src.evaluation.label_mapping import (  # noqa: E402
    get_author_id_to_name,
    get_label_names,
    probability_column_name,
    save_author_mapping,
)
from src.evaluation.metrics import (  # noqa: E402
    build_classification_report,
    build_confusion_matrices,
    compute_classification_metrics,
)
from src.evaluation.plots import save_confusion_matrix_plot  # noqa: E402
from src.evaluation.reports import (  # noqa: E402
    save_classification_report,
    save_confusion_matrix_csv,
    save_json,
)
from src.training.train_transformer_classifier import (  # noqa: E402
    TransformerTrainingConfig,
    prepare_phase2_dirs,
    run_transformer_baseline,
)


DEFAULT_SEEDS = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run 10x BERT-base majority-vote reproduction.")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--validation_size", type=float, default=0.1)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/phase2"))
    parser.add_argument("--phase1_dataset_path", type=Path, default=Path("outputs/phase1/periad_cleaned"))
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--fp16", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--probability_averaging", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run per-seed baselines and aggregate their predictions."""
    args = parse_args()
    seeds = args.seeds[: args.num_runs]
    paths = prepare_phase2_dirs(args.output_dir)
    save_author_mapping(paths["reports"] / "author_label_mapping.json")

    prediction_paths: list[Path] = []
    for seed in seeds:
        run_slug = f"bert_base_seed{seed}"
        prediction_path = paths["predictions"] / f"{run_slug}_predictions.csv"
        if args.skip_existing and prediction_path.exists():
            prediction_paths.append(prediction_path)
            continue

        config = TransformerTrainingConfig(
            model_name=args.model_name,
            seed=seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            max_length=args.max_length,
            validation_size=args.validation_size,
            output_dir=args.output_dir,
            phase1_dataset_path=args.phase1_dataset_path,
            save_total_limit=args.save_total_limit,
            fp16=args.fp16,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            max_test_samples=args.max_test_samples,
            run_name=run_slug,
        )
        result = run_transformer_baseline(config)
        prediction_paths.append(Path(result["predictions_path"]))

    prediction_frames = [pd.read_csv(path) for path in prediction_paths]
    if not prediction_frames:
        raise RuntimeError("No per-seed predictions were available for majority voting.")
    aggregate = aggregate_majority_vote(prediction_frames)
    save_majority_vote_outputs(aggregate, paths, method_name="bert_majority_vote")

    if args.probability_averaging:
        averaged = aggregate_probability_average(prediction_frames)
        save_majority_vote_outputs(averaged, paths, method_name="bert_probability_average")


def aggregate_majority_vote(prediction_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-seed predictions by majority vote."""
    label_id_to_name = get_author_id_to_name()
    probability_columns = [probability_column_name(label_id) for label_id in sorted(label_id_to_name)]
    base = prediction_frames[0].copy()
    rows: list[dict[str, Any]] = []

    for _, base_row in base.iterrows():
        sample_id = base_row["sample_id"]
        matching_rows = [frame[frame["sample_id"] == sample_id].iloc[0] for frame in prediction_frames]
        votes = [int(row["predicted_label"]) for row in matching_rows]
        averaged_probs = {
            column: float(np.mean([float(row[column]) for row in matching_rows])) for column in probability_columns
        }
        predicted_label = choose_majority_label(votes, averaged_probs)
        true_label = int(base_row["true_label"])
        sorted_probs = sorted(averaged_probs.values(), reverse=True)
        confidence = float(max(averaged_probs.values()))
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else confidence
        row = base_row.to_dict()
        row.update(averaged_probs)
        row.update(
            {
                "predicted_label": predicted_label,
                "predicted_label_name": label_id_to_name[predicted_label],
                "true_label": true_label,
                "true_label_name": label_id_to_name[true_label],
                "confidence": confidence,
                "margin": margin,
                "vote_count": Counter(votes)[predicted_label],
                "num_votes": len(votes),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_probability_average(prediction_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-seed predictions by averaged probabilities."""
    label_id_to_name = get_author_id_to_name()
    probability_columns = [probability_column_name(label_id) for label_id in sorted(label_id_to_name)]
    base = prediction_frames[0].copy()
    rows: list[dict[str, Any]] = []

    for _, base_row in base.iterrows():
        sample_id = base_row["sample_id"]
        matching_rows = [frame[frame["sample_id"] == sample_id].iloc[0] for frame in prediction_frames]
        averaged_probs = {
            column: float(np.mean([float(row[column]) for row in matching_rows])) for column in probability_columns
        }
        labels = sorted(label_id_to_name)
        values = np.asarray([averaged_probs[probability_column_name(label_id)] for label_id in labels])
        predicted_label = int(labels[int(values.argmax())])
        sorted_probs = sorted(averaged_probs.values(), reverse=True)
        confidence = float(sorted_probs[0])
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else confidence
        true_label = int(base_row["true_label"])
        row = base_row.to_dict()
        row.update(averaged_probs)
        row.update(
            {
                "predicted_label": predicted_label,
                "predicted_label_name": label_id_to_name[predicted_label],
                "true_label": true_label,
                "true_label_name": label_id_to_name[true_label],
                "confidence": confidence,
                "margin": margin,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def choose_majority_label(votes: list[int], averaged_probs: dict[str, float]) -> int:
    """Choose majority label with deterministic tie-breaking."""
    counts = Counter(votes)
    max_count = max(counts.values())
    tied_labels = [label for label, count in counts.items() if count == max_count]
    if len(tied_labels) == 1:
        return int(tied_labels[0])
    best_label = max(
        tied_labels,
        key=lambda label: (averaged_probs[probability_column_name(label)], -label),
    )
    return int(best_label)


def save_majority_vote_outputs(predictions: pd.DataFrame, paths: dict[str, Path], method_name: str) -> None:
    """Save aggregate predictions, metrics, report, and confusion matrix."""
    label_id_to_name = get_author_id_to_name()
    label_ids = sorted(label_id_to_name)
    label_names = get_label_names(label_ids)
    y_true = predictions["true_label"].astype(int).to_numpy()
    y_pred = predictions["predicted_label"].astype(int).to_numpy()

    predictions_path = paths["predictions"] / f"{method_name}_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    metrics = compute_classification_metrics(y_true, y_pred, label_ids, label_names)
    metrics["method"] = method_name
    save_json(metrics, paths["metrics"] / f"{method_name}_metrics.json")

    report = build_classification_report(y_true, y_pred, label_ids, label_names)
    save_classification_report(
        report,
        paths["reports"] / f"{method_name}_classification_report.json",
        paths["reports"] / f"{method_name}_classification_report.csv",
    )

    raw_cm, normalized_cm = build_confusion_matrices(y_true, y_pred, label_ids)
    save_confusion_matrix_csv(raw_cm, label_names, paths["reports"] / f"{method_name}_confusion_matrix.csv")
    save_confusion_matrix_csv(
        normalized_cm,
        label_names,
        paths["reports"] / f"{method_name}_confusion_matrix_normalized.csv",
    )
    save_confusion_matrix_plot(raw_cm, label_names, paths["plots"] / f"{method_name}_confusion_matrix.png")
    save_error_analysis(
        predictions,
        paths["reports"] / f"{method_name}_error_analysis.csv",
        paths["reports"] / f"{method_name}_high_confidence_errors.csv",
        paths["reports"] / f"{method_name}_low_confidence_hard_samples.csv",
    )


if __name__ == "__main__":
    main()
