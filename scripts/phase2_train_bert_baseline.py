"""Run the Phase 2 BERT-base baseline on PERIAD."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_transformer_classifier import (  # noqa: E402
    TransformerTrainingConfig,
    run_transformer_baseline,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate the Phase 2 BERT-base baseline.")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run one baseline experiment."""
    args = parse_args()
    config = TransformerTrainingConfig(
        model_name=args.model_name,
        seed=args.seed,
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
        run_name=args.run_name,
    )
    run_transformer_baseline(config)


if __name__ == "__main__":
    main()
