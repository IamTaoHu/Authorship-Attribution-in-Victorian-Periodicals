"""Run the Phase 1 dataset inspection, cleaning, and tokenization pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import (  # noqa: E402
    inspect_and_save_dataset,
    prepare_output_dirs,
    process_periad_dataset,
    save_dataset_inspection_outputs,
    write_warnings,
)
from src.data.inspect_dataset import validate_periad_splits  # noqa: E402
from src.data.load_datasets import (  # noqa: E402
    MODIFIED_VEAA_DATASET_NAME,
    PERIAD_DATASET_NAME,
    infer_label_column,
    infer_text_column,
    load_modified_veaa,
    load_periad,
)
from src.utils.logging import setup_logger  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse Phase 1 command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Phase 1 dataset preparation pipeline.")
    parser.add_argument("--periad_dataset_name", default=PERIAD_DATASET_NAME)
    parser.add_argument("--modified_veaa_dataset_name", default=MODIFIED_VEAA_DATASET_NAME)
    parser.add_argument("--output_dir", default="outputs/phase1")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_samples_for_token_analysis", type=int, default=2000)
    parser.add_argument("--skip_tokenization", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the full Phase 1 pipeline without model training."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    prepare_output_dirs(output_dir)
    logger = setup_logger("phase1_dataset_pipeline", log_file=str(output_dir / "phase1_dataset_pipeline.log"))
    warnings: list[str] = []

    def warn(message: str) -> None:
        warnings.append(message)
        logger.warning(message)
        write_warnings(warnings, output_dir)

    set_seed(42)
    logger.info("Starting Phase 1 dataset pipeline.")
    logger.info("Output directory: %s", output_dir)

    reports: dict[str, dict] = {}
    distributions = []
    periad_dataset = None

    if not args.periad_dataset_name or args.periad_dataset_name == PERIAD_DATASET_NAME:
        warn("PERIAD dataset name is missing or still TODO. Pass --periad_dataset_name YOUR_PERIAD_HF_NAME.")
    else:
        periad_dataset = _try_load_periad(args.periad_dataset_name, args.cache_dir, warn)
        if periad_dataset is not None:
            logger.info("Loaded PERIAD dataset: %s", args.periad_dataset_name)
            split_warnings = validate_periad_splits(periad_dataset)
            for split_warning in split_warnings:
                warn(split_warning)
            report, distribution = inspect_and_save_dataset(periad_dataset, "PERIAD")
            reports["PERIAD"] = report
            distributions.append(distribution)

            text_column = infer_text_column(periad_dataset)
            label_column = infer_label_column(periad_dataset)
            logger.info("PERIAD inferred text column: %s", text_column)
            logger.info("PERIAD inferred label column: %s", label_column)

            if text_column is None:
                warn("PERIAD text column could not be inferred. Skipping PERIAD cleaning and text analysis.")
            if label_column is None:
                warn("PERIAD label column could not be inferred. Skipping PERIAD label mapping and class plots.")
            if text_column is not None and label_column is not None:
                process_periad_dataset(
                    periad_dataset,
                    text_column=text_column,
                    label_column=label_column,
                    output_dir=output_dir,
                    skip_tokenization=args.skip_tokenization,
                    max_samples_for_token_analysis=args.max_samples_for_token_analysis,
                    warn=warn,
                )

    modified_veaa_dataset = _try_load_modified_veaa(args.modified_veaa_dataset_name, args.cache_dir, warn)
    if modified_veaa_dataset is not None:
        logger.info("Loaded ModifiedVEAA dataset: %s", args.modified_veaa_dataset_name)
        report, distribution = inspect_and_save_dataset(modified_veaa_dataset, "ModifiedVEAA")
        reports["ModifiedVEAA"] = report
        distributions.append(distribution)
        logger.info("ModifiedVEAA inferred text column: %s", infer_text_column(modified_veaa_dataset))
        logger.info("ModifiedVEAA inferred label column: %s", infer_label_column(modified_veaa_dataset))

    save_dataset_inspection_outputs(reports, distributions, output_dir)
    write_warnings(warnings, output_dir)

    logger.info("Saved dataset report: %s", output_dir / "dataset_report.json")
    logger.info("Saved class distribution: %s", output_dir / "class_distribution.csv")
    logger.info("Saved warnings: %s", output_dir / "warnings.txt")
    logger.info("Phase 1 dataset pipeline complete.")


def _try_load_periad(dataset_name: str, cache_dir: Optional[str], warn: Callable[[str], None]):
    """Load PERIAD while converting failures into pipeline warnings."""
    try:
        return load_periad(dataset_name, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001 - continue when one HF dataset is unavailable.
        warn(f"Failed to load PERIAD dataset '{dataset_name}': {exc}")
        return None


def _try_load_modified_veaa(dataset_name: str, cache_dir: Optional[str], warn: Callable[[str], None]):
    """Load ModifiedVEAA while converting failures into pipeline warnings."""
    try:
        return load_modified_veaa(cache_dir=cache_dir, dataset_name=dataset_name)
    except Exception as exc:  # noqa: BLE001 - continue when one HF dataset is unavailable.
        warn(f"Failed to load ModifiedVEAA dataset '{dataset_name}': {exc}")
        return None


if __name__ == "__main__":
    main()
