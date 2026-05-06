"""Smoke test for Phase 0 project setup utilities."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.experiment import build_experiment_name, get_experiment_paths
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    config = load_config(str(config_path))

    seed = config["experiment"]["seed"]
    set_seed(seed)

    experiment_name = build_experiment_name(config)
    experiment_paths = get_experiment_paths(config)

    log_file = Path(experiment_paths["logs_dir"]) / "check_setup.log"
    logger = setup_logger("check_setup", log_file=str(log_file))

    logger.info("Loaded config path: %s", config_path)
    logger.info("Seed: %s", seed)
    logger.info("Experiment name: %s", experiment_name)
    logger.info("Experiment paths:")
    for key, value in experiment_paths.items():
        logger.info("  %s: %s", key, value)


if __name__ == "__main__":
    main()
