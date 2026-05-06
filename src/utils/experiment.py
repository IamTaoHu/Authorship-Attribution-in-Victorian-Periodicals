"""Experiment naming and output path utilities."""

import re
from pathlib import Path
from typing import Any


def _safe_name(value: str) -> str:
    value = value.lower().replace("/", "-").replace("\\", "-")
    value = value.replace(" ", "")
    return re.sub(r"[^a-z0-9_-]+", "", value)


def _format_learning_rate(learning_rate: Any) -> str:
    return str(learning_rate).lower().replace(".", "").replace("-", "")


def _derive_model_short_name(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1]


def build_experiment_name(config: dict[str, Any]) -> str:
    """Build a filesystem-safe experiment name from config values."""
    model_config = config.get("model", {})
    experiment_config = config.get("experiment", {})
    training_config = config.get("training", {})

    model_short_name = model_config.get("model_short_name")
    if not model_short_name:
        model_short_name = _derive_model_short_name(str(model_config.get("model_name", "model")))

    seed = experiment_config.get("seed", 42)
    learning_rate = _format_learning_rate(training_config.get("learning_rate", "lr"))
    epochs = training_config.get("epochs", "epochs")

    return _safe_name(f"{model_short_name}_seed{seed}_lr{learning_rate}_epoch{epochs}")


def get_experiment_paths(config: dict[str, Any]) -> dict[str, str]:
    """Return and create the output directories for an experiment."""
    experiment_config = config.get("experiment", {})
    output_dir = Path(experiment_config.get("output_dir", "outputs"))
    experiment_name = build_experiment_name(config)
    experiment_dir = output_dir / experiment_name

    paths = {
        "experiment_dir": experiment_dir,
        "checkpoints_dir": experiment_dir / "checkpoints",
        "logs_dir": experiment_dir / "logs",
        "metrics_dir": experiment_dir / "metrics",
        "plots_dir": experiment_dir / "plots",
        "predictions_dir": experiment_dir / "predictions",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return {key: str(path) for key, path in paths.items()}
