"""Path configuration helpers for local and Colab workflows."""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_CONFIG = REPO_ROOT / "configs" / "paths.local.yaml"
DEFAULT_EXAMPLE_CONFIG = REPO_ROOT / "configs" / "paths.example.yaml"
PLACEHOLDER_PREFIX = "/path/to/"


def _resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        if DEFAULT_LOCAL_CONFIG.exists():
            return DEFAULT_LOCAL_CONFIG

        warnings.warn(
            (
                f"{DEFAULT_LOCAL_CONFIG} was not found; falling back to "
                f"{DEFAULT_EXAMPLE_CONFIG}. Directory-producing helpers will "
                "refuse to create placeholder paths."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return DEFAULT_EXAMPLE_CONFIG

    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _as_path_dict(config: dict[str, Any]) -> dict[str, Path]:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Path config must contain a 'paths' mapping.")

    required = (
        "workspace_root",
        "repo_root",
        "artifacts_root",
        "checkpoints_root",
        "datasets_root",
        "exports_root",
    )
    missing = [key for key in required if key not in paths]
    if missing:
        raise ValueError(f"Path config is missing required keys: {', '.join(missing)}")

    return {key: Path(paths[key]).expanduser() for key in required}


def _contains_placeholder(path: Path) -> bool:
    normalized = path.as_posix()
    return normalized.startswith(PLACEHOLDER_PREFIX) or PLACEHOLDER_PREFIX in normalized


def _ensure_no_placeholders(paths: dict[str, Path]) -> None:
    placeholder_keys = [key for key, value in paths.items() if _contains_placeholder(value)]
    if placeholder_keys:
        keys = ", ".join(placeholder_keys)
        raise ValueError(
            "Refusing to create directories from placeholder path config values "
            f"({keys}). Create configs/paths.local.yaml with real paths or pass "
            "a concrete config_path."
        )


def _join(base: Path, parts: tuple[Any, ...]) -> Path:
    path = base
    for part in parts:
        path = path / str(part)
    return path


def load_paths(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the configured path YAML.

    By default this loads configs/paths.local.yaml. If that file is absent, it
    falls back to configs/paths.example.yaml and warns clearly.
    """

    resolved_config = _resolve_config_path(config_path)
    if not resolved_config.exists():
        raise FileNotFoundError(f"Path config not found: {resolved_config}")

    with resolved_config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Path config must be a YAML mapping: {resolved_config}")

    config["_config_path"] = str(resolved_config)
    return config


def get_path_config(config_path: str | Path | None = None) -> dict[str, Path]:
    """Return configured root paths as pathlib.Path objects."""

    return _as_path_dict(load_paths(config_path))


def phase_artifact_dir(
    phase_name: str,
    *parts: Any,
    config_path: str | Path | None = None,
) -> Path:
    paths = get_path_config(config_path)
    _ensure_no_placeholders(paths)
    path = _join(paths["artifacts_root"] / str(phase_name), parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_dir(
    experiment_name: str,
    *parts: Any,
    config_path: str | Path | None = None,
) -> Path:
    paths = get_path_config(config_path)
    _ensure_no_placeholders(paths)
    path = _join(paths["checkpoints_root"] / str(experiment_name), parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_dir(*parts: Any, config_path: str | Path | None = None) -> Path:
    paths = get_path_config(config_path)
    _ensure_no_placeholders(paths)
    path = _join(paths["datasets_root"], parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_dir(*parts: Any, config_path: str | Path | None = None) -> Path:
    paths = get_path_config(config_path)
    _ensure_no_placeholders(paths)
    path = _join(paths["exports_root"], parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_path(*parts: Any, config_path: str | Path | None = None) -> Path:
    paths = get_path_config(config_path)
    return _join(paths["repo_root"], parts)
