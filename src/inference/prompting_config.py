"""Configuration loading for Phase 4 decoder prompting runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


REQUIRED_FIELDS = {
    "model_name",
    "prompt_type",
    "max_new_tokens",
    "temperature",
    "do_sample",
    "top_p",
    "few_shot_examples_path",
    "batch_size",
    "output_dir",
    "use_4bit",
    "max_input_chars",
    "max_samples",
    "seed",
}
VALID_PROMPT_TYPES = {"zero_shot", "cot_zero_shot", "few_shot"}


@dataclass(frozen=True)
class PromptingConfig:
    """Resolved configuration for one Phase 4 prompting run."""

    model_name: str
    prompt_type: str
    max_new_tokens: int
    temperature: float
    do_sample: bool
    top_p: float
    few_shot_examples_path: Path
    batch_size: int
    output_dir: Path
    use_4bit: bool
    max_input_chars: int
    max_samples: int | None
    seed: int

    @property
    def model_short(self) -> str:
        return model_short_name(self.model_name)

    @property
    def run_name(self) -> str:
        return f"{self.model_short}_{self.prompt_type}_seed{self.seed}"

    @property
    def output_run_dir(self) -> Path:
        return self.output_dir / self.run_name


def load_prompting_config(config_path: str | Path) -> PromptingConfig:
    """Load, validate, and resolve a Phase 4 YAML config."""
    path = resolve_repo_path(config_path)
    raw = load_config(str(path))
    return resolve_prompting_config(raw)


def resolve_prompting_config(raw: dict[str, Any]) -> PromptingConfig:
    """Validate a raw config mapping and return a typed config."""
    raw = dict(raw)
    if "do_sample" not in raw:
        raw["do_sample"] = False if float(raw.get("temperature", 0.0)) == 0.0 else True

    missing = sorted(REQUIRED_FIELDS - set(raw))
    if missing:
        raise ValueError(f"Phase 4 config is missing required fields: {missing}")
    extra = sorted(set(raw) - REQUIRED_FIELDS)
    if extra:
        raise ValueError(f"Phase 4 config contains unsupported fields: {extra}")

    prompt_type = str(raw["prompt_type"])
    if prompt_type not in VALID_PROMPT_TYPES:
        raise ValueError(f"prompt_type must be one of {sorted(VALID_PROMPT_TYPES)}; got {prompt_type!r}.")

    max_samples = raw["max_samples"]
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("max_samples must be null or a positive integer.")

    batch_size = int(raw["batch_size"])
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    max_new_tokens = int(raw["max_new_tokens"])
    max_input_chars = int(raw["max_input_chars"])
    if max_new_tokens <= 0 or max_input_chars <= 0:
        raise ValueError("max_new_tokens and max_input_chars must be positive integers.")

    return PromptingConfig(
        model_name=str(raw["model_name"]),
        prompt_type=prompt_type,
        max_new_tokens=max_new_tokens,
        temperature=float(raw["temperature"]),
        do_sample=bool(raw["do_sample"]),
        top_p=float(raw["top_p"]),
        few_shot_examples_path=resolve_repo_path(raw["few_shot_examples_path"]),
        batch_size=batch_size,
        output_dir=resolve_repo_path(raw["output_dir"]),
        use_4bit=bool(raw["use_4bit"]),
        max_input_chars=max_input_chars,
        max_samples=max_samples,
        seed=int(raw["seed"]),
    )


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve repo-relative paths while leaving absolute paths unchanged."""
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def model_short_name(model_name: str) -> str:
    """Return stable short model names for run directories."""
    lowered = model_name.lower()
    if "tinyllama" in lowered:
        return "smoke_tinyllama"
    if "mistral" in lowered:
        return "mistral"
    if "llama" in lowered:
        return "llama"
    if "gemma" in lowered:
        return "gemma"
    return model_name.rstrip("/").split("/")[-1].lower().replace("-", "_").replace(".", "_")
