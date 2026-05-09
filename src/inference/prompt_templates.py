"""Prompt template loading and rendering for Phase 4."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.label_mapping import get_label_names
from src.inference.prompting_config import PromptingConfig


PROMPT_TEMPLATE_DIR = PROJECT_ROOT / "prompts" / "phase4"
PROMPT_FILES = {
    "zero_shot": "zero_shot.txt",
    "cot_zero_shot": "cot_zero_shot.txt",
    "few_shot": "few_shot.txt",
}
MIN_FEW_SHOT_EXAMPLE_CHARS = 120
MAX_FEW_SHOT_EXAMPLE_CHARS = 500


def render_prompt(config: PromptingConfig, paragraph_text: str, few_shot_examples: list[dict[str, Any]] | None = None) -> str:
    """Render a prompt for one paragraph."""
    template = load_prompt_template(config.prompt_type)
    paragraph = truncate_text(str(paragraph_text), config.max_input_chars)
    examples_text = ""
    if config.prompt_type == "few_shot":
        if not few_shot_examples:
            raise ValueError("few_shot prompt requires examples from few_shot_examples.json.")
        examples_text = format_few_shot_examples(few_shot_examples, config.max_input_chars)

    return template.format(
        author_list=format_author_list(),
        paragraph_text=paragraph,
        few_shot_examples=examples_text,
    )


def load_prompt_template(prompt_type: str) -> str:
    """Load the prompt template for a prompt type."""
    if prompt_type not in PROMPT_FILES:
        raise ValueError(f"Unknown prompt_type {prompt_type!r}.")
    path = PROMPT_TEMPLATE_DIR / PROMPT_FILES[prompt_type]
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def load_few_shot_examples(path: Path) -> list[dict[str, Any]]:
    """Load few-shot examples from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Few-shot examples file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Few-shot examples file has invalid schema: {path}")
    return examples


def format_author_list() -> str:
    """Return a bullet list of canonical authors."""
    return "\n".join(f"- {name}" for name in get_label_names())


def format_few_shot_examples(examples: list[dict[str, Any]], max_input_chars: int) -> str:
    """Format saved examples for insertion into a prompt."""
    lines: list[str] = []
    example_budget = few_shot_example_budget(max_input_chars, len(examples))
    for example in examples:
        author = str(example.get("author", "")).strip()
        text = truncate_text(str(example.get("text", "")), example_budget)
        if not author or not text:
            raise ValueError("Few-shot examples must include non-empty author and text fields.")
        lines.extend(
            [
                f"Text: {text}",
                f"Answer: {author}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def few_shot_example_budget(max_input_chars: int, example_count: int) -> int:
    """Return a compact per-example budget to keep few-shot prompts within small-model contexts."""
    if example_count <= 0:
        return MIN_FEW_SHOT_EXAMPLE_CHARS
    budget = max_input_chars // max(example_count // 2, 1)
    return max(MIN_FEW_SHOT_EXAMPLE_CHARS, min(MAX_FEW_SHOT_EXAMPLE_CHARS, budget))


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to a character budget without cutting mid-word when possible."""
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized
    truncated = normalized[:max_chars].rsplit(" ", 1)[0]
    return truncated if truncated else normalized[:max_chars]
