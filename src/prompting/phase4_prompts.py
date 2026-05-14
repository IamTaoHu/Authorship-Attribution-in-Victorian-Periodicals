"""Prompt construction and few-shot example selection for Phase 4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.prompting.phase4_parser import CANONICAL_AUTHORS


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "prompts" / "phase4"
ZERO_SHOT_TEMPLATE = PROMPT_DIR / "zero_shot_direct.txt"
FEW_SHOT_TEMPLATE = PROMPT_DIR / "few_shot_direct.txt"


def load_template(prompt_type: str) -> str:
    if prompt_type == "zero_shot_direct":
        path = ZERO_SHOT_TEMPLATE
    elif prompt_type == "few_shot_direct":
        path = FEW_SHOT_TEMPLATE
    else:
        raise ValueError(f"Unsupported Phase 4 prompt_type: {prompt_type}")
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def truncate_text(text: Any, max_input_chars: int | None) -> str:
    normalized = str(text or "").strip()
    if max_input_chars is None or max_input_chars <= 0:
        return normalized
    return normalized[: int(max_input_chars)].strip()


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def select_shortest_valid_examples(
    train_frame: pd.DataFrame,
    *,
    min_text_chars: int = 200,
    max_example_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Select one short deterministic training example per canonical author."""

    _require_columns(train_frame, ("sample_id", "text", "author"), "train_frame")
    frame = train_frame.loc[:, ["sample_id", "text", "author"]].copy()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame = frame.loc[frame["text"].ne("")]
    frame["text_length_chars"] = frame["text"].str.len().astype(int)
    frame = frame.loc[frame["text_length_chars"] >= int(min_text_chars)]
    examples: list[dict[str, Any]] = []
    for author in CANONICAL_AUTHORS:
        author_rows = frame.loc[frame["author"].astype(str).eq(author)].copy()
        if author_rows.empty:
            raise ValueError(f"No valid few-shot example found for author: {author}")
        author_rows = author_rows.sort_values(["text_length_chars", "sample_id"], kind="mergesort")
        row = author_rows.iloc[0]
        text = truncate_text(row["text"], max_example_chars)
        examples.append(
            {
                "sample_id": str(row["sample_id"]),
                "author": author,
                "text": text,
                "text_length_chars": int(row["text_length_chars"]),
                "min_text_chars": int(min_text_chars),
            }
        )
    return examples


def load_or_build_few_shot_examples(
    train_frame: pd.DataFrame,
    output_path: Path,
    *,
    rebuild: bool = False,
    min_text_chars: int = 200,
    max_example_chars: int | None = None,
) -> list[dict[str, Any]]:
    if output_path.exists() and not rebuild:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        examples = payload.get("examples", payload)
        if not isinstance(examples, list):
            raise ValueError(f"Invalid few-shot example file: {output_path}")
        authors = [str(example.get("author", "")) for example in examples]
        if authors != list(CANONICAL_AUTHORS):
            raise ValueError(f"Few-shot examples do not match canonical author order: {output_path}")
        return examples

    examples = select_shortest_valid_examples(
        train_frame,
        min_text_chars=min_text_chars,
        max_example_chars=max_example_chars,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_policy": "shortest_valid_text_per_author",
        "split": "train",
        "min_text_chars": int(min_text_chars),
        "max_example_chars": max_example_chars,
        "authors": list(CANONICAL_AUTHORS),
        "examples": examples,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")
    return examples


def format_few_shot_examples(examples: list[dict[str, Any]]) -> str:
    blocks = []
    for index, example in enumerate(examples, start=1):
        author = str(example["author"])
        text = str(example["text"]).strip()
        blocks.append(
            "\n".join(
                [
                    f"Example {index}",
                    f"Paragraph: {text}",
                    f'Output: {{"author": "{author}"}}',
                ]
            )
        )
    return "\n\n".join(blocks)


def build_prompt(
    *,
    text: Any,
    prompt_type: str,
    max_input_chars: int | None,
    few_shot_examples: list[dict[str, Any]] | None = None,
) -> str:
    template = load_template(prompt_type)
    paragraph = truncate_text(text, max_input_chars)
    if prompt_type == "few_shot_direct":
        if not few_shot_examples:
            raise ValueError("few_shot_direct requires few_shot_examples.")
        return template.format(text=paragraph, examples=format_few_shot_examples(few_shot_examples))
    return template.format(text=paragraph)
