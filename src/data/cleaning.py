"""Text cleaning utilities for OCRed historical periodicals."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from datasets import Dataset, DatasetDict


def clean_text(
    text: str,
    lowercase: bool = False,
    normalize_unicode: bool = True,
    normalize_spaces: bool = True,
    remove_control_chars: bool = True,
) -> str:
    """Clean OCRed paragraph text while preserving stylistic attribution signals."""
    if text is None:
        return ""

    cleaned = str(text)
    if normalize_unicode:
        cleaned = unicodedata.normalize("NFKC", cleaned)

    if remove_control_chars:
        cleaned = "".join(
            "\n" if character in {"\n", "\r"} else character
            for character in cleaned
            if character in {"\n", "\r", "\t"} or unicodedata.category(character)[0] != "C"
        )

    if normalize_spaces:
        cleaned = re.sub(r"\r\n?", "\n", cleaned)
        cleaned = re.sub(r"[ \t\f\v]+", " ", cleaned)
        cleaned = re.sub(r" *\n *", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    if lowercase:
        cleaned = cleaned.lower()

    return cleaned.strip()


def clean_dataset(
    dataset: Dataset | DatasetDict,
    text_column: str,
    lowercase: bool = False,
    normalize_unicode: bool = True,
    normalize_spaces: bool = True,
    remove_control_chars: bool = True,
) -> Dataset | DatasetDict:
    """Apply clean_text to a Dataset or DatasetDict text column."""
    def _clean_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        return {
            text_column: [
                clean_text(
                    value,
                    lowercase=lowercase,
                    normalize_unicode=normalize_unicode,
                    normalize_spaces=normalize_spaces,
                    remove_control_chars=remove_control_chars,
                )
                for value in batch[text_column]
            ]
        }

    return dataset.map(_clean_batch, batched=True)
