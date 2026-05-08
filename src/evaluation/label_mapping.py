"""Author label mapping utilities for PERIAD evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


AUTHOR_ID_TO_NAME: dict[int, str] = {
    0: "Leslie Stephen",
    1: "John Morley",
    2: "Eliza Lynn Linton",
    3: "George Henry Lewes",
    4: "Anne Mozley",
    5: "James Fitzjames Stephen",
}


def get_author_id_to_name() -> dict[int, str]:
    """Return the canonical Phase 2 PERIAD author mapping."""
    return dict(AUTHOR_ID_TO_NAME)


def get_label_names(label_ids: list[int] | None = None) -> list[str]:
    """Return author display names ordered by numeric label ID."""
    ids = sorted(AUTHOR_ID_TO_NAME) if label_ids is None else label_ids
    return [AUTHOR_ID_TO_NAME.get(int(label_id), f"label_{label_id}") for label_id in ids]


def get_label_to_author_name() -> dict[str, str]:
    """Return a JSON-friendly label-to-author-name mapping."""
    return {str(label_id): name for label_id, name in sorted(AUTHOR_ID_TO_NAME.items())}


def save_author_mapping(output_path: str | Path) -> None:
    """Save the canonical author mapping to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "explicit_phase2_mapping",
        "label_to_author_name": get_label_to_author_name(),
        "note": "Numeric labels are preserved for training; author names are used for reports and plots.",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_author_labels(labels: list[Any]) -> list[str]:
    """Return warnings for labels not covered by the canonical mapping."""
    warnings: list[str] = []
    unique_labels = sorted({int(label) for label in labels})
    expected = sorted(AUTHOR_ID_TO_NAME)
    if unique_labels != expected:
        warnings.append(
            f"Observed labels {unique_labels} do not exactly match expected PERIAD labels {expected}."
        )
    missing_names = [label for label in unique_labels if label not in AUTHOR_ID_TO_NAME]
    if missing_names:
        warnings.append(f"No author names configured for labels: {missing_names}.")
    return warnings


def probability_column_name(label_id: int) -> str:
    """Build a stable probability column name for a label."""
    author = AUTHOR_ID_TO_NAME.get(int(label_id), f"label_{label_id}")
    safe_author = author.lower().replace(" ", "_").replace("-", "_")
    return f"prob_{int(label_id)}_{safe_author}"
