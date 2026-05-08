"""Error analysis exports for Phase 2 predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return wrong predictions with confidence and margin details."""
    wrong = predictions[predictions["true_label"] != predictions["predicted_label"]].copy()
    if wrong.empty:
        return wrong
    return wrong.sort_values(["confidence", "margin"], ascending=[False, False])


def save_error_analysis(
    predictions: pd.DataFrame,
    output_path: str | Path,
    high_confidence_path: str | Path,
    low_confidence_path: str | Path,
    top_n: int = 25,
) -> None:
    """Save all wrong predictions plus focused high- and low-confidence slices."""
    errors = build_error_analysis(predictions)
    columns = [
        "sample_id",
        "true_label",
        "true_label_name",
        "predicted_label",
        "predicted_label_name",
        "confidence",
        "margin",
        "text_preview",
        "text",
    ]
    existing_columns = [column for column in columns if column in errors.columns]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    errors[existing_columns].to_csv(path, index=False)

    high_path = Path(high_confidence_path)
    high_path.parent.mkdir(parents=True, exist_ok=True)
    errors.head(top_n)[existing_columns].to_csv(high_path, index=False)

    low_path = Path(low_confidence_path)
    low_path.parent.mkdir(parents=True, exist_ok=True)
    errors.sort_values(["confidence", "margin"], ascending=[True, True]).head(top_n)[existing_columns].to_csv(
        low_path,
        index=False,
    )
