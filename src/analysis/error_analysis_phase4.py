"""Error analysis for Phase 4 decoder prompting predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import traceback
from typing import Any
import warnings

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_COLUMNS = [
    "sample_id",
    "text",
    "true_label",
    "true_author",
    "predicted_author",
    "raw_output",
    "parsed_status",
    "prompt_type",
    "model_name",
    "input_tokens",
    "output_tokens",
    "inference_time_sec",
]
REQUIRED_COLUMNS = {"true_author"}
EXPORT_COLUMNS = [
    "sample_id",
    "true_author",
    "predicted_author",
    "parsed_status",
    "prompt_type",
    "model_name",
    "raw_output",
    "text",
]
SUMMARY_COLUMNS = [
    "model_name",
    "prompt_type",
    "n_samples",
    "mean_raw_output_words",
    "median_raw_output_words",
    "max_raw_output_words",
    "invalid_output_rate",
    "verbose_output_rate",
    "accuracy",
]
VERBOSE_MARKERS = (
    "because",
    "i think",
    "the author is",
    "reason",
    "style",
    "therefore",
)
TOPIC_MARKERS = (
    "topic",
    "subject",
    "theme",
    "content",
    "about",
    "politics",
)
AUTHOR_LIKE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+){1,3}\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 decoder prompting error analysis.")
    parser.add_argument("--run_dir", type=Path, default=None, help="Analyze one Phase 4 run directory.")
    parser.add_argument("--phase4_dir", type=Path, default=None, help="Analyze all Phase 4 run directories.")
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_valid_authors() -> list[str]:
    return [
        "Leslie Stephen",
        "John Morley",
        "Eliza Lynn Linton",
        "George Henry Lewes",
        "Anne Mozley",
        "James Fitzjames Stephen",
    ]


def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_prediction_columns(df: pd.DataFrame) -> None:
    missing_required = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_required:
        raise ValueError(f"predictions.csv is missing required columns: {missing_required}")
    missing_optional = sorted(set(EXPECTED_COLUMNS) - REQUIRED_COLUMNS - set(df.columns))
    if missing_optional:
        warnings.warn(
            f"predictions.csv is missing optional columns: {missing_optional}; placeholder values will be used.",
            RuntimeWarning,
            stacklevel=2,
        )


def add_error_columns(df: pd.DataFrame, valid_authors: list[str]) -> pd.DataFrame:
    frame = df.copy()
    for column in EXPECTED_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    valid_set = set(valid_authors)
    normalized_true = frame["true_author"].map(normalize_author)
    normalized_pred = frame["predicted_author"].map(normalize_author)
    parsed_status = frame["parsed_status"].fillna("").astype(str)
    raw_output = frame["raw_output"].fillna("").astype(str)

    frame["true_author"] = normalized_true
    frame["predicted_author"] = normalized_pred
    frame["correct"] = normalized_pred.eq(normalized_true)
    frame["raw_output_chars"] = raw_output.str.len()
    frame["raw_output_words"] = raw_output.map(count_words)
    frame["is_valid_prediction"] = normalized_pred.isin(valid_set) & parsed_status.map(is_success_status)
    frame["is_invalid_output"] = (
        ~frame["is_valid_prediction"]
        | normalized_pred.eq("")
        | normalized_pred.str.upper().isin({"INVALID", "__INVALID__"})
        | ~normalized_pred.isin(valid_set)
    )
    frame["is_verbose_output"] = (frame["raw_output_words"] > 20) | raw_output.map(has_verbose_marker)
    frame["is_hallucinated_author"] = frame.apply(
        lambda row: bool(row["is_invalid_output"]) and has_hallucinated_author(row, valid_authors),
        axis=1,
    )
    frame["mentions_topic_bias"] = raw_output.map(has_topic_marker)
    return frame


def write_false_positives(df: pd.DataFrame, output_dir: Path) -> None:
    errors = df[~df["correct"].astype(bool)].copy()
    false_positives = errors[errors["predicted_author"].isin(get_valid_authors())].copy()
    false_positives = false_positives.sort_values(["predicted_author", "true_author", "sample_id"], na_position="last")
    save_columns(false_positives, output_dir / "false_positives.csv", EXPORT_COLUMNS)


def write_false_negatives(df: pd.DataFrame, output_dir: Path) -> None:
    false_negatives = df[~df["correct"].astype(bool)].copy()
    false_negatives = false_negatives.sort_values(["true_author", "predicted_author", "sample_id"], na_position="last")
    save_columns(false_negatives, output_dir / "false_negatives.csv", EXPORT_COLUMNS)


def write_hard_author_pairs(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    errors = df[~df["correct"].astype(bool)].copy()
    if errors.empty:
        hard_pairs = pd.DataFrame(columns=["true_author", "predicted_author", "count", "percentage_of_all_errors"])
    else:
        hard_pairs = (
            errors.groupby(["true_author", "predicted_author"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        hard_pairs["percentage_of_all_errors"] = hard_pairs["count"] / len(errors) * 100.0
    hard_pairs.to_csv(output_dir / "hard_author_pairs.csv", index=False)
    return hard_pairs


def write_invalid_outputs(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    invalid_df = df[df["is_invalid_output"].astype(bool)].copy()
    save_columns(invalid_df, output_dir / "invalid_outputs.csv", EXPORT_COLUMNS + ["is_hallucinated_author"])
    return invalid_df


def write_verbose_outputs(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    verbose_df = df[df["is_verbose_output"].astype(bool)].copy()
    save_columns(verbose_df, output_dir / "verbose_outputs.csv", EXPORT_COLUMNS + ["raw_output_chars", "raw_output_words"])
    return verbose_df


def write_hallucinated_authors(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    hallucinated_df = df[df["is_hallucinated_author"].astype(bool)].copy()
    save_columns(hallucinated_df, output_dir / "hallucinated_authors.csv", EXPORT_COLUMNS + ["raw_output_chars", "raw_output_words"])
    return hallucinated_df


def write_cot_reasoning_samples(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    prompt_type = df["prompt_type"].fillna("").astype(str).str.lower()
    cot_df = df[prompt_type.str.contains("cot", na=False)].copy()
    if not cot_df.empty:
        cot_df["priority"] = (
            (~cot_df["correct"].astype(bool)).astype(int) * 8
            + cot_df["is_invalid_output"].astype(int) * 4
            + cot_df["is_verbose_output"].astype(int) * 2
            + cot_df["mentions_topic_bias"].astype(int)
        )
        cot_df = cot_df.sort_values(["priority", "raw_output_words"], ascending=[False, False])
    columns = [
        "sample_id",
        "true_author",
        "predicted_author",
        "correct",
        "parsed_status",
        "raw_output",
        "text",
    ]
    save_columns(cot_df, output_dir / "cot_reasoning_samples.csv", columns)
    return cot_df.drop(columns=["priority"], errors="ignore")


def write_output_length_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if df.empty:
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
    else:
        summary = (
            df.groupby(["model_name", "prompt_type"], dropna=False)
            .agg(
                n_samples=("sample_id", "size"),
                mean_raw_output_words=("raw_output_words", "mean"),
                median_raw_output_words=("raw_output_words", "median"),
                max_raw_output_words=("raw_output_words", "max"),
                invalid_output_rate=("is_invalid_output", "mean"),
                verbose_output_rate=("is_verbose_output", "mean"),
                accuracy=("correct", "mean"),
            )
            .reset_index()
        )
    summary.to_csv(output_dir / "output_length_summary.csv", index=False)
    return summary


def write_error_summary(
    df: pd.DataFrame,
    output_dir: Path,
    run_name: str,
    hard_pairs: pd.DataFrame,
    invalid_df: pd.DataFrame,
    verbose_df: pd.DataFrame,
) -> None:
    total = len(df)
    errors = df[~df["correct"].astype(bool)].copy()
    accuracy = float(df["correct"].mean()) if total else 0.0
    invalid_rate = float(df["is_invalid_output"].mean()) if total else 0.0
    verbose_rate = float(df["is_verbose_output"].mean()) if total else 0.0
    model_names = ", ".join(unique_nonempty(df["model_name"])) or "unknown"
    prompt_types = ", ".join(unique_nonempty(df["prompt_type"])) or "unknown"
    hallucinated_df = df[df["is_hallucinated_author"].astype(bool)].copy()

    lines = [
        f"# Phase 4 Error Summary: {run_name}",
        "",
        f"- Run name: {run_name}",
        f"- Model name: {model_names}",
        f"- Prompt type: {prompt_types}",
        f"- Number of samples: {total}",
        f"- Accuracy: {accuracy:.4f}",
        f"- Number of errors: {len(errors)}",
        f"- Invalid output rate: {invalid_rate:.4f}",
        f"- Verbose output rate: {verbose_rate:.4f}",
        "",
        "## Top Hard Author Pairs",
        "",
        table_to_markdown(hard_pairs.head(10)),
        "",
        "## Top Hallucinated / Invalid Examples",
        "",
        table_to_markdown(example_rows(pd.concat([hallucinated_df, invalid_df], ignore_index=True).drop_duplicates("sample_id")).head(10)),
        "",
        "## Verbose Output Examples",
        "",
        table_to_markdown(example_rows(verbose_df).head(10)),
        "",
        "## Qualitative Notes",
        "",
    ]
    lines.extend(build_qualitative_notes(df, errors, hard_pairs, invalid_df, verbose_df))
    output_dir.joinpath("error_summary.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_run(run_dir: Path) -> None:
    run_dir = resolve_repo_path(run_dir)
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.exists():
        warnings.warn(f"Skipping {run_dir}: predictions.csv not found.", RuntimeWarning, stacklevel=2)
        return

    print(f"[INFO] Processing Phase 4 run: {run_dir.name}")
    output_dir = run_dir / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_authors = get_valid_authors()
    predictions = load_predictions(predictions_path)
    validate_prediction_columns(predictions)
    predictions = add_error_columns(predictions, valid_authors)

    write_false_positives(predictions, output_dir)
    write_false_negatives(predictions, output_dir)
    hard_pairs = write_hard_author_pairs(predictions, output_dir)
    invalid_df = write_invalid_outputs(predictions, output_dir)
    verbose_df = write_verbose_outputs(predictions, output_dir)
    write_hallucinated_authors(predictions, output_dir)
    write_cot_reasoning_samples(predictions, output_dir)
    write_output_length_summary(predictions, output_dir)
    write_error_summary(predictions, output_dir, run_dir.name, hard_pairs, invalid_df, verbose_df)
    print(f"[OK] Wrote Phase 4 error analysis to {output_dir}")


def main() -> None:
    args = parse_args()
    if args.run_dir is not None:
        try:
            analyze_run(args.run_dir)
        except Exception:
            print(f"[ERROR] Failed error analysis for {args.run_dir}")
            traceback.print_exc()
        return

    phase4_dir = resolve_repo_path(args.phase4_dir or Path("outputs/phase4"))
    if not phase4_dir.exists():
        warnings.warn(f"Phase 4 directory does not exist: {phase4_dir}", RuntimeWarning, stacklevel=2)
        return

    prediction_paths = sorted(phase4_dir.glob("*/predictions.csv"))
    if not prediction_paths:
        print(f"[WARN] No Phase 4 predictions.csv files found under {phase4_dir}")
        return

    for predictions_path in prediction_paths:
        try:
            analyze_run(predictions_path.parent)
        except Exception:
            print(f"[ERROR] Failed error analysis for {predictions_path.parent.name}")
            traceback.print_exc()


def save_columns(frame: pd.DataFrame, output_path: Path, columns: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame.loc[:, columns].to_csv(output_path, index=False)


def normalize_author(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def count_words(value: Any) -> int:
    return len(str(value or "").split())


def is_success_status(value: Any) -> bool:
    return str(value or "").strip().lower().startswith("ok")


def has_verbose_marker(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in VERBOSE_MARKERS)


def has_topic_marker(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in TOPIC_MARKERS)


def has_hallucinated_author(row: pd.Series, valid_authors: list[str]) -> bool:
    status = str(row.get("parsed_status", "") or "").lower()
    if not any(marker in status for marker in ("invalid", "unknown", "no_match", "no author", "no_author")):
        return False

    raw_output = str(row.get("raw_output", "") or "")
    normalized_output = normalize_for_match(raw_output)
    if any(normalize_for_match(author) in normalized_output for author in valid_authors):
        return False

    ignored = {
        "Final Answer",
        "The Author",
        "Valid Authors",
        "Target Text",
        "Classify This",
    }
    for candidate in AUTHOR_LIKE_RE.findall(raw_output):
        if candidate in ignored:
            continue
        normalized_candidate = normalize_for_match(candidate)
        if normalized_candidate and normalized_candidate not in {normalize_for_match(author) for author in valid_authors}:
            return True
    return False


def normalize_for_match(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def unique_nonempty(series: pd.Series) -> list[str]:
    return [str(value).strip() for value in series.dropna().unique() if str(value).strip()]


def example_rows(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["sample_id", "true_author", "predicted_author", "parsed_status", "raw_output", "text"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    examples = frame.copy()
    examples["raw_output"] = examples["raw_output"].map(lambda value: truncate_text(value, 160))
    examples["text"] = examples["text"].map(lambda value: truncate_text(value, 180))
    return examples.loc[:, columns]


def truncate_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_qualitative_notes(
    df: pd.DataFrame,
    errors: pd.DataFrame,
    hard_pairs: pd.DataFrame,
    invalid_df: pd.DataFrame,
    verbose_df: pd.DataFrame,
) -> list[str]:
    total = len(df)
    notes: list[str] = []
    invalid_count = len(invalid_df)
    error_count = len(errors)
    if invalid_count:
        share = invalid_count / error_count if error_count else 0.0
        notes.append(f"- Parser failures are present: {invalid_count} invalid outputs ({share:.2%} of errors).")
    else:
        notes.append("- Parser failures are not evident in this run.")

    valid_wrong = errors[~errors["is_invalid_output"].astype(bool)]
    if not valid_wrong.empty and not hard_pairs.empty:
        top_pair = hard_pairs.iloc[0]
        notes.append(
            "- Model confusion is visible in valid wrong predictions; "
            f"top pair is {top_pair['true_author']} -> {top_pair['predicted_author']} ({int(top_pair['count'])} cases)."
        )
    else:
        notes.append("- Model confusion among valid author labels is limited or absent.")

    cot_rows = df[df["prompt_type"].fillna("").astype(str).str.lower().str.contains("cot", na=False)]
    if cot_rows.empty:
        notes.append("- CoT-specific behavior is not assessable because this run has no CoT prompt rows.")
    else:
        cot_invalid_rate = float(cot_rows["is_invalid_output"].mean())
        cot_verbose_rate = float(cot_rows["is_verbose_output"].mean())
        cot_accuracy = float(cot_rows["correct"].mean())
        notes.append(
            "- CoT rows show "
            f"accuracy {cot_accuracy:.4f}, invalid rate {cot_invalid_rate:.4f}, and verbose rate {cot_verbose_rate:.4f}."
        )

    topic_rows = df[df["mentions_topic_bias"].astype(bool)]
    if not topic_rows.empty:
        notes.append(f"- Topic-bias markers appear in {len(topic_rows)} outputs; inspect CoT/verbose samples for topic-over-style reasoning.")
    else:
        notes.append("- Topic-bias markers are not prominent in raw outputs.")

    if verbose_df.empty:
        notes.append("- Verbose explanation-style outputs are not prominent.")
    else:
        notes.append(f"- Verbose outputs are present in {len(verbose_df)} samples and may require stricter prompt or parser controls.")

    if total == 0:
        notes.append("- This predictions file is empty; all rates are reported as zero.")
    return notes


def table_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = [format_markdown_value(row[column]) for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
