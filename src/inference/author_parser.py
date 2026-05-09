"""Parse decoder author predictions into canonical PERIAD labels."""

from __future__ import annotations

import re
from typing import Any

from src.evaluation.label_mapping import get_author_id_to_name


AUTHOR_ID_TO_NAME = get_author_id_to_name()
AUTHOR_NAME_TO_ID = {name: label for label, name in AUTHOR_ID_TO_NAME.items()}

ALIAS_TO_AUTHOR: dict[str, str] = {
    "leslie stephen": "Leslie Stephen",
    "john morley": "John Morley",
    "morley": "John Morley",
    "eliza lynn linton": "Eliza Lynn Linton",
    "eliza linton": "Eliza Lynn Linton",
    "lynn linton": "Eliza Lynn Linton",
    "linton": "Eliza Lynn Linton",
    "george henry lewes": "George Henry Lewes",
    "georges lewes": "George Henry Lewes",
    "george lewes": "George Henry Lewes",
    "g h lewes": "George Henry Lewes",
    "gh lewes": "George Henry Lewes",
    "lewes": "George Henry Lewes",
    "anne mozley": "Anne Mozley",
    "mozley": "Anne Mozley",
    "james fitzjames stephen": "James Fitzjames Stephen",
    "fitzjames stephen": "James Fitzjames Stephen",
    "james stephen": "James Fitzjames Stephen",
}

FULL_NAME_ALIASES = {name.lower(): name for name in AUTHOR_NAME_TO_ID}
FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*:\s*(.+)", flags=re.IGNORECASE | re.DOTALL)
AUTHOR_LIKE_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+){1,3})\b"
)
PROMPT_CONTROL_PREFIXES = (
    "answer:",
    "author:",
    "classify this target text",
    "example:",
    "examples:",
    "explanation:",
    "final answer:",
    "label:",
    "paragraph:",
    "target:",
    "task:",
    "text:",
    "valid labels:",
)
INSTRUCTION_ECHO_PHRASES = (
    "only use the author s name",
    "only use the author's name",
    "reply using only the exact author name",
    "use a clear and concise sentence structure",
    "choose one author from this list only",
    "do not explain",
    "do not repeat the prompt",
    "do not copy examples",
    "do not output anything except one author name",
    "answer with only one exact author name",
)


def parse_author_prediction(output_text: str) -> dict[str, Any]:
    """Parse model output into a canonical author label and parser status."""
    text = str(output_text or "").strip()
    if not text:
        return invalid("invalid_no_author", "empty output")

    final_text = extract_final_answer_text(text)
    if final_text is not None:
        final_matches = find_author_matches(final_text)
        if len(final_matches) == 1:
            return parsed(final_matches[0], "ok_final_answer", "parsed from Final answer field")
        if len(final_matches) > 1:
            return invalid("invalid_multiple_authors", "multiple valid authors found after Final answer")
        hallucination = detect_hallucinated_author(final_text)
        if hallucination:
            return invalid("invalid_hallucinated_author", f"unknown author-like name after Final answer: {hallucination}")
        if is_instruction_echo(final_text):
            return invalid_instruction_echo("Final answer field repeated prompt instructions")
        return invalid("invalid_no_author", "Final answer field did not contain a valid author")

    matches = find_author_matches(text)
    unique_authors = unique_in_order(matches)
    if len(unique_authors) > 1:
        return invalid("invalid_multiple_authors", "multiple valid authors found in output")
    if len(unique_authors) == 1:
        author = unique_authors[0]
        status = "ok_full_name" if contains_full_name(text, author) else "ok_last_name"
        return parsed(author, status, "parsed canonical author mention")

    hallucination = detect_hallucinated_author(text)
    if hallucination:
        return invalid("invalid_hallucinated_author", f"unknown author-like name: {hallucination}")
    if is_instruction_echo(text):
        return invalid_instruction_echo("output repeated prompt instructions instead of an author")
    return invalid("invalid_no_author", "no valid author found")


def extract_final_answer_text(text: str) -> str | None:
    """Return text after the last Final answer marker, limited to its first line."""
    matches = list(FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return None
    answer = matches[-1].group(1).strip()
    return answer.splitlines()[0].strip()


def find_author_matches(text: str) -> list[str]:
    """Find canonical authors mentioned by known aliases."""
    normalized = normalize_text(text)
    found: list[tuple[int, str, int]] = []
    for alias, author in ALIAS_TO_AUTHOR.items():
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")
        match = pattern.search(normalized)
        if match:
            found.append((match.start(), author, len(alias)))
    for alias, author in FULL_NAME_ALIASES.items():
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")
        match = pattern.search(normalized)
        if match:
            found.append((match.start(), author, len(alias)))
    found.sort(key=lambda item: (item[0], -item[2]))
    return unique_in_order([author for _, author, _ in found])


def contains_full_name(text: str, author: str) -> bool:
    """Return whether the output contains the exact canonical full name after normalization."""
    normalized = normalize_text(text)
    return normalize_text(author) in normalized


def detect_hallucinated_author(text: str) -> str | None:
    """Detect unknown proper-name answers where possible."""
    candidates = [candidate.strip() for candidate in AUTHOR_LIKE_RE.findall(text)]
    valid_names = set(AUTHOR_NAME_TO_ID)
    ignored = {"Final Answer", "The Author", "Valid Authors", "Target Text", "Classify This Target"}
    for candidate in candidates:
        if candidate in valid_names or candidate in ignored:
            continue
        if any(
            token in candidate.lower()
            for token in ("author", "paragraph", "answer", "target", "text", "classify", "label", "example", "explanation")
        ):
            continue
        return candidate
    return None


def canonical_author_from_single_line(line: str) -> str | None:
    """Return a canonical author only when a single line is exactly a known label or alias."""
    stripped = str(line or "").strip().strip(" \t\r\n\"'`.,;")
    if not stripped:
        return None
    lowered = stripped.lower()
    if any(lowered.startswith(prefix) for prefix in PROMPT_CONTROL_PREFIXES):
        return None
    normalized = normalize_text(stripped)
    if not normalized:
        return None
    if normalized in ALIAS_TO_AUTHOR:
        return ALIAS_TO_AUTHOR[normalized]
    for author in AUTHOR_NAME_TO_ID:
        if normalized == normalize_text(author):
            return author
    return None


def parsed(author: str, status: str, notes: str) -> dict[str, Any]:
    """Build a successful parser response."""
    label = AUTHOR_NAME_TO_ID[author]
    return {
        "parsed_label": int(label),
        "parsed_author": author,
        "parse_status": status,
        "parser_notes": notes,
    }


def invalid(status: str, notes: str) -> dict[str, Any]:
    """Build an unsuccessful parser response."""
    return {
        "parsed_label": None,
        "parsed_author": None,
        "parse_status": status,
        "parser_notes": notes,
    }


def invalid_instruction_echo(notes: str) -> dict[str, Any]:
    """Build an invalid parser response for instruction/meta echo outputs."""
    return {
        "parsed_label": "",
        "parsed_author": None,
        "parse_status": "invalid_instruction_echo",
        "parser_notes": notes,
    }


def is_instruction_echo(text: str) -> bool:
    """Return whether output appears to repeat prompt instructions."""
    normalized = normalize_text(text)
    return any(phrase in normalized for phrase in INSTRUCTION_ECHO_PHRASES)


def normalize_text(text: str) -> str:
    """Normalize text for alias matching."""
    lowered = str(text).lower()
    lowered = lowered.replace("&", " and ")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def unique_in_order(values: list[str]) -> list[str]:
    """Return unique values in first-seen order."""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
