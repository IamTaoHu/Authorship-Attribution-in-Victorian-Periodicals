"""Parse Phase 4 decoder outputs into canonical PERIAD authors."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


CANONICAL_AUTHORS = (
    "Leslie Stephen",
    "John Morley",
    "Eliza Lynn Linton",
    "George Henry Lewes",
    "Anne Mozley",
    "James Fitzjames Stephen",
)

OK_STATUSES = {
    "ok_json",
    "ok_embedded_json",
    "ok_full_name",
    "ok_answer_prefix",
    "ok_partial_unique",
}


@dataclass(frozen=True)
class ParseResult:
    predicted_author: str
    parsed_status: str
    parse_error: str
    raw_output: str
    cleaned_output: str


def clean_output(raw_output: Any) -> str:
    if raw_output is None:
        return ""
    text = str(raw_output).strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _author_mentions(text: str) -> list[str]:
    lowered = text.lower()
    return [author for author in CANONICAL_AUTHORS if author.lower() in lowered]


def _canonical_match(value: Any) -> str | None:
    text = str(value or "").strip().strip("\"'.,;:")
    for author in CANONICAL_AUTHORS:
        if text.lower() == author.lower():
            return author
    return None


def _partial_unique(value: str) -> str | None:
    text = value.strip().strip("\"'.,;:").lower()
    if not text:
        return None
    matches = [author for author in CANONICAL_AUTHORS if author.lower().startswith(text)]
    if len(matches) == 1:
        return matches[0]
    return None


def _status_for_unknown_author(value: str) -> str:
    compact = value.strip().strip("\"'.,;:")
    if not compact:
        return "invalid_empty"
    canonical_tokens = {token.lower() for author in CANONICAL_AUTHORS for token in author.split()}
    output_tokens = set(re.findall(r"[A-Za-z]+", compact.lower()))
    if output_tokens and not output_tokens.intersection(canonical_tokens):
        return "invalid_hallucinated_author"
    return "invalid_unknown_author"


def _parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "JSON output is not an object."
    return parsed, None


def _extract_embedded_json(text: str) -> tuple[str | None, str | None]:
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None, None
    candidate = text[first : last + 1]
    _, error = _parse_json_object(candidate)
    return (candidate, None) if error is None else (candidate, error)


def _result(author: str, status: str, raw_output: str, cleaned: str, error: str = "") -> ParseResult:
    return ParseResult(
        predicted_author=author,
        parsed_status=status,
        parse_error=error,
        raw_output=str(raw_output or ""),
        cleaned_output=cleaned,
    )


def parse_phase4_output(raw_output: Any) -> ParseResult:
    """Return a canonical prediction and parser status for one decoder output."""

    raw = str(raw_output or "")
    cleaned = clean_output(raw_output)
    if not cleaned:
        return _result("", "invalid_empty", raw, cleaned, "Output is empty.")

    parsed, json_error = _parse_json_object(cleaned)
    if parsed is not None:
        author = _canonical_match(parsed.get("author"))
        if author:
            return _result(author, "ok_json", raw, cleaned)
        value = str(parsed.get("author", ""))
        partial = _partial_unique(value)
        if partial:
            return _result(partial, "ok_partial_unique", raw, cleaned)
        return _result("", _status_for_unknown_author(value), raw, cleaned, f"Unknown JSON author: {value}")

    embedded, embedded_error = _extract_embedded_json(cleaned)
    if embedded is not None and embedded_error is None:
        parsed_embedded, _ = _parse_json_object(embedded)
        author_value = parsed_embedded.get("author") if parsed_embedded else ""
        author = _canonical_match(author_value)
        if author:
            return _result(author, "ok_embedded_json", raw, cleaned)
        partial = _partial_unique(str(author_value))
        if partial:
            return _result(partial, "ok_partial_unique", raw, cleaned)
        return _result(
            "",
            _status_for_unknown_author(str(author_value)),
            raw,
            cleaned,
            f"Unknown embedded JSON author: {author_value}",
        )

    mentions = _author_mentions(cleaned)
    if len(mentions) > 1:
        return _result("", "invalid_multiple_authors", raw, cleaned, "Output mentions multiple canonical authors.")
    if len(mentions) == 1:
        if re.match(r"^\s*answer\s*:\s*", cleaned, flags=re.IGNORECASE):
            return _result(mentions[0], "ok_answer_prefix", raw, cleaned)
        if cleaned.strip().strip("\"'.,;:").lower() == mentions[0].lower():
            return _result(mentions[0], "ok_full_name", raw, cleaned)

    answer_match = re.match(r"^\s*answer\s*:\s*(.+?)\s*$", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        author = _canonical_match(answer)
        if author:
            return _result(author, "ok_answer_prefix", raw, cleaned)
        partial = _partial_unique(answer)
        if partial:
            return _result(partial, "ok_partial_unique", raw, cleaned)
        return _result("", _status_for_unknown_author(answer), raw, cleaned, f"Unknown answer author: {answer}")

    partial = _partial_unique(cleaned)
    if partial:
        return _result(partial, "ok_partial_unique", raw, cleaned)

    if "{" in cleaned or "}" in cleaned:
        return _result("", "invalid_json", raw, cleaned, embedded_error or json_error or "Invalid JSON output.")
    return _result("", _status_for_unknown_author(cleaned), raw, cleaned, "Could not map output to a canonical author.")


def is_valid_parse(status: str) -> bool:
    return status in OK_STATUSES
