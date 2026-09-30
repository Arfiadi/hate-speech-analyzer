from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and math.isnan(float(value)))


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [value]


def _to_binary(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return int(value)
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
        return int(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"0", "0.0", "false", "no", "non-toxic", "non_toxic"}:
            return 0
        if normalised in {"1", "1.0", "true", "yes", "toxic"}:
            return 1
    return None


def parse_votes(value: Any) -> list[int]:
    """Read scalar, Python-list, JSON-list, or comma-separated annotation votes safely."""
    if _is_missing(value):
        return []

    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = [part.strip() for part in text.split(",")]

    votes: list[int] = []
    for item in _flatten(parsed):
        binary = _to_binary(item)
        if binary is not None:
            votes.append(binary)
    return votes


def vote_summary(votes: list[int]) -> dict[str, int | float | str | None]:
    """Return a consensus label while keeping perfect 50:50 disagreement visible."""
    if not votes:
        return {
            "label": None,
            "status": "missing",
            "annotator_count": 0,
            "positive_votes": 0,
            "negative_votes": 0,
        }

    positive = int(sum(votes))
    total = len(votes)
    negative = total - positive
    if positive == negative:
        label: int | float = 0.5
        status = "tie"
    elif positive > negative:
        label = 1
        status = "positive"
    else:
        label = 0
        status = "negative"
    return {
        "label": label,
        "status": status,
        "annotator_count": total,
        "positive_votes": positive,
        "negative_votes": negative,
    }


def combine_vote_lists(values: list[list[int]]) -> list[int]:
    combined: list[int] = []
    for value in values:
        combined.extend(parse_votes(value))
    return combined
