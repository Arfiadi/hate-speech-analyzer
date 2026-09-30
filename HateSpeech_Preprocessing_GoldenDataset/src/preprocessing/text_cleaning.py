from __future__ import annotations

import re
from typing import Any


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
WHITESPACE_RE = re.compile(r"\s+")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}", flags=re.DOTALL)
REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "atau",
    "pada", "sebagai", "oleh", "dalam", "kami", "kita", "saya", "kamu", "dia",
    "mereka", "ada", "akan", "sudah", "belum", "saja", "juga", "karena", "agar",
}


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalise_for_dedup(value: Any) -> str:
    """Conservative key used only for grouping likely duplicate messages."""
    text = as_text(value).lower()
    text = URL_RE.sub(" [URL] ", text)
    text = MENTION_RE.sub(" [USER] ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = NON_WORD_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_text(value: Any, variant: str) -> str:
    """Create raw, heavy, or minimal text variants for a controlled experiment."""
    text = as_text(value)
    if variant == "raw":
        return WHITESPACE_RE.sub(" ", text).strip()

    if variant == "minimal":
        text = URL_RE.sub(" [URL] ", text)
        text = MENTION_RE.sub(" [USER] ", text)
        text = HASHTAG_RE.sub(r" \1 ", text)
        text = REPEAT_CHAR_RE.sub(r"\1\1", text)
        text = REPEAT_PUNCT_RE.sub(r"\1\1", text)
        return WHITESPACE_RE.sub(" ", text).strip()

    if variant == "heavy":
        text = text.lower()
        text = URL_RE.sub(" ", text)
        text = MENTION_RE.sub(" ", text)
        text = HASHTAG_RE.sub(r" \1 ", text)
        text = EMOJI_RE.sub(" ", text)
        text = NON_WORD_RE.sub(" ", text)
        tokens = [token for token in WHITESPACE_RE.split(text) if token and token not in STOPWORDS_ID]
        return " ".join(tokens)

    raise ValueError(f"Unknown text-cleaning variant: {variant}")


def text_is_valid(value: Any, min_characters: int, min_tokens: int) -> bool:
    text = WHITESPACE_RE.sub(" ", as_text(value)).strip()
    tokens = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    return len(text) >= min_characters and len(tokens) >= min_tokens
