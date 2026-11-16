"""Pure text rules used to check model output against its sources. No I/O.

A quote counts as found when it appears in the source after normalising case, whitespace,
quotation marks and dashes, so a model re-typing "smart quotes" or a line break is not a miss.
"""

import hashlib
import json
import re
import unicodedata
from typing import Any

# Typographic quotes, dashes and the no-break space, written as escapes so the intent is visible.
_TRANSLATE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)
_SPACE = re.compile(r"\s+")
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|\u2026)\s*")

# Money: a currency sign or code next to a number, or a number with k / m / bn after a sign or code.
_AMOUNT = re.compile(
    r"(?:[€$£₹]\s?\d[\d,.\s]*(?:\s?(?:k|m|bn|million|thousand))?)"
    r"|(?:\b(?:EUR|USD|GBP|INR|AED|SAR)\s?\d[\d,.]*(?:\s?(?:k|m|bn|million|thousand))?)"
    r"|(?:\b\d[\d,.]*\s?(?:EUR|USD|GBP|INR|AED|SAR|euros?|dollars?|rupees?)\b)",
    re.IGNORECASE,
)
_PRICING_QUESTION = re.compile(
    r"\b(price|pricing|priced|cost|costs|fee|fees|rate card|day rate|daily rate|hourly rate|budget|"
    r"quote|quotation|commercial offer|payment terms)\b",
    re.IGNORECASE,
)
AMOUNT_PLACEHOLDER = "[amount]"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    return _SPACE.sub(" ", text).strip().casefold()


def contains_quote(source: str, quote: str) -> bool:
    """True when every part of the quote (split at an ellipsis) appears in the source, in order."""
    parts = [normalize(p) for p in _ELLIPSIS.split(quote) if p.strip()]
    if not parts:
        return False
    haystack = normalize(source)
    start = 0
    for part in parts:
        found = haystack.find(part, start)
        if found < 0:
            return False
        start = found + len(part)
    return True


def mentions_amount(text: str) -> bool:
    return bool(_AMOUNT.search(text))


def redact_amounts(text: str) -> str:
    """Prices from past proposals are restricted and never drafted, so they never reach a prompt."""
    return _AMOUNT.sub(AMOUNT_PLACEHOLDER, text)


def is_pricing_question(text: str) -> bool:
    return bool(_PRICING_QUESTION.search(text))


def stable_hash(*parts: Any) -> str:
    """Order-stable SHA-256 of JSON-serialisable parts (dict keys sorted)."""
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


# Two questions sharing this share of their words (Jaccard) ask the same thing in other words.
NEAR_DUPLICATE = 0.75


def near_duplicate(a: str, b: str) -> bool:
    """The same text once normalised, or nearly the same words."""
    ka, kb = dedupe_key(a), dedupe_key(b)
    if ka == kb:
        return True
    wa, wb = set(ka.split()), set(kb.split())
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= NEAR_DUPLICATE


def dedupe_key(text: str) -> str:
    """Punctuation counts as a word break, so "real-time" and "real time" are the same item."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalize(text)).split())
