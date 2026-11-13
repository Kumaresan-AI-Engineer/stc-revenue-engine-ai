"""Split pages into passages for the library index. Deterministic: the same document always gives
the same passages with the same ids, so re-indexing replaces rather than duplicates."""

import re
from dataclasses import dataclass

from documents.parsing import Page
from rules.text import redact_amounts, stable_hash

MAX_PASSAGE_CHARS = 1200
OVERLAP_CHARS = 150
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Passage:
    id: str
    section: str  # rfp | proposal | summary
    page: int
    ordinal: int
    text: str


def _split(text: str) -> list[str]:
    """Pack whole sentences into passages; carry a short overlap so a fact split at a boundary survives."""
    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
    out: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > MAX_PASSAGE_CHARS:
            out.append(current.strip())
            current = current[-OVERLAP_CHARS:]
        current = f"{current} {sentence}" if current else sentence
        while len(current) > MAX_PASSAGE_CHARS:  # one very long sentence
            out.append(current[:MAX_PASSAGE_CHARS].strip())
            current = current[MAX_PASSAGE_CHARS - OVERLAP_CHARS :]
    if current.strip():
        out.append(current.strip())
    return out


def passages(entry_id: str, section: str, pages: list[Page]) -> list[Passage]:
    out: list[Passage] = []
    for page in pages:
        for text in _split(redact_amounts(page.text)):
            ordinal = len(out)
            out.append(
                Passage(
                    stable_hash(entry_id, section, ordinal, text)[:32], section, page.number, ordinal, text
                )
            )
    return out


def summary_passage(entry_id: str, text: str) -> Passage:
    clean = redact_amounts(text.strip())
    return Passage(stable_hash(entry_id, "summary", clean)[:32], "summary", 0, 0, clean)
