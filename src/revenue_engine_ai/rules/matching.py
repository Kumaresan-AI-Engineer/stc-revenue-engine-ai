"""Pure ranking rules for semantic matches. No I/O.

A past proposal covers a requirement when its best passage is at least `threshold` similar. The
order is fixed: highest similarity first, then entry id, so the same scores always list the same way.
"""

from dataclasses import dataclass

from revenue_engine_ai.store.ports import Hit

# Starting point for bge-small-en-v1.5 cosine, where unrelated texts already score ~0.55-0.6;
# tune on the evaluation set of past RFPs.
DEFAULT_MATCH_THRESHOLD = 0.75
SEARCH_POOL = 60
# A standalone document is indexed in its own section, under one of these kinds: supporting evidence next to
# the past bids (a case study, or a reference document such as a certificate, a policy or a capability deck).
DOCUMENT_SECTION = "document"
DOCUMENT_KINDS = ("casestudy", "other")
KIND_LABELS = {"proposal": "past bid", "casestudy": "case study", "other": "reference document"}


@dataclass(frozen=True)
class EntryMatch:
    entry_id: str
    similarity: float
    passage_id: str
    page: int
    text: str


def best_per_entry(hits: list[Hit], threshold: float) -> list[EntryMatch]:
    best: dict[str, Hit] = {}
    for hit in hits:
        current = best.get(hit.passage.entry_id)
        if current is None or (hit.similarity, current.passage.id) > (current.similarity, hit.passage.id):
            best[hit.passage.entry_id] = hit
    matches = [
        EntryMatch(h.passage.entry_id, h.similarity, h.passage.id, h.passage.page, h.passage.text)
        for h in best.values()
        if h.similarity >= threshold
    ]
    return sorted(matches, key=lambda m: (-m.similarity, m.entry_id))


# Document search (reports): looser than matching a requirement, since the person reads the passage and the
# score. On a handful of real queries (bge-small) the passage that answers a plain-words description scored
# 0.56-0.65 and text about something else 0.35-0.52: 0.55 keeps the first and drops most of the second.
# A starting point to tune on the evaluation set.
SEARCH_DEFAULT_THRESHOLD = 0.55
# An entry's summary line ("Title (client, year, kind).") is short and generic: it scores 0.5-0.56 against a
# query about something else entirely, while the passages that answer a query score 0.7+. A summary hit has to
# clear the threshold by this much: a title that really matches (0.73-0.76 measured) still counts, and a
# coincidence does not.
SUMMARY_MARGIN = 0.10
SUMMARY_SECTION = "summary"


@dataclass(frozen=True)
class EntryHits:
    """An entry's passages at or above the threshold, best first; `similarity` is the best one's."""

    entry_id: str
    similarity: float
    passages: list[Hit]


def group_hits(hits: list[Hit], threshold: float, per_entry: int, limit: int) -> list[EntryHits]:
    """Group passage hits by entry: each entry keeps its best `per_entry` passages (ties by passage id), and
    entries run best first (ties by entry id), at most `limit`: the same scores always list the same way.
    A hit on an entry's summary line must clear `threshold` by SUMMARY_MARGIN."""
    by_entry: dict[str, list[Hit]] = {}
    for hit in hits:
        needed = threshold + (SUMMARY_MARGIN if hit.passage.section == SUMMARY_SECTION else 0)
        if hit.similarity >= needed:
            by_entry.setdefault(hit.passage.entry_id, []).append(hit)
    grouped = []
    for entry_id, found in by_entry.items():
        best = sorted(found, key=lambda h: (-h.similarity, h.passage.id))[:per_entry]
        grouped.append(EntryHits(entry_id, best[0].similarity, best))
    return sorted(grouped, key=lambda g: (-g.similarity, g.entry_id))[:limit]
