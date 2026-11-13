"""Pure ranking rules for semantic matches. No I/O.

A past proposal covers a requirement when its best passage is at least `threshold` similar. The
order is fixed: highest similarity first, then entry id, so the same scores always list the same way.
"""

from dataclasses import dataclass

from store.ports import Hit

# Starting point for bge-small-en-v1.5 cosine, where unrelated texts already score ~0.55-0.6;
# tune on the evaluation set of past RFPs.
DEFAULT_MATCH_THRESHOLD = 0.75
SEARCH_POOL = 60


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
