from rules.matching import best_per_entry
from store.ports import Hit, StoredPassage


def _hit(entry: str, pid: str, similarity: float) -> Hit:
    return Hit(StoredPassage(pid, entry, "proposal", 1, f"text {pid}"), similarity)


def test_best_passage_per_entry_above_threshold_in_fixed_order() -> None:
    hits = [_hit("b", "p1", 0.7), _hit("a", "p2", 0.7), _hit("a", "p3", 0.9), _hit("c", "p4", 0.3)]
    matches = best_per_entry(hits, threshold=0.5)
    assert [(m.entry_id, m.passage_id, m.similarity) for m in matches] == [("a", "p3", 0.9), ("b", "p1", 0.7)]


def test_equal_similarity_picks_the_lowest_passage_id() -> None:
    matches = best_per_entry([_hit("a", "p9", 0.8), _hit("a", "p1", 0.8)], threshold=0)
    assert matches[0].passage_id == "p1"


def test_order_does_not_depend_on_input_order() -> None:
    hits = [_hit("b", "p1", 0.6), _hit("a", "p2", 0.6), _hit("c", "p3", 0.6)]
    assert best_per_entry(hits, 0) == best_per_entry(list(reversed(hits)), 0)
