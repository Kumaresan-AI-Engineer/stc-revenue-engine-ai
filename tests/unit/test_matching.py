from revenue_engine_ai.rules.matching import best_per_entry, group_hits
from revenue_engine_ai.store.ports import Hit, StoredPassage


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


def test_group_hits_keeps_each_entrys_best_passages_and_orders_entries() -> None:
    hits = [
        _hit("b", "p1", 0.7),
        _hit("a", "p2", 0.7),
        _hit("a", "p3", 0.9),
        _hit("a", "p4", 0.8),
        _hit("c", "p5", 0.3),
    ]
    grouped = group_hits(hits, threshold=0.5, per_entry=2, limit=10)
    assert [(g.entry_id, g.similarity, [h.passage.id for h in g.passages]) for g in grouped] == [
        ("a", 0.9, ["p3", "p4"]),
        ("b", 0.7, ["p1"]),
    ]
    assert [g.entry_id for g in group_hits(hits, 0.5, 2, limit=1)] == ["a"]


def test_a_summary_line_must_clear_the_threshold_by_the_margin() -> None:
    def summary(entry: str, similarity: float) -> Hit:
        return Hit(
            StoredPassage(f"s-{entry}", entry, "summary", 0, "Title (client, 2024, other)."), similarity
        )

    hits = [summary("a", 0.62), summary("b", 0.66), _hit("c", "p1", 0.56), _hit("a", "p2", 0.2)]
    grouped = group_hits(hits, threshold=0.55, per_entry=2, limit=10)
    assert [(g.entry_id, g.similarity) for g in grouped] == [
        ("b", 0.66),
        ("c", 0.56),
    ]  # a's title is a coincidence


def test_group_hits_order_does_not_depend_on_input_order() -> None:
    hits = [_hit("b", "p1", 0.6), _hit("a", "p2", 0.6), _hit("a", "p0", 0.6)]
    assert group_hits(hits, 0, 3, 10) == group_hits(list(reversed(hits)), 0, 3, 10)
