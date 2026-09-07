"""Hybrid retrieval: text + visual run together, ranking preserved, top-k honoured."""

from __future__ import annotations

import pytest

from local_rag.config import settings
from local_rag.retrieval import hybrid
from local_rag.retrieval.hybrid import Retrievers

from .conftest import FakeColPaliStore, FakeTextStore, make_text_hit, make_visual_hit


def _ranked_text_hits() -> list:
    # Already in score-descending order, as the store returns them.
    return [
        make_text_hit("chart caption", page=2, kind="caption", score=0.92),
        make_text_hit("naive rag prose", page=1, score=0.81),
        make_text_hit("appendix", page=3, score=0.40),
    ]


def test_returns_text_and_visual_hits_in_store_order() -> None:
    text = FakeTextStore(_ranked_text_hits())
    visual = FakeColPaliStore([make_visual_hit(2, 19.83), make_visual_hit(1, 9.22)])
    text_hits, visual_hits = Retrievers(text=text, visual=visual).retrieve("figure 1")
    assert [h.metadata["page_number"] for h in text_hits] == [2, 1, 3]
    assert [h.score for h in text_hits] == sorted((h.score for h in text_hits), reverse=True)
    assert [v.page_number for v in visual_hits] == [2, 1]
    assert visual_hits[0].score > visual_hits[1].score


def test_both_retrievers_receive_the_same_query_and_default_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "text_top_k", 2)
    monkeypatch.setattr(settings, "visual_top_k", 1)
    text = FakeTextStore(_ranked_text_hits())
    visual = FakeColPaliStore([make_visual_hit(2), make_visual_hit(1)])
    text_hits, visual_hits = Retrievers(text=text, visual=visual).retrieve("q")
    assert text.queries == [("q", 2)]
    assert visual.queries == [("q", 1)]
    assert len(text_hits) == 2 and len(visual_hits) == 1


def test_explicit_top_k_overrides_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "text_top_k", 5)
    monkeypatch.setattr(settings, "visual_top_k", 3)
    text = FakeTextStore(_ranked_text_hits())
    visual = FakeColPaliStore([make_visual_hit(2), make_visual_hit(1)])
    Retrievers(text=text, visual=visual).retrieve("q", text_k=1, visual_k=2)
    assert text.queries == [("q", 1)]
    assert visual.queries == [("q", 2)]


def test_visual_is_skipped_when_disabled() -> None:
    text = FakeTextStore(_ranked_text_hits())
    text_hits, visual_hits = Retrievers(text=text, visual=None).retrieve("q")
    assert len(text_hits) == 3
    assert visual_hits == []


def test_visual_is_skipped_when_index_is_empty() -> None:
    text = FakeTextStore(_ranked_text_hits())
    visual = FakeColPaliStore([])  # count() == 0
    _, visual_hits = Retrievers(text=text, visual=visual).retrieve("q")
    assert visual_hits == []
    assert visual.queries == [], "an empty ColPali index must not be queried (no model load)"


def test_text_and_visual_winners_may_disagree_and_both_are_kept() -> None:
    """Text says *what*, ColPali says *where*: fusion never collapses one into the other."""
    text = FakeTextStore([make_text_hit("prose", page=1, score=0.9)])
    visual = FakeColPaliStore([make_visual_hit(2, 16.1), make_visual_hit(1, 6.7)])
    text_hits, visual_hits = Retrievers(text=text, visual=visual).retrieve("chart")
    assert text_hits[0].metadata["page_number"] == 1
    assert visual_hits[0].page_number == 2


def test_load_builds_text_store_and_optional_visual_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []

    class StubText:
        def __init__(self) -> None:
            built.append("text")

    class StubVisual:
        def __init__(self) -> None:
            built.append("visual")

    monkeypatch.setattr(hybrid, "TextStore", StubText)
    monkeypatch.setattr(hybrid, "ColPaliStore", StubVisual)

    r = Retrievers.load(use_visual=True)
    assert isinstance(r.text, StubText) and isinstance(r.visual, StubVisual)
    assert built == ["text", "visual"]

    built.clear()
    r = Retrievers.load(use_visual=False)
    assert r.visual is None
    assert built == ["text"], "ColPali must not be constructed when visual retrieval is off"
