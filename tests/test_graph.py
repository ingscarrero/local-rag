"""The compiled LangGraph: edge wiring and every routing path end to end.

The graph, state reducer, and conditional edges are real; only the LLM replies
are scripted. Each test asserts the *trace*, which is the observable contract the
CLI prints.
"""

from __future__ import annotations

import pytest

from local_rag.agent import graph as graph_module
from local_rag.agent.graph import answer_question, build_graph
from local_rag.config import settings
from local_rag.retrieval.hybrid import Retrievers

from .conftest import (
    FakeColPaliStore,
    FakeOpenAIClient,
    FakeTextStore,
    make_text_hit,
    make_visual_hit,
)

RETRIEVE = '{"route": "retrieve"}'
DIRECT = '{"route": "direct"}'
YES = '{"relevant": true}'
NO = '{"relevant": false}'


def _retrievers(n_text: int = 2, visual: bool = True) -> tuple[Retrievers, FakeTextStore]:
    text = FakeTextStore([make_text_hit(f"passage {i}", page=i + 1) for i in range(n_text)])
    vis = FakeColPaliStore([make_visual_hit(2, 19.8), make_visual_hit(1, 9.2)]) if visual else None
    return Retrievers(text=text, visual=vis), text


def test_graph_wiring_matches_the_documented_state_machine() -> None:
    retrievers, _ = _retrievers()
    g = build_graph(retrievers).get_graph()
    assert set(g.nodes) >= {"route", "retrieve", "grade", "rewrite", "generate", "answer_directly"}
    edges = {(e.source, e.target) for e in g.edges}
    assert ("__start__", "route") in edges
    assert ("route", "retrieve") in edges and ("route", "answer_directly") in edges
    assert ("retrieve", "grade") in edges
    assert ("grade", "generate") in edges and ("grade", "rewrite") in edges
    assert ("rewrite", "retrieve") in edges, "the corrective loop must cycle back to retrieve"
    assert ("generate", "__end__") in edges and ("answer_directly", "__end__") in edges
    conditional = {(e.source, e.target) for e in g.edges if e.conditional}
    assert {("route", "retrieve"), ("route", "answer_directly")} <= conditional
    assert {("grade", "generate"), ("grade", "rewrite")} <= conditional


def test_direct_route_skips_retrieval_entirely(fake_client: FakeOpenAIClient) -> None:
    retrievers, text = _retrievers()
    fake_client.replies = [DIRECT, "Hi! Ask me about your documents."]
    final = build_graph(retrievers).invoke({"question": "hello"})
    assert final["answer"] == "Hi! Ask me about your documents."
    assert final["trace"] == ["route → direct", "direct answer"]
    assert text.queries == [], "no vector search on a greeting"
    assert "relevant" not in final


def test_happy_path_retrieve_grade_generate(fake_client: FakeOpenAIClient) -> None:
    retrievers, text = _retrievers(n_text=3)
    fake_client.replies = [RETRIEVE, YES, NO, NO, "It shows X (doc.pdf p.1)."]
    final = build_graph(retrievers).invoke({"question": "What does Figure 1 show?"})
    assert final["trace"] == [
        "route → retrieve",
        "retrieve → 3 text, 2 visual (iter 1)",
        "grade → 1/3 relevant",
        "generate → answer",
    ]
    assert final["answer"] == "It shows X (doc.pdf p.1)."
    assert [h.metadata["page_number"] for h in final["relevant"]] == [1]
    assert final["iterations"] == 1
    assert text.queries == [("What does Figure 1 show?", settings.text_top_k)]


def test_weak_round_triggers_rewrite_and_retry(fake_client: FakeOpenAIClient) -> None:
    retrievers, text = _retrievers(n_text=2)
    fake_client.replies = [
        RETRIEVE,
        NO,
        NO,  # round 1: nothing relevant
        '"throughput batch size chart"',  # rewrite
        YES,
        YES,  # round 2: both relevant
        "final answer",
    ]
    final = build_graph(retrievers).invoke({"question": "figure?"})
    assert final["trace"] == [
        "route → retrieve",
        "retrieve → 2 text, 2 visual (iter 1)",
        "grade → 0/2 relevant",
        "rewrite → 'throughput batch size chart'",
        "retrieve → 2 text, 2 visual (iter 2)",
        "grade → 2/2 relevant",
        "generate → answer",
    ]
    assert [q for q, _ in text.queries] == ["figure?", "throughput batch size chart"]
    assert final["query"] == "throughput batch size chart"
    assert final["question"] == "figure?", "the original question is never overwritten"
    assert final["iterations"] == 2
    assert len(final["relevant"]) == 2


def test_iteration_cap_terminates_the_loop_and_declines_honestly(
    fake_client: FakeOpenAIClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "max_agent_iterations", 2)
    retrievers, text = _retrievers(n_text=1, visual=False)
    fake_client.replies = [RETRIEVE, NO, '"rewrite 1"', NO]
    final = build_graph(retrievers).invoke({"question": "unanswerable?"})
    assert len(text.queries) == 2, "exactly max_agent_iterations retrieval rounds"
    assert final["iterations"] == 2
    assert final["trace"][-1] == "generate → no evidence"
    assert "couldn't find supporting evidence" in final["answer"]
    assert fake_client.replies == [], "every scripted reply was consumed — no extra LLM calls"


def test_iteration_cap_still_generates_when_visual_hits_exist(
    fake_client: FakeOpenAIClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "max_agent_iterations", 1)
    retrievers, _ = _retrievers(n_text=1, visual=True)
    fake_client.replies = [RETRIEVE, NO, "Look at page 2."]
    final = build_graph(retrievers).invoke({"question": "chart?"})
    assert final["trace"] == [
        "route → retrieve",
        "retrieve → 1 text, 2 visual (iter 1)",
        "grade → 0/1 relevant",
        "generate → answer",
    ]
    assert final["answer"] == "Look at page 2."


def test_answer_question_shapes_the_result_for_the_cli(
    fake_client: FakeOpenAIClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrievers, _ = _retrievers(n_text=2)
    seen: list[bool] = []

    def fake_load(use_visual: bool = True) -> Retrievers:
        seen.append(use_visual)
        return retrievers

    monkeypatch.setattr(graph_module.Retrievers, "load", staticmethod(fake_load))
    fake_client.replies = [RETRIEVE, YES, NO, "answer"]
    result = answer_question("q", use_visual=False)
    assert seen == [False]
    assert set(result) == {"answer", "trace", "text_hits", "visual_hits"}
    assert result["answer"] == "answer"
    assert result["trace"][0] == "route → retrieve"
    assert [h.text for h in result["text_hits"]] == ["passage 0"], "text_hits are the *graded* hits"
    assert [v.page_number for v in result["visual_hits"]] == [2, 1]


def test_answer_question_direct_route_returns_empty_hits(
    fake_client: FakeOpenAIClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrievers, _ = _retrievers()
    monkeypatch.setattr(
        graph_module.Retrievers, "load", staticmethod(lambda use_visual=True: retrievers)
    )
    fake_client.replies = [DIRECT, "hey"]
    result = answer_question("hello")
    assert result == {
        "answer": "hey",
        "trace": ["route → direct", "direct answer"],
        "text_hits": [],
        "visual_hits": [],
    }
