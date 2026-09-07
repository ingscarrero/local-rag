"""Agent nodes, one at a time, against the scripted OpenAI-compatible fake."""

from __future__ import annotations

import pytest

from local_rag.agent import nodes
from local_rag.config import settings
from local_rag.retrieval.hybrid import Retrievers

from .conftest import (
    FakeColPaliStore,
    FakeOpenAIClient,
    FakeTextStore,
    make_text_hit,
    make_visual_hit,
)

# ── route ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"route": "direct"}', "direct"),
        ('Sure! Here you go: {"route": "retrieve"} — done.', "retrieve"),
        ('{"route": "banana"}', "retrieve"),  # unknown label → safe default
        ("I have no idea", "retrieve"),  # no JSON at all → safe default
        ("", "retrieve"),
    ],
)
def test_route_parses_reply_and_defaults_to_retrieve(
    fake_client: FakeOpenAIClient, reply: str, expected: str
) -> None:
    fake_client.replies = [reply]
    out = nodes.route_question({"question": "What does Figure 1 show?"})
    assert out["route"] == expected
    assert out["query"] == "What does Figure 1 show?"
    assert out["iterations"] == 0
    assert out["trace"] == [f"route → {expected}"]


def test_route_prompt_is_tiny_and_biased_to_retrieval(fake_client: FakeOpenAIClient) -> None:
    nodes.route_question({"question": "hello"})
    call = fake_client.calls[0]
    assert call["max_tokens"] == 20
    assert "DEFAULT to retrieval" in call["messages"][0]["content"]
    assert call["messages"][1] == {"role": "user", "content": "hello"}


# ── rewrite ───────────────────────────────────────────────────────────────────


def test_rewrite_strips_quotes_and_whitespace(fake_client: FakeOpenAIClient) -> None:
    fake_client.replies = ['  "throughput batch size bar chart"\n']
    out = nodes.rewrite_query({"question": "What does Figure 1 show?", "query": "figure 1"})
    assert out["query"] == "throughput batch size bar chart"
    assert out["trace"] == ["rewrite → 'throughput batch size bar chart'"]
    user = fake_client.calls[0]["messages"][1]["content"]
    assert "Original question: What does Figure 1 show?" in user
    assert "Previous query: figure 1" in user
    assert fake_client.calls[0]["max_tokens"] == 64


def test_rewrite_falls_back_to_question_when_model_returns_nothing(
    fake_client: FakeOpenAIClient,
) -> None:
    fake_client.replies = ['""']
    out = nodes.rewrite_query({"question": "original?"})
    assert out["query"] == "original?"


# ── retrieve ──────────────────────────────────────────────────────────────────


def test_retrieve_node_records_hits_and_increments_iterations() -> None:
    text = FakeTextStore([make_text_hit("a"), make_text_hit("b")])
    visual = FakeColPaliStore([make_visual_hit(2)])
    retrieve = nodes.make_retrieve_node(Retrievers(text=text, visual=visual))

    out = retrieve({"question": "q", "query": "rewritten q"})
    assert text.queries[0][0] == "rewritten q", "retrieval must use the *current* query"
    assert len(out["text_hits"]) == 2 and len(out["visual_hits"]) == 1
    assert out["iterations"] == 1
    assert out["trace"] == ["retrieve → 2 text, 1 visual (iter 1)"]

    out = retrieve({"question": "q", "query": "q", "iterations": 2})
    assert out["iterations"] == 3
    assert out["trace"] == ["retrieve → 2 text, 1 visual (iter 3)"]


# ── grade ─────────────────────────────────────────────────────────────────────


def test_grade_keeps_only_passages_judged_relevant(fake_client: FakeOpenAIClient) -> None:
    hits = [make_text_hit("chart", page=2), make_text_hit("prose", page=1), make_text_hit("x")]
    fake_client.replies = ['{"relevant": true}', '{"relevant": false}', "unparseable"]
    out = nodes.grade_documents({"question": "Figure 1?", "text_hits": hits})
    assert out["relevant"] == [hits[0]]
    assert out["trace"] == ["grade → 1/3 relevant"]


def test_grade_is_one_call_per_passage_with_question_and_passage(
    fake_client: FakeOpenAIClient,
) -> None:
    hits = [make_text_hit("first passage"), make_text_hit("second passage")]
    nodes.grade_documents({"question": "the question", "text_hits": hits})
    assert len(fake_client.calls) == 2
    for call, hit in zip(fake_client.calls, hits):
        user = call["messages"][1]["content"]
        assert "Question: the question" in user
        assert hit.text in user
        assert call["max_tokens"] == 20


def test_grade_with_no_hits_makes_no_llm_calls(fake_client: FakeOpenAIClient) -> None:
    out = nodes.grade_documents({"question": "q"})
    assert out["relevant"] == []
    assert out["trace"] == ["grade → 0/0 relevant"]
    assert fake_client.calls == []


def test_grade_treats_string_true_as_not_relevant(fake_client: FakeOpenAIClient) -> None:
    fake_client.replies = ['{"relevant": "true"}']
    out = nodes.grade_documents({"question": "q", "text_hits": [make_text_hit()]})
    assert out["relevant"] == [], "only a JSON boolean true counts"


# ── decide_after_grade ────────────────────────────────────────────────────────


def test_decide_generates_when_evidence_exists() -> None:
    assert nodes.decide_after_grade({"relevant": [make_text_hit()], "iterations": 1}) == "generate"


def test_decide_rewrites_while_under_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_agent_iterations", 3)
    assert nodes.decide_after_grade({"relevant": [], "iterations": 1}) == "rewrite"
    assert nodes.decide_after_grade({"relevant": [], "iterations": 2}) == "rewrite"
    assert nodes.decide_after_grade({}) == "rewrite"


def test_decide_gives_up_at_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_agent_iterations", 3)
    assert nodes.decide_after_grade({"relevant": [], "iterations": 3}) == "generate"
    assert nodes.decide_after_grade({"relevant": [], "iterations": 7}) == "generate"


# ── generate ──────────────────────────────────────────────────────────────────


def test_generate_without_any_evidence_declines_without_calling_the_llm(
    fake_client: FakeOpenAIClient,
) -> None:
    out = nodes.generate({"question": "q", "relevant": [], "visual_hits": []})
    assert "couldn't find supporting evidence" in out["answer"]
    assert out["trace"] == ["generate → no evidence"]
    assert fake_client.calls == []


def test_generate_builds_tagged_context_and_visual_refs(fake_client: FakeOpenAIClient) -> None:
    fake_client.replies = ["Throughput rises from 42 to 232 tok/s (sample.pdf p.2)."]
    relevant = [
        make_text_hit(
            "Bar chart: throughput vs batch size", page=2, kind="caption", source="sample.pdf"
        ),
        make_text_hit("Naive RAG stuffs top-k…", page=1, source="sample.pdf"),
    ]
    visual = [
        make_visual_hit(2, 19.834, doc_id="sample-abc"),
        make_visual_hit(1, 9.2, "sample-abc"),
    ]
    out = nodes.generate({"question": "Figure 1?", "relevant": relevant, "visual_hits": visual})

    assert out["answer"].startswith("Throughput rises")
    assert out["trace"] == ["generate → answer"]
    call = fake_client.calls[0]
    assert call["max_tokens"] == 768
    system, user = call["messages"][0]["content"], call["messages"][1]["content"]
    assert "ONLY the provided context" in system and "(source p.N)" in system
    assert "Question: Figure 1?" in user
    assert "[FIGURE | sample.pdf p.2]\nBar chart: throughput vs batch size" in user
    assert "[TEXT | sample.pdf p.1]\nNaive RAG stuffs top-k…" in user
    assert "- sample-abc p.2 (visual match, score 19.83)" in user
    assert "- sample-abc p.1 (visual match, score 9.20)" in user


def test_generate_with_only_visual_evidence_still_answers(fake_client: FakeOpenAIClient) -> None:
    fake_client.replies = ["See page 2."]
    out = nodes.generate({"question": "q", "relevant": [], "visual_hits": [make_visual_hit(2)]})
    assert out["answer"] == "See page 2."
    user = fake_client.last_messages()[1]["content"]
    assert "(no text passages)" in user
    assert "p.2 (visual match" in user


def test_generate_with_only_text_evidence_says_no_visual_pages(
    fake_client: FakeOpenAIClient,
) -> None:
    fake_client.replies = ["answer"]
    nodes.generate({"question": "q", "relevant": [make_text_hit()], "visual_hits": []})
    assert "(none)" in fake_client.last_messages()[1]["content"]


# ── answer_directly ───────────────────────────────────────────────────────────


def test_answer_directly_passes_question_through(fake_client: FakeOpenAIClient) -> None:
    fake_client.replies = ["Hello! How can I help?"]
    out = nodes.answer_directly({"question": "hello"})
    assert out == {"answer": "Hello! How can I help?", "trace": ["direct answer"]}
    assert fake_client.last_messages()[1] == {"role": "user", "content": "hello"}
