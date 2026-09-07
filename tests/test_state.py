"""AgentState: the shared TypedDict and its trace reducer."""

from __future__ import annotations

from typing import get_type_hints

from local_rag.agent.state import AgentState

EXPECTED_KEYS = {
    "question",
    "query",
    "route",
    "text_hits",
    "visual_hits",
    "relevant",
    "iterations",
    "answer",
    "trace",
}


def test_state_exposes_exactly_the_documented_keys() -> None:
    assert set(AgentState.__annotations__) == EXPECTED_KEYS


def test_all_keys_are_optional_so_nodes_can_return_partial_updates() -> None:
    assert AgentState.__total__ is False
    assert AgentState.__required_keys__ == frozenset()
    partial: AgentState = {"question": "q"}
    assert partial["question"] == "q"


def _trace_reducer():
    hints = get_type_hints(AgentState, include_extras=True)
    return hints["trace"].__metadata__[0]


def test_trace_reducer_appends_instead_of_overwriting() -> None:
    reducer = _trace_reducer()
    assert reducer(["route → retrieve"], ["retrieve → 3 text"]) == [
        "route → retrieve",
        "retrieve → 3 text",
    ]


def test_trace_reducer_handles_empty_sides() -> None:
    reducer = _trace_reducer()
    assert reducer([], ["x"]) == ["x"]
    assert reducer(["x"], []) == ["x"]


def test_trace_reducer_does_not_mutate_inputs() -> None:
    reducer = _trace_reducer()
    left, right = ["a"], ["b"]
    reducer(left, right)
    assert left == ["a"] and right == ["b"]


def test_only_trace_carries_a_reducer() -> None:
    hints = get_type_hints(AgentState, include_extras=True)
    annotated = {k for k, v in hints.items() if hasattr(v, "__metadata__")}
    assert annotated == {"trace"}
