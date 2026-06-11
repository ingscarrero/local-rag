"""Assemble the agentic-RAG state graph.

route ─┬─(direct)──────────────► answer_directly ──► END
       └─(retrieve)► retrieve ──► grade ─┬─(generate)─► generate ──► END
                        ▲                │
                        └──── rewrite ◄──┘ (relevant==0 and iters<cap)
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from ..retrieval.hybrid import Retrievers
from .state import AgentState
from . import nodes


def build_graph(retrievers: Retrievers):
    g = StateGraph(AgentState)

    g.add_node("route", nodes.route_question)
    g.add_node("retrieve", nodes.make_retrieve_node(retrievers))
    g.add_node("grade", nodes.grade_documents)
    g.add_node("rewrite", nodes.rewrite_query)
    g.add_node("generate", nodes.generate)
    g.add_node("answer_directly", nodes.answer_directly)

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        lambda s: s.get("route", "retrieve"),
        {"retrieve": "retrieve", "direct": "answer_directly"},
    )
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        nodes.decide_after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("answer_directly", END)

    return g.compile()


def answer_question(question: str, use_visual: bool = True) -> dict:
    """Convenience one-shot: build graph, run, return {answer, trace, hits}."""
    retrievers = Retrievers.load(use_visual=use_visual)
    graph = build_graph(retrievers)
    final = graph.invoke({"question": question})
    return {
        "answer": final.get("answer", ""),
        "trace": final.get("trace", []),
        "text_hits": final.get("relevant", []),
        "visual_hits": final.get("visual_hits", []),
    }
