"""Agent nodes. Each is a pure-ish function State -> partial State.

The LLM is used for three *decisions* (route, grade, rewrite) and one
*generation* (answer). Keeping the prompts small and the parsing forgiving
makes the agent robust on small local models.
"""

from __future__ import annotations

from .. import models
from ..config import settings
from ..retrieval.hybrid import Retrievers
from ..retrieval.text_store import TextHit
from ..utils import extract_json as _extract_json
from .state import AgentState


# ── Node: route ────────────────────────────────────────────────────────────────
def route_question(state: AgentState) -> AgentState:
    """Adaptive RAG: does this question need the document corpus, or can the
    model answer directly (greetings, meta, general chit-chat)?"""
    q = state["question"]
    reply = models.chat(
        [
            {
                "role": "system",
                "content": (
                    "You route a user's message in a document-Q&A assistant whose job "
                    "is to answer from a private PDF collection. DEFAULT to retrieval. "
                    'Reply JSON {"route": "direct"} ONLY for greetings, thanks, or '
                    "meta-questions about the conversation itself (e.g. 'hello', "
                    "'thanks', 'what did I just ask?'). For ANY question about facts, "
                    "concepts, definitions, or content that could plausibly be in the "
                    'documents, reply {"route": "retrieve"}.'
                ),
            },
            {"role": "user", "content": q},
        ],
        max_tokens=20,
    )
    route = _extract_json(reply).get("route", "retrieve")
    route = route if route in ("retrieve", "direct") else "retrieve"
    return {"route": route, "query": q, "iterations": 0, "trace": [f"route → {route}"]}


# ── Node: rewrite query ──────────────────────────────────────────────────────
def rewrite_query(state: AgentState) -> AgentState:
    """Reformulate the query to improve retrieval after a weak round."""
    reply = models.chat(
        [
            {
                "role": "system",
                "content": (
                    "Rewrite the user's question into a single, keyword-rich search "
                    "query that will retrieve relevant passages. Output only the query."
                ),
            },
            {
                "role": "user",
                "content": f"Original question: {state['question']}\n"
                f"Previous query: {state.get('query', '')}",
            },
        ],
        max_tokens=64,
    )
    new_query = reply.strip().strip('"') or state["question"]
    return {"query": new_query, "trace": [f"rewrite → {new_query!r}"]}


# ── Node: retrieve ──────────────────────────────────────────────────────────
def make_retrieve_node(retrievers: Retrievers):
    def retrieve(state: AgentState) -> AgentState:
        text_hits, visual_hits = retrievers.retrieve(state["query"])
        return {
            "text_hits": text_hits,
            "visual_hits": visual_hits,
            "iterations": state.get("iterations", 0) + 1,
            "trace": [
                f"retrieve → {len(text_hits)} text, {len(visual_hits)} visual "
                f"(iter {state.get('iterations', 0) + 1})"
            ],
        }

    return retrieve


# ── Node: grade documents ─────────────────────────────────────────────────────
def grade_documents(state: AgentState) -> AgentState:
    """Keep only text hits the LLM judges relevant to the question (CRAG)."""
    question = state["question"]
    relevant: list[TextHit] = []
    for hit in state.get("text_hits", []):
        reply = models.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You grade whether a retrieved passage is relevant to the "
                        'question. Reply JSON: {"relevant": true} or {"relevant": false}.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nPassage:\n{hit.text}",
                },
            ],
            max_tokens=20,
        )
        if _extract_json(reply).get("relevant") is True:
            relevant.append(hit)
    return {
        "relevant": relevant,
        "trace": [f"grade → {len(relevant)}/{len(state.get('text_hits', []))} relevant"],
    }


# ── Conditional edge after grading ────────────────────────────────────────────
def decide_after_grade(state: AgentState) -> str:
    """Enough evidence -> generate. Otherwise rewrite & retry, up to the cap."""
    if state.get("relevant"):
        return "generate"
    if state.get("iterations", 0) >= settings.max_agent_iterations:
        return "generate"  # give up retrying; generate will say it can't find it
    return "rewrite"


# ── Node: generate ─────────────────────────────────────────────────────────────
def generate(state: AgentState) -> AgentState:
    question = state["question"]
    relevant = state.get("relevant", [])
    visual_hits = state.get("visual_hits", [])

    if not relevant and not visual_hits:
        return {
            "answer": (
                "I couldn't find supporting evidence in the indexed documents to "
                "answer that. Try rephrasing, or check that the relevant PDF was "
                "ingested."
            ),
            "trace": ["generate → no evidence"],
        }

    context_blocks = []
    for h in relevant:
        tag = "FIGURE" if h.metadata.get("kind") == "caption" else "TEXT"
        context_blocks.append(
            f"[{tag} | {h.metadata.get('source')} p.{h.metadata.get('page_number')}]\n{h.text}"
        )
    visual_refs = "\n".join(
        f"- {v.doc_id} p.{v.page_number} (visual match, score {v.score:.2f})" for v in visual_hits
    )
    context = "\n\n".join(context_blocks) if context_blocks else "(no text passages)"

    answer = models.chat(
        [
            {
                "role": "system",
                "content": (
                    "Answer the question using ONLY the provided context. Cite sources "
                    "inline as (source p.N). If the context is insufficient, say so. "
                    "Be concise and factual."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Context:\n{context}\n\n"
                    f"Visually relevant pages (may contain figures/tables):\n{visual_refs or '(none)'}"
                ),
            },
        ],
        max_tokens=768,
    )
    return {"answer": answer, "trace": ["generate → answer"]}


def answer_directly(state: AgentState) -> AgentState:
    answer = models.chat(
        [
            {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
            {"role": "user", "content": state["question"]},
        ]
    )
    return {"answer": answer, "trace": ["direct answer"]}
