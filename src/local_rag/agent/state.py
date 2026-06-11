"""Shared graph state. Every node reads from and writes to this typed dict."""

from __future__ import annotations

from typing import Annotated, TypedDict

from ..retrieval.text_store import TextHit
from ..retrieval.colpali_store import VisualHit


class AgentState(TypedDict, total=False):
    question: str  # original user question
    query: str  # current (possibly rewritten) retrieval query
    route: str  # "retrieve" | "direct"
    text_hits: list[TextHit]
    visual_hits: list[VisualHit]
    relevant: list[TextHit]  # text hits judged relevant by the grader
    iterations: int  # retrieval attempts so far
    answer: str
    trace: Annotated[list[str], lambda a, b: a + b]  # human-readable step log
