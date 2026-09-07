"""Clients for the local model servers (llama.cpp `llama-server`, OpenAI-compatible).

Three independent servers are used so each model is loaded once and stays warm:
  - chat LLM        (agent reasoning, grading, generation)
  - embedding model (text chunks + image captions)
  - vision LLM      (captioning figures during ingestion)

ColPali visual embeddings are handled separately in `retrieval.colpali_store`.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from functools import lru_cache

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .config import settings


@lru_cache(maxsize=3)
def _client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=settings.openai_api_key, timeout=600.0)


def chat(
    messages: Sequence[ChatCompletionMessageParam],
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Single-turn chat completion against the local reasoning LLM."""
    resp = _client(settings.llm_base_url).chat.completions.create(
        model=settings.llm_model,
        messages=list(messages),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the local embedding model."""
    if not texts:
        return []
    resp = _client(settings.embed_base_url).embeddings.create(
        model=settings.embed_model,
        input=texts,
    )
    return [d.embedding for d in resp.data]


def caption_image(image_bytes: bytes, prompt: str | None = None) -> str:
    """Describe a figure/chart/table image with the local vision-language model."""
    prompt = prompt or (
        "You are analyzing a figure extracted from a technical PDF. "
        "Describe what it shows in detail: chart type, axes, trends, labels, "
        "table contents, or diagram structure. Be specific and factual so the "
        "description can be used to retrieve this figure later. Do not speculate."
    )
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    resp = _client(settings.vlm_base_url).chat.completions.create(
        model=settings.vlm_model,
        temperature=0.0,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    )
    text = resp.choices[0].message.content or ""
    # Some VLMs (e.g. GLM-4.x-V) wrap output in special box markers — strip them.
    for marker in ("<|begin_of_box|>", "<|end_of_box|>"):
        text = text.replace(marker, "")
    return text.strip()
