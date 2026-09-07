"""Model-client wrappers, exercised against a scripted OpenAI-compatible fake."""

from __future__ import annotations

import base64

import pytest

from local_rag import models
from local_rag.config import settings

from .conftest import FakeOpenAIClient


def test_chat_returns_message_content_and_forwards_params(fake_client: FakeOpenAIClient) -> None:
    fake_client.replies = ["hello back"]
    out = models.chat([{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=17)
    assert out == "hello back"
    call = fake_client.calls[0]
    assert call["model"] == settings.llm_model
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 17
    assert call["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_defaults_to_temperature_zero(fake_client: FakeOpenAIClient) -> None:
    models.chat([{"role": "user", "content": "hi"}])
    assert fake_client.calls[0]["temperature"] == 0.0
    assert fake_client.calls[0]["max_tokens"] == 1024


def test_chat_maps_null_content_to_empty_string(fake_client: FakeOpenAIClient) -> None:
    fake_client.replies = [None]
    assert models.chat([{"role": "user", "content": "hi"}]) == ""


def test_embed_empty_batch_short_circuits(fake_client: FakeOpenAIClient) -> None:
    assert models.embed([]) == []
    assert fake_client.embed_calls == []


def test_embed_returns_one_vector_per_input(fake_client: FakeOpenAIClient) -> None:
    vecs = models.embed(["alpha", "beta", "alpha"])
    assert len(vecs) == 3
    assert vecs[0] == vecs[2], "identical inputs embed identically"
    assert vecs[0] != vecs[1]
    assert fake_client.embed_calls[0]["model"] == settings.embed_model
    assert fake_client.embed_calls[0]["input"] == ["alpha", "beta", "alpha"]


def test_caption_image_sends_base64_data_url_and_default_prompt(
    fake_client: FakeOpenAIClient,
) -> None:
    fake_client.replies = ["  A bar chart of throughput vs batch size.  "]
    png = b"\x89PNG\r\n\x1a\nfake"
    out = models.caption_image(png)
    assert out == "A bar chart of throughput vs batch size."
    call = fake_client.calls[0]
    assert call["model"] == settings.vlm_model
    assert call["temperature"] == 0.0
    content = call["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "figure extracted from a technical PDF" in content[0]["text"]
    expected_url = "data:image/png;base64," + base64.b64encode(png).decode()
    assert content[1] == {"type": "image_url", "image_url": {"url": expected_url}}


def test_caption_image_uses_custom_prompt(fake_client: FakeOpenAIClient) -> None:
    models.caption_image(b"img", prompt="Describe the table.")
    assert fake_client.calls[0]["messages"][0]["content"][0]["text"] == "Describe the table."


def test_caption_image_strips_glm_box_markers(fake_client: FakeOpenAIClient) -> None:
    fake_client.replies = ["<|begin_of_box|>Line plot, x=batch size<|end_of_box|>"]
    assert models.caption_image(b"img") == "Line plot, x=batch size"


def test_caption_image_null_content_is_empty(fake_client: FakeOpenAIClient) -> None:
    fake_client.replies = [None]
    assert models.caption_image(b"img") == ""


def test_client_is_cached_per_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: list[dict] = []

    class RecordingOpenAI:
        def __init__(self, **kwargs: object) -> None:
            constructed.append(dict(kwargs))

    monkeypatch.setattr(models, "OpenAI", RecordingOpenAI)
    models._client.cache_clear()
    try:
        a = models._client("http://a/v1")
        b = models._client("http://a/v1")
        c = models._client("http://b/v1")
        assert a is b and a is not c
        assert [k["base_url"] for k in constructed] == ["http://a/v1", "http://b/v1"]
        assert all(k["api_key"] == settings.openai_api_key for k in constructed)
        assert all(k["timeout"] == 600.0 for k in constructed)
    finally:
        models._client.cache_clear()
