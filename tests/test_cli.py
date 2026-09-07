"""Typer CLI: ingest / ask / status, with the heavy layers stubbed at their import seams."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from local_rag import cli
from local_rag.config import settings

from .conftest import make_text_hit, make_visual_hit

runner = CliRunner()


@pytest.fixture(autouse=True)
def wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rich would wrap tables at 80 cols; keep assertions on single lines."""
    monkeypatch.setattr(cli, "console", Console(width=200, force_terminal=False))


def test_help_lists_the_three_commands() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("ingest", "ask", "status"):
        assert cmd in result.output


# ── status ────────────────────────────────────────────────────────────────────


def test_status_reports_counts_and_endpoints(
    tmp_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import local_rag.retrieval.text_store as text_store_module

    class StubTextStore:
        def count(self) -> int:
            return 42

    monkeypatch.setattr(text_store_module, "TextStore", StubTextStore)
    manifest = settings.colpali_dir / "manifest.json"
    manifest.write_text(
        json.dumps([{"doc_id": "d", "page_number": p, "png_path": ""} for p in (1, 2, 3)])
    )

    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "chroma items" in out and "42" in out
    assert "colpali pages" in out and "3" in out
    assert f"{settings.llm_model} @ {settings.llm_base_url}" in out
    assert f"{settings.embed_model} @ {settings.embed_base_url}" in out
    assert f"{settings.vlm_model} @ {settings.vlm_base_url}" in out
    assert f"{settings.colpali_model} ({settings.colpali_device})" in out


def test_status_survives_an_unreachable_text_store(
    tmp_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import local_rag.retrieval.text_store as text_store_module

    class BrokenTextStore:
        def __init__(self) -> None:
            raise RuntimeError("chroma exploded")

    monkeypatch.setattr(text_store_module, "TextStore", BrokenTextStore)
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0
    assert "error: chroma exploded" in result.output
    assert "colpali pages" in result.output and "0" in result.output


# ── ask ───────────────────────────────────────────────────────────────────────


@pytest.fixture
def stub_answer(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import local_rag.agent.graph as graph_module

    calls: dict[str, Any] = {}
    payload: dict[str, Any] = {
        "answer": "Throughput rises from 42 to 232 tok/s (sample.pdf p.2).",
        "trace": ["route → retrieve", "retrieve → 3 text, 2 visual (iter 1)", "generate → answer"],
        "text_hits": [make_text_hit("cap", page=2, kind="caption", source="sample.pdf")],
        "visual_hits": [
            make_visual_hit(2, 19.83, "sample-abc"),
            make_visual_hit(1, 9.2, "sample-abc"),
        ],
    }

    def fake_answer_question(question: str, use_visual: bool = True) -> dict[str, Any]:
        calls["question"], calls["use_visual"] = question, use_visual
        return payload

    monkeypatch.setattr(graph_module, "answer_question", fake_answer_question)
    calls["payload"] = payload
    return calls


def test_ask_prints_trace_answer_and_both_tables(stub_answer: dict[str, Any]) -> None:
    result = runner.invoke(cli.app, ["ask", "What does Figure 1 show?"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert stub_answer["question"] == "What does Figure 1 show?"
    assert stub_answer["use_visual"] is True
    assert "agent trace" in out and "route → retrieve" in out
    assert "Throughput rises from 42 to 232 tok/s" in out
    assert "text sources" in out and "sample.pdf" in out and "caption" in out
    assert "visual matches (ColPali)" in out and "19.83" in out and "9.20" in out


def test_ask_no_visual_flag_is_forwarded(stub_answer: dict[str, Any]) -> None:
    result = runner.invoke(cli.app, ["ask", "q", "--no-visual"])
    assert result.exit_code == 0
    assert stub_answer["use_visual"] is False


def test_ask_can_hide_the_trace(stub_answer: dict[str, Any]) -> None:
    result = runner.invoke(cli.app, ["ask", "q", "--no-show-trace"])
    assert result.exit_code == 0
    assert "agent trace" not in result.output
    assert "answer" in result.output


def test_ask_omits_tables_when_there_are_no_hits(stub_answer: dict[str, Any]) -> None:
    stub_answer["payload"].update(
        {
            "answer": "Hi!",
            "trace": ["route → direct", "direct answer"],
            "text_hits": [],
            "visual_hits": [],
        }
    )
    result = runner.invoke(cli.app, ["ask", "hello"])
    assert result.exit_code == 0
    assert "Hi!" in result.output
    assert "text sources" not in result.output
    assert "visual matches" not in result.output


def test_ask_requires_a_question() -> None:
    result = runner.invoke(cli.app, ["ask"])
    assert result.exit_code != 0


# ── ingest ────────────────────────────────────────────────────────────────────


@pytest.fixture
def stub_ingest(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import local_rag.ingest.pipeline as pipeline_module
    import local_rag.retrieval.colpali_store as colpali_module
    import local_rag.retrieval.text_store as text_store_module

    calls: dict[str, Any] = {}

    class StubTextStore:
        pass

    class StubColPaliStore:
        pass

    def fake_ingest_pdf(path: Path, text_store: Any, colpali: Any, *, caption_images: bool) -> dict:
        calls["pdf"] = {
            "path": path,
            "text_store": text_store,
            "colpali": colpali,
            "captions": caption_images,
        }
        return {}

    def fake_ingest_folder(folder: Path, *, caption_images: bool, use_colpali: bool) -> list:
        calls["folder"] = {"folder": folder, "captions": caption_images, "use_colpali": use_colpali}
        return []

    monkeypatch.setattr(text_store_module, "TextStore", StubTextStore)
    monkeypatch.setattr(colpali_module, "ColPaliStore", StubColPaliStore)
    monkeypatch.setattr(pipeline_module, "ingest_pdf", fake_ingest_pdf)
    monkeypatch.setattr(pipeline_module, "ingest_folder", fake_ingest_folder)
    calls["StubTextStore"], calls["StubColPaliStore"] = StubTextStore, StubColPaliStore
    return calls


def test_ingest_defaults_to_the_data_dir_folder(
    stub_ingest: dict[str, Any], tmp_storage: Path
) -> None:
    result = runner.invoke(cli.app, ["ingest"])
    assert result.exit_code == 0, result.output
    assert stub_ingest["folder"] == {
        "folder": settings.data_dir,
        "captions": True,
        "use_colpali": True,
    }
    assert "pdf" not in stub_ingest
    assert "Ingestion complete." in result.output


def test_ingest_folder_flags_are_forwarded(stub_ingest: dict[str, Any], tmp_path: Path) -> None:
    folder = tmp_path / "papers"
    folder.mkdir()
    result = runner.invoke(cli.app, ["ingest", str(folder), "--no-captions", "--no-visual"])
    assert result.exit_code == 0, result.output
    assert stub_ingest["folder"] == {"folder": folder, "captions": False, "use_colpali": False}


def test_ingest_single_file_builds_both_stores(stub_ingest: dict[str, Any], tmp_path: Path) -> None:
    pdf = tmp_path / "one.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    result = runner.invoke(cli.app, ["ingest", str(pdf)])
    assert result.exit_code == 0, result.output
    call = stub_ingest["pdf"]
    assert call["path"] == pdf and call["captions"] is True
    assert isinstance(call["text_store"], stub_ingest["StubTextStore"])
    assert isinstance(call["colpali"], stub_ingest["StubColPaliStore"])
    assert "folder" not in stub_ingest


def test_ingest_single_file_no_visual_skips_colpali(
    stub_ingest: dict[str, Any], tmp_path: Path
) -> None:
    pdf = tmp_path / "one.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    result = runner.invoke(cli.app, ["ingest", str(pdf), "--no-visual", "--no-captions"])
    assert result.exit_code == 0, result.output
    call = stub_ingest["pdf"]
    assert call["colpali"] is None
    assert call["captions"] is False
