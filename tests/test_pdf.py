"""PyMuPDF parsing on PDFs generated in the test itself (no fixtures on disk)."""

from __future__ import annotations

import io
import re
from pathlib import Path

import fitz
import pytest
from PIL import Image

from local_rag.ingest import pdf as pdf_module
from local_rag.ingest.pdf import ParsedDoc, parse_pdf


def _png_bytes(size: int, colour: str = "red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(buf, format="PNG")
    return buf.getvalue()


def _make_pdf(path: Path, *, big_image: bool = True, tiny_image: bool = True) -> Path:
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Naive RAG has problems with ambiguous queries.")
    p2 = doc.new_page()
    p2.insert_text((72, 72), "Figure 1: throughput versus batch size.")
    if big_image:
        p2.insert_image(fitz.Rect(72, 100, 372, 400), stream=_png_bytes(200))
    if tiny_image:
        p2.insert_image(fitz.Rect(400, 100, 420, 120), stream=_png_bytes(20, "blue"))
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def parsed(tmp_path: Path) -> tuple[ParsedDoc, Path]:
    pdf = _make_pdf(tmp_path / "sample.pdf")
    return parse_pdf(pdf, tmp_path / "page_images"), pdf


def test_pages_are_one_based_with_text(parsed: tuple[ParsedDoc, Path]) -> None:
    doc, pdf = parsed
    assert doc.path == pdf and doc.name == "sample.pdf"
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert "Naive RAG has problems" in doc.pages[0].text
    assert "throughput versus batch size" in doc.pages[1].text


def test_doc_id_is_stem_plus_content_hash(parsed: tuple[ParsedDoc, Path], tmp_path: Path) -> None:
    doc, pdf = parsed
    assert re.fullmatch(r"sample-[0-9a-f]{12}", doc.doc_id)
    same = parse_pdf(pdf, tmp_path / "again")
    assert same.doc_id == doc.doc_id, "same bytes → same id (idempotent ingestion)"
    other = parse_pdf(_make_pdf(tmp_path / "sample2.pdf", big_image=False), tmp_path / "x")
    assert other.doc_id != doc.doc_id


def test_every_page_is_rendered_to_png(parsed: tuple[ParsedDoc, Path], tmp_path: Path) -> None:
    doc, _ = parsed
    for p in doc.pages:
        assert (
            p.page_png_path
            == tmp_path / "page_images" / doc.doc_id / f"page-{p.page_number:04d}.png"
        )
        assert p.page_png_path.stat().st_size > 0
        with Image.open(p.page_png_path) as im:
            assert im.format == "PNG"
            # US-letter at 150 DPI is ~1275×1650; anything in that ballpark proves the zoom applied.
            assert im.width > 1000 and im.height > 1300


def test_render_dpi_is_150(parsed: tuple[ParsedDoc, Path]) -> None:
    assert pdf_module._RENDER_DPI == 150
    assert pdf_module._ZOOM == pytest.approx(150 / 72)


def test_only_non_decorative_images_are_extracted(parsed: tuple[ParsedDoc, Path]) -> None:
    doc, _ = parsed
    assert doc.pages[0].images == []
    imgs = doc.pages[1].images
    assert len(imgs) == 1, "the 20×20 image is below the decorative threshold"
    img = imgs[0]
    assert (img.width, img.height) == (200, 200)
    assert img.page_number == 2
    assert img.image_id == f"p2-x{img.xref}"
    assert img.image_bytes[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0"), "raw image bytes are kept"


def test_page_without_images_yields_empty_list(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "plain.pdf", big_image=False, tiny_image=False)
    doc = parse_pdf(pdf, tmp_path / "imgs")
    assert all(p.images == [] for p in doc.pages)


def test_unextractable_image_is_skipped(
    parsed: tuple[ParsedDoc, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pdf = parsed

    def broken_extract(self: fitz.Document, xref: int) -> dict:
        raise RuntimeError("corrupt image stream")

    monkeypatch.setattr(fitz.Document, "extract_image", broken_extract)
    doc = parse_pdf(pdf, tmp_path / "imgs2")
    assert doc.pages[1].images == []
    assert len(doc.pages) == 2, "parsing continues past a bad image"
