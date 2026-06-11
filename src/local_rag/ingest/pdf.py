"""PDF parsing with PyMuPDF.

Produces three things per document:
  - per-page text blocks
  - embedded images (figures/charts) with their bytes, for vision captioning
  - a rendered PNG of each full page, for ColPali visual retrieval
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

# Render pages at ~150 DPI: a good balance of ColPali quality vs. memory.
_RENDER_DPI = 150
_ZOOM = _RENDER_DPI / 72.0
# Skip tiny decorative images (logos, bullets, rules).
_MIN_IMAGE_PIXELS = 100 * 100


@dataclass
class PageImage:
    page_number: int  # 1-based
    xref: int
    image_bytes: bytes
    width: int
    height: int

    @property
    def image_id(self) -> str:
        return f"p{self.page_number}-x{self.xref}"


@dataclass
class ParsedPage:
    page_number: int  # 1-based
    text: str
    page_png_path: Path
    images: list[PageImage] = field(default_factory=list)


@dataclass
class ParsedDoc:
    path: Path
    doc_id: str
    pages: list[ParsedPage]

    @property
    def name(self) -> str:
        return self.path.name


def _doc_id(path: Path) -> str:
    h = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    return f"{path.stem}-{h}"


def parse_pdf(path: Path, page_image_dir: Path) -> ParsedDoc:
    """Parse a PDF into text, embedded images, and per-page renders."""
    path = Path(path)
    doc_id = _doc_id(path)
    out_dir = page_image_dir / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_pages: list[ParsedPage] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            page_number = i + 1
            text = page.get_text("text").strip()

            # Render full page for ColPali.
            pix = page.get_pixmap(matrix=fitz.Matrix(_ZOOM, _ZOOM))
            png_path = out_dir / f"page-{page_number:04d}.png"
            pix.save(png_path)

            # Extract embedded images for captioning.
            images: list[PageImage] = []
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    base = doc.extract_image(xref)
                except Exception:
                    continue
                w, h = base.get("width", 0), base.get("height", 0)
                if w * h < _MIN_IMAGE_PIXELS:
                    continue
                images.append(
                    PageImage(
                        page_number=page_number,
                        xref=xref,
                        image_bytes=base["image"],
                        width=w,
                        height=h,
                    )
                )

            parsed_pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=text,
                    page_png_path=png_path,
                    images=images,
                )
            )

    return ParsedDoc(path=path, doc_id=doc_id, pages=parsed_pages)
