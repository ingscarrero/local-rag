"""End-to-end ingestion: PDF -> text chunks + figure captions (Chroma) + page
images (ColPali). Idempotent per document via content-hashed doc_id."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ..config import settings
from .. import models
from ..retrieval.text_store import TextStore
from ..retrieval.colpali_store import ColPaliStore
from .pdf import parse_pdf
from .chunk import chunk_page_text

console = Console()


def ingest_pdf(
    path: Path,
    text_store: TextStore,
    colpali_store: ColPaliStore | None = None,
    *,
    caption_images: bool = True,
) -> dict:
    """Ingest one PDF. Returns a small stats dict."""
    path = Path(path)
    console.print(f"[bold]Parsing[/bold] {path.name} …")
    doc = parse_pdf(path, settings.page_image_dir)

    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict] = []
    chunk_counter = 0
    n_captions = 0

    for page in doc.pages:
        # 1) text chunks
        for ch in chunk_page_text(page.text, page.page_number, start_index=chunk_counter):
            ids.append(f"{doc.doc_id}::text::{ch.chunk_index}")
            texts.append(ch.text)
            metas.append(
                {
                    "kind": "text",
                    "doc_id": doc.doc_id,
                    "source": doc.name,
                    "page_number": page.page_number,
                }
            )
            chunk_counter += 1

        # 2) figure captions (vision LLM) embedded alongside text
        if caption_images:
            for img in page.images:
                try:
                    caption = models.caption_image(img.image_bytes)
                except Exception as e:  # noqa: BLE001 — server may be unreachable
                    console.print(f"  [yellow]caption failed[/yellow] ({img.image_id}): {e}")
                    continue
                if not caption.strip():
                    continue
                ids.append(f"{doc.doc_id}::caption::{img.image_id}")
                texts.append(f"[Figure on page {page.page_number}] {caption}")
                metas.append(
                    {
                        "kind": "caption",
                        "doc_id": doc.doc_id,
                        "source": doc.name,
                        "page_number": page.page_number,
                        "image_id": img.image_id,
                    }
                )
                n_captions += 1

    console.print(
        f"  embedding {len(texts)} items ({len(texts) - n_captions} text, {n_captions} captions) …"
    )
    text_store.add(ids=ids, texts=texts, metadatas=metas)

    # 3) ColPali visual page index
    n_pages = 0
    if colpali_store is not None:
        records = [
            {
                "doc_id": doc.doc_id,
                "page_number": p.page_number,
                "png_path": str(p.page_png_path),
            }
            for p in doc.pages
        ]
        console.print(f"  ColPali: embedding {len(records)} page images …")
        n_pages = colpali_store.index_pages(records)

    stats = {
        "doc_id": doc.doc_id,
        "source": doc.name,
        "pages": len(doc.pages),
        "text_items": len(texts) - n_captions,
        "captions": n_captions,
        "colpali_pages": n_pages,
    }
    console.print(f"  [green]done[/green]: {stats}")
    return stats


def ingest_folder(
    folder: Path,
    *,
    caption_images: bool = True,
    use_colpali: bool = True,
) -> list[dict]:
    folder = Path(folder)
    pdfs = sorted(folder.glob("**/*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDFs found under {folder}[/yellow]")
        return []
    text_store = TextStore()
    colpali_store = ColPaliStore() if use_colpali else None
    results = []
    for pdf in pdfs:
        results.append(ingest_pdf(pdf, text_store, colpali_store, caption_images=caption_images))
    return results
