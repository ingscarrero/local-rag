"""Command-line interface for local-rag."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import settings

app = typer.Typer(add_completion=False, help="Fully-local agentic RAG over PDFs.")
console = Console()


@app.command()
def ingest(
    path: Optional[Path] = typer.Argument(None, help="PDF file or folder. Defaults to DATA_DIR."),
    no_captions: bool = typer.Option(False, help="Skip vision-LLM figure captioning."),
    no_visual: bool = typer.Option(False, help="Skip ColPali page-image indexing."),
):
    """Parse PDFs and build the text + visual indexes."""
    from .ingest.pipeline import ingest_folder, ingest_pdf
    from .retrieval.text_store import TextStore
    from .retrieval.colpali_store import ColPaliStore

    target = path or settings.data_dir
    if target.is_file():
        text_store = TextStore()
        colpali = None if no_visual else ColPaliStore()
        ingest_pdf(target, text_store, colpali, caption_images=not no_captions)
    else:
        ingest_folder(target, caption_images=not no_captions, use_colpali=not no_visual)
    console.print("[bold green]Ingestion complete.[/bold green]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question."),
    no_visual: bool = typer.Option(False, help="Disable ColPali visual retrieval."),
    show_trace: bool = typer.Option(True, help="Show the agent's step trace."),
):
    """Ask the agent a question over the ingested documents."""
    from .agent.graph import answer_question

    result = answer_question(question, use_visual=not no_visual)

    if show_trace and result["trace"]:
        console.print(Panel("\n".join(result["trace"]), title="agent trace", expand=False))

    console.print(Panel(result["answer"], title="answer", border_style="green"))

    if result["text_hits"]:
        t = Table(title="text sources", show_lines=False)
        t.add_column("source")
        t.add_column("page")
        t.add_column("kind")
        for h in result["text_hits"]:
            t.add_row(
                str(h.metadata.get("source")),
                str(h.metadata.get("page_number")),
                str(h.metadata.get("kind")),
            )
        console.print(t)
    if result["visual_hits"]:
        v = Table(title="visual matches (ColPali)")
        v.add_column("doc")
        v.add_column("page")
        v.add_column("score")
        for h in result["visual_hits"]:
            v.add_row(h.doc_id, str(h.page_number), f"{h.score:.2f}")
        console.print(v)


@app.command()
def status():
    """Show index counts and configured endpoints."""
    from .retrieval.text_store import TextStore

    t = Table(title="local-rag status")
    t.add_column("item")
    t.add_column("value")
    try:
        n_text = TextStore().count()
    except Exception as e:  # noqa: BLE001
        n_text = f"error: {e}"
    t.add_row("chroma items", str(n_text))
    colpali_manifest = settings.colpali_dir / "manifest.json"
    n_pages = 0
    if colpali_manifest.exists():
        import json

        n_pages = len(json.loads(colpali_manifest.read_text()))
    t.add_row("colpali pages", str(n_pages))
    t.add_row("LLM", f"{settings.llm_model} @ {settings.llm_base_url}")
    t.add_row("embed", f"{settings.embed_model} @ {settings.embed_base_url}")
    t.add_row("VLM", f"{settings.vlm_model} @ {settings.vlm_base_url}")
    t.add_row("ColPali", f"{settings.colpali_model} ({settings.colpali_device})")
    console.print(t)


if __name__ == "__main__":
    app()
