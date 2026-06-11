"""Generate a small, self-contained sample PDF with text + a real chart figure.

Produces data/sample.pdf: two pages of prose plus an embedded bar chart, so both
retrieval paths (text/caption via Chroma, and visual page via ColPali) have
something meaningful to find. No external assets or libraries beyond PIL + PyMuPDF.

    uv run python scripts/make_sample_pdf.py
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

OUT = Path("data/sample.pdf")

PAGE1 = (
    "Local-First Retrieval-Augmented Generation: A Technical Note\n\n"
    "Retrieval-augmented generation (RAG) grounds a language model's answers in an "
    "external corpus instead of relying solely on parametric memory. A naive RAG "
    "pipeline embeds document chunks, retrieves the nearest neighbours of a query, and "
    "stuffs them into the prompt. This works for simple lookups but fails when the "
    "query is ambiguous, when relevant evidence is spread across several passages, or "
    "when the answer lives in a figure or table rather than prose.\n\n"
    "Agentic RAG addresses these failure modes by treating retrieval as a decision "
    "process. An agent decides whether to retrieve at all, reformulates weak queries, "
    "grades whether retrieved passages are actually relevant, and retries when the "
    "evidence is insufficient. Empirically, this corrective loop improves answer "
    "faithfulness on multi-hop and visually grounded questions.\n\n"
    "Running the entire stack locally — the language model, the embedding model, and a "
    "vision model — keeps private documents private and removes per-token API costs. "
    "On Apple Silicon with unified memory, models that once required a data-centre GPU "
    "now run comfortably on a workstation."
)

PAGE2_INTRO = (
    "Throughput Benchmark\n\n"
    "The chart below reports decode throughput in tokens per second as a function of "
    "batch size for a 7-billion-parameter model running on a local Metal backend. "
    "Throughput rises steeply from batch size 1 to 8 as the GPU is better saturated, "
    "then begins to plateau beyond batch size 16 as memory bandwidth becomes the "
    "bottleneck. The practical takeaway: small batches leave the accelerator idle, "
    "while very large batches yield diminishing returns."
)


def make_chart() -> bytes:
    W, H = 900, 520
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    # axes
    left, bottom, top, right = 90, 440, 60, 850
    d.line([(left, top), (left, bottom)], fill="black", width=2)
    d.line([(left, bottom), (right, bottom)], fill="black", width=2)
    d.text((W // 2 - 120, 20), "Figure 1: Throughput vs. Batch Size", fill="black")
    d.text((10, 240), "tok/s", fill="black")
    d.text((430, 470), "batch size", fill="black")

    batches = [1, 2, 4, 8, 16, 32]
    tput = [42, 78, 130, 190, 220, 232]  # tokens/sec
    max_t = 250
    n = len(batches)
    slot = (right - left) / n
    bar_w = slot * 0.55
    for i, (b, t) in enumerate(zip(batches, tput)):
        x0 = left + i * slot + (slot - bar_w) / 2
        x1 = x0 + bar_w
        y1 = bottom
        y0 = bottom - (t / max_t) * (bottom - top)
        d.rectangle([x0, y0, x1, y1], fill=(60, 120, 200))
        d.text((x0 + 4, y0 - 16), str(t), fill="black")
        d.text((x0 + bar_w / 2 - 6, bottom + 6), str(b), fill="black")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_textbox(fitz.Rect(60, 60, 540, 760), PAGE1, fontsize=11, fontname="helv")

    p2 = doc.new_page()
    p2.insert_textbox(fitz.Rect(60, 60, 540, 240), PAGE2_INTRO, fontsize=11, fontname="helv")
    chart = make_chart()
    p2.insert_image(fitz.Rect(70, 260, 540, 540), stream=chart)

    doc.save(OUT)
    doc.close()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
