# 00 — Requirements

What local-rag must do (functional) and how well it must do it (non-functional).
Every functional requirement is traced to the module and CLI command that
satisfies it and to the test that proves it. Non-functional numbers are either
**measured** on the reference machine without any model loaded, **derived** from
published model sizes, or explicit **targets** — each row says which.

Reference machine: Apple M3 Ultra, 512 GB unified memory, macOS, Python 3.12.
Measured figures come from `scripts/make_sample_pdf.py` (a 2-page, 1.4 MB PDF
with one chart) and the test suite; they are reproducible with the snippet at
the end of this document.

## Functional requirements

| ID | Requirement | Satisfied by | Verified by |
|----|-------------|--------------|-------------|
| FR-1 | Ingest every PDF under a folder (recursively) or a single PDF file into the indexes. | `cli.ingest` → `ingest/pipeline.ingest_folder` / `ingest_pdf` | `test_cli.py::test_ingest_*`, `test_pipeline.py::test_ingest_folder_walks_recursively_in_sorted_order` |
| FR-2 | Extract per-page text, embedded figures, and a full-page render from each PDF in one parsing pass. | `ingest/pdf.parse_pdf` (PyMuPDF) | `test_pdf.py` |
| FR-3 | Split page text into fixed word windows with overlap so no word is lost at a boundary. | `ingest/chunk.chunk_page_text` (220 words, 40 overlap) | `test_chunking.py::test_all_words_covered`, `test_overlap_*` |
| FR-4 | Skip decorative images (< 100×100 px) and tolerate unextractable image streams without aborting the document. | `ingest/pdf.parse_pdf` | `test_pdf.py::test_only_non_decorative_images_are_extracted`, `test_unextractable_image_is_skipped` |
| FR-5 | Caption each retained figure with a local vision-LLM and index the caption as searchable text tagged with its page; a captioning failure skips that figure only. | `models.caption_image`, `ingest/pipeline.ingest_pdf` | `test_pipeline.py::test_caption_failure_is_logged_and_skipped`, `test_empty_caption_is_not_indexed` |
| FR-6 | Embed text chunks and captions with the *same* local embedding model and store them in one ChromaDB collection distinguished by `kind`. | `retrieval/text_store.TextStore.add`, `models.embed` | `test_text_store.py`, `test_pipeline.py::test_ingest_pdf_indexes_text_chunks_captions_and_pages` |
| FR-7 | Embed every page render with ColQwen2 (late-interaction multi-vectors) and persist the index to disk with a manifest. | `retrieval/colpali_store.ColPaliStore.index_pages` | `test_colpali_store.py::test_index_pages_embeds_persists_and_reports_count` |
| FR-8 | Ingestion is idempotent: re-ingesting an unchanged file overwrites rather than duplicates, in both indexes. | content-hashed `doc_id` (`ingest/pdf._doc_id`), Chroma `upsert`, ColPali manifest de-dup | `test_pdf.py::test_doc_id_is_stem_plus_content_hash`, `test_text_store.py::test_upsert_is_idempotent_by_id`, `test_colpali_store.py::test_index_pages_skips_pages_already_indexed` |
| FR-9 | `--no-captions` skips the vision model; `--no-visual` skips ColPali at ingest and at query time. | `cli.ingest`, `cli.ask`, `retrieval/hybrid.Retrievers.load` | `test_cli.py::test_ingest_*`, `test_ask_no_visual_flag_is_forwarded`, `test_hybrid.py::test_load_*` |
| FR-10 | Route each question: greetings / meta-questions get a direct answer without touching the indexes; everything else retrieves. Unparseable routing replies default to retrieval. | `agent/nodes.route_question` | `test_nodes.py::test_route_*`, `test_graph.py::test_direct_route_skips_retrieval_entirely` |
| FR-11 | Retrieve from both indexes with one query: top-`TEXT_TOP_K` cosine hits from Chroma and top-`VISUAL_TOP_K` MaxSim pages from ColPali; skip ColPali when it is disabled or empty (no model load). | `retrieval/hybrid.Retrievers.retrieve` | `test_hybrid.py` |
| FR-12 | Grade each retrieved passage individually for relevance and drop the rest (Corrective RAG). | `agent/nodes.grade_documents` | `test_nodes.py::test_grade_*` |
| FR-13 | When nothing survives grading, rewrite the query into a keyword-rich form and retry; never exceed `MAX_AGENT_ITERATIONS` retrieval rounds. | `agent/nodes.rewrite_query`, `decide_after_grade`, `agent/graph.build_graph` | `test_graph.py::test_weak_round_triggers_rewrite_and_retry`, `test_iteration_cap_terminates_the_loop_and_declines_honestly` |
| FR-14 | Generate the answer only from graded context, cite sources inline as `(source p.N)`, and list visually matching pages; with no evidence at all, decline explicitly without calling the LLM. | `agent/nodes.generate` | `test_nodes.py::test_generate_*` |
| FR-15 | Expose a human-readable step trace of every agent decision, and print it with the answer, cited text sources, and visual matches. | `agent/state.AgentState.trace` (append reducer), `cli.ask` | `test_state.py`, `test_cli.py::test_ask_prints_trace_answer_and_both_tables` |
| FR-16 | Report index sizes and configured endpoints, even when the text store is unreachable. | `cli.status` | `test_cli.py::test_status_*` |
| FR-17 | All configuration comes from environment variables / `.env` with sane llama.cpp defaults; unknown variables are ignored, malformed ones fail fast. | `config.Settings` | `test_config.py` |
| FR-18 | Talk to any OpenAI-compatible server (LM Studio, llama.cpp, vLLM…) by changing base URLs only; the three roles may share one server or use three. | `models._client` (one cached client per base URL) | `test_models.py::test_client_is_cached_per_base_url` |
| FR-19 | Tolerate small-model quirks: JSON embedded in prose, GLM-style `<\|begin_of_box\|>` markers, null message content. | `utils.extract_json`, `models.caption_image`, `models.chat` | `test_json_extraction.py`, `test_models.py` |

## Non-functional requirements

### Latency

| ID | Metric | Value | Basis |
|----|--------|-------|-------|
| NFR-L1 | CLI cold start (`local-rag --help`) | 0.41 s | **measured** |
| NFR-L2 | Import cost of the query path (`local_rag.agent.graph`: torch + LangGraph + Chroma) | 1.24 s, paid once per `ask` | **measured** |
| NFR-L3 | PDF parsing incl. 150-DPI page render | 36 ms / page (median of 5 runs) | **measured** |
| NFR-L4 | Chunking throughput | 29 M words / s (200 k words → 1 111 chunks in 7 ms) | **measured** |
| NFR-L5 | Unit-test suite | 125 tests in ≈ 3 s | **measured** |
| NFR-L6 | End-to-end `ask` on a 7B-class chat model, Apple Silicon | ≤ 30 s for the common path (1 route + ≤ 5 grades + 1 generate); ≤ 90 s at the iteration cap | **target** — dominated by `TEXT_TOP_K + 2` LLM calls per round |
| NFR-L7 | ColPali MaxSim scoring | O(pages) per query; sub-second up to a few thousand pages on MPS | **target**; see [06 — System design](06-system-design.md#capacity-and-scaling) for the ANN migration point |

### Memory and VRAM footprint

| ID | Component | Value | Basis |
|----|-----------|-------|-------|
| NFR-M1 | Python environment on disk (`uv sync --extra dev`) | 1.0 GB, of which torch 326 MB | **measured** |
| NFR-M2 | Page render on disk (A4/Letter at 150 DPI, 1240×1755 PNG) | 72–124 KB / page | **measured** |
| NFR-M3 | ColPali index on disk | ≈ 384 KB / page (≤ 768 patches × 128-d × float32) → ≈ 38 MB per 100 pages | **derived** from the ColQwen2 processor's patch cap |
| NFR-M4 | Chroma index on disk | ≈ 3 KB / chunk (768-d float32 + text + metadata) | **derived** |
| NFR-M5 | Resident model memory, default llama.cpp models | chat 4.4 GB + vision 5.2 GB + embed 0.1 GB + ColQwen2 ≈ 4.5 GB (bf16) ≈ **14–15 GB** | **derived** from GGUF / safetensors sizes; the captioning VLM is only needed during ingest |
| NFR-M6 | Minimum machine | 16 GB unified memory with `--no-visual` and a ≤ 7B chat model; 32 GB recommended for the full pipeline | **target** |

### Reliability and failure modes

| ID | Requirement | Mechanism | Verified by |
|----|-------------|-----------|-------------|
| NFR-R1 | The agent always terminates. | Hard iteration cap; the only cycle (`rewrite → retrieve`) is guarded by `decide_after_grade`. | `test_graph.py::test_iteration_cap_*` |
| NFR-R2 | A vision-server outage degrades ingestion (no captions) rather than failing it. | per-figure `try/except` in `ingest_pdf` | `test_pipeline.py::test_caption_failure_is_logged_and_skipped` |
| NFR-R3 | A corrupt embedded image never aborts a document. | per-image `try/except` in `parse_pdf` | `test_pdf.py::test_unextractable_image_is_skipped` |
| NFR-R4 | Unparseable LLM decisions fall back to the *safe* choice (retrieve; not relevant). | `extract_json` returns `{}`; explicit defaults | `test_nodes.py::test_route_parses_reply_and_defaults_to_retrieve`, `test_grade_treats_string_true_as_not_relevant` |
| NFR-R5 | `status` works with a broken vector store. | `try/except` around `TextStore()` | `test_cli.py::test_status_survives_an_unreachable_text_store` |
| NFR-R6 | Indexes survive process restarts. | Chroma persistent client; ColPali tensors + manifest on disk | `test_text_store.py::test_data_persists_across_instances`, `test_colpali_store.py::test_index_is_reloaded_from_disk_by_a_new_instance` |
| NFR-R7 | Known unhandled failure modes (documented, not mitigated): swapping the embedding model without deleting `storage/chroma` (dimension mismatch); ColPali OOM at 150 DPI on small GPUs; a chat-server outage fails `ask` loudly with a connection error. | see [04 — Setup › Troubleshooting](04-setup.md#troubleshooting) and [06 — System design › Failure modes](06-system-design.md#failure-modes) | — |

### Cost

| ID | Item | Value | Basis |
|----|------|-------|-------|
| NFR-C1 | Per-query / per-token cost | $0 — no cloud API | by construction |
| NFR-C2 | One-time downloads, default llama.cpp models | chat 4.4 GB, vision 5.2 GB, embed 84 MB, ColQwen2 ≈ 5 GB → **≈ 15 GB** | **measured** (HF cache) / published sizes |
| NFR-C3 | Hardware | any Apple Silicon Mac with ≥ 32 GB for the full pipeline; CUDA or CPU work via `COLPALI_DEVICE` (CPU is slow for ColPali) | **target** |
| NFR-C4 | CI | two ubuntu jobs, no GPU, no model download; dependency install is the dominant cost | by construction |

### Privacy and security

(Lifted from [03 — Architecture › Privacy & security posture](03-architecture.md#privacy--security-posture); see also [SECURITY.md](../SECURITY.md).)

| ID | Requirement |
|----|-------------|
| NFR-S1 | No network egress at inference time. All base URLs default to `127.0.0.1`; the only outbound call is the first-run model download from Hugging Face. |
| NFR-S2 | No secrets. `OPENAI_API_KEY` is a dummy value the local servers ignore; nothing is logged or persisted from it. |
| NFR-S3 | No telemetry from the application; Chroma's anonymised telemetry is disabled in CI and can be disabled locally with `ANONYMIZED_TELEMETRY=False`. |
| NFR-S4 | The only untrusted input is the PDF corpus. Parsing is delegated to PyMuPDF; embedded JavaScript is never executed and links are never followed. |
| NFR-S5 | All writes stay under `STORAGE_DIR` (`chroma/`, `colpali/`, `page_images/`, `logs/`, `pids/`). |

### Maintainability

| ID | Requirement | Value |
|----|-------------|-------|
| NFR-X1 | Statement coverage | 98.6 % measured; CI fails below 95 % |
| NFR-X2 | Static checks | `ruff check`, `ruff format --check`, `mypy` on `src/` — all green in CI |
| NFR-X3 | No test needs a model server, GPU, network, or downloaded weights | enforced by design: every model call is faked at the client seam |

## Reproducing the measured numbers

```bash
uv sync --extra dev
uv run python scripts/make_sample_pdf.py
uv run python -c '
import time, statistics
from pathlib import Path
t0 = time.perf_counter(); import local_rag.agent.graph; print(f"import agent.graph: {time.perf_counter()-t0:.2f}s")
from local_rag.ingest.pdf import parse_pdf
ts = []
for _ in range(5):
    t0 = time.perf_counter(); doc = parse_pdf(Path("data/sample.pdf"), Path("/tmp/lr-imgs")); ts.append(time.perf_counter()-t0)
print(f"parse: {statistics.median(ts)*1000/len(doc.pages):.0f} ms/page")
for p in doc.pages: print(f"  page {p.page_number}: {p.page_png_path.stat().st_size//1024} KB")
'
time uv run local-rag --help > /dev/null
uv run pytest -q
```
