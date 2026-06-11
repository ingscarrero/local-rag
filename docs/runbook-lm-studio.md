# Runbook — LM Studio runtime

Operational reference for running **local-rag** with LM Studio as the model
server. Covers day-to-day start/stop, model swaps, and incident response.

---

## Prerequisites

| Requirement | Check |
|---|---|
| LM Studio ≥ 0.3.x installed | `lms --version` |
| Python deps installed | `uv sync` in project root |
| `.env` configured for LM Studio | `LLM_BASE_URL=http://127.0.0.1:1234/v1` |

---

## Day-to-day operation

### Start

```bash
# 1. Start the LM Studio local server (or use the app's Developer tab)
lms server start

# 2. Confirm the server is up and models are reachable
lms ps
curl -s http://127.0.0.1:1234/v1/models | python3 -m json.tool | head -30

# 3. (Optional) Pre-load models to avoid cold-start delays
lms load text-embedding-nomic-embed-text-v1.5 -y
lms load zai-org/glm-4.6v-flash -y
# Chat model loads JIT on first request — acceptable, or pre-load it too:
# lms load ministral-3-14b-instruct-2512 -y
```

### Stop

```bash
lms server stop
# or close the LM Studio app
```

### Ingest documents

```bash
cp /path/to/your/papers/*.pdf data/
uv run local-rag ingest
# Flags:
#   --no-captions   skip VLM figure captioning (faster, loses chart Q&A)
#   --no-visual     skip ColPali page embeddings (much faster, loses visual retrieval)
```

### Query

```bash
uv run local-rag ask "What does Figure 1 show about throughput vs batch size?"
uv run local-rag status    # shows ingested doc count, chunk count, page count
```

---

## Model management

### List available models

```bash
lms ls            # all downloaded models
lms ps            # currently loaded models + their exact ids
```

### Swap a model

1. Update `LLM_MODEL` / `EMBED_MODEL` / `VLM_MODEL` in `.env` to the new model's
   id (must match `lms ps` exactly).
2. If switching the **embedding model**: delete `storage/chroma/` and re-ingest
   (embeddings must be homogeneous — all from the same model).
3. Restart the query session (no server restart needed).

```bash
# Example: switch chat model to a bigger one
lms get qwen2.5-72b-instruct        # download if not already local
# Edit .env: LLM_MODEL=qwen2.5-72b-instruct
uv run local-rag ask "..."
```

### Recommended model IDs (verified working)

| Role | Model | Notes |
|---|---|---|
| Chat / reasoning | `ministral-3-14b-instruct-2512` | fast, good instruction following |
| Text embeddings | `text-embedding-nomic-embed-text-v1.5` | 768-dim, strong retrieval |
| Vision (figure captions) | `zai-org/glm-4.6v-flash` | fast; strips `<\|begin_of_box\|>` markers automatically |

---

## Smoke test

With `data/sample.pdf` ingested, run the golden questions and verify each:

```bash
uv run local-rag ask "What does Figure 1 show about throughput versus batch size?"
# Expect: route→retrieve, ColPali page 2 ranked first, answer cites tok/s numbers

uv run local-rag ask "What problems does naive RAG have?"
# Expect: route→retrieve, answer cites page 1

uv run local-rag ask "hello there"
# Expect: route→direct, NO retrieve step in trace
```

---

## Incident response

### `connection refused` / `404 model not found`

```bash
# Is the server running?
lms ps
# If not: lms server start

# Does the model id in .env match exactly?
lms ps | grep -i embed    # copy the id exactly, including capitalisation
# Edit .env MODEL= to match
```

### Embeddings dimension mismatch after a model swap

```bash
# ChromaDB rejects mixed-dimension vectors — wipe and re-ingest
rm -rf storage/chroma
uv run local-rag ingest
```

### ColPali out-of-memory on MPS

```bash
# Option 1: lower render DPI (less detail, less memory)
# Edit src/local_rag/ingest/pdf.py  →  _RENDER_DPI = 100

# Option 2: fall back to CPU
# Edit .env  →  COLPALI_DEVICE=cpu
```

### Captions contain `<|begin_of_box|>` markers

Already handled in `models.py` — GLM-style box tokens are stripped automatically.
If your VLM uses different markers, add them to the stripping loop in
`src/local_rag/models.py` → `caption_image()`.

### Full reset

```bash
lms server stop
rm -rf storage/     # clears ChromaDB, ColPali index, page renders
# Re-ingest after
```
