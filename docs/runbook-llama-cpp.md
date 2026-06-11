# Runbook — llama.cpp runtime

Operational reference for running **local-rag** with `llama.cpp` (`llama-server`)
as the model server. Three separate server processes, one model each.

---

## Prerequisites

| Requirement | Check |
|---|---|
| llama.cpp installed via Homebrew | `llama-server --version` |
| Python deps installed | `uv sync` in project root |
| `.env` configured for llama.cpp | `LLM_BASE_URL=http://127.0.0.1:8080/v1` |

---

## Day-to-day operation

### Start all three servers

```bash
./scripts/serve.sh
```

On first run this downloads three GGUF models from Hugging Face (cached to
`~/.cache/huggingface` for subsequent runs). Typical download sizes:

| Server | Port | Default model | Size |
|---|---|---|---|
| chat | `:8080` | `bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` | ~4.4 GB |
| embeddings | `:8081` | `nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M` | ~84 MB |
| vision | `:8082` | `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` | ~5.2 GB |

Logs stream to `storage/logs/{chat,embed,vlm}.log`.

### Check health

```bash
for p in 8080 8081 8082; do
  printf ":%s " "$p"
  curl -sf http://127.0.0.1:$p/health || echo "UNREACHABLE"
done
echo
# Expected: :8080 {"status":"ok"} :8081 {"status":"ok"} :8082 {"status":"ok"}
```

### Stop all servers

```bash
./scripts/stop.sh
```

### Ingest documents

```bash
cp /path/to/your/papers/*.pdf data/
uv run local-rag ingest
# Flags:
#   --no-captions   skip VLM figure captioning
#   --no-visual     skip ColPali page embeddings
```

### Query

```bash
uv run local-rag ask "What does Figure 1 show about throughput vs batch size?"
uv run local-rag status
```

---

## Model management

### Use larger models (you have the RAM)

Override the default models by passing env vars to `serve.sh`:

```bash
# Example: 72B chat + 32B vision on an M3 Ultra 512 GB
CHAT_HF=bartowski/Qwen2.5-72B-Instruct-GGUF:Q4_K_M \
VLM_HF=ggml-org/Qwen2.5-VL-32B-Instruct-GGUF \
./scripts/serve.sh
```

### Swap the embedding model

1. Stop all servers (`./scripts/stop.sh`).
2. Delete `storage/chroma/` (embeddings must be homogeneous).
3. Edit `serve.sh` `EMBED_HF=` to the new model, update `EMBED_MODEL=` in `.env`.
4. Restart (`./scripts/serve.sh`) and re-ingest.

---

## Smoke test

With `data/sample.pdf` ingested, run the golden questions:

```bash
uv run local-rag ask "What does Figure 1 show about throughput versus batch size?"
# Expect: ColPali page 2 ranked first, answer cites tok/s numbers from p.2

uv run local-rag ask "What problems does naive RAG have?"
# Expect: answer cites page 1

uv run local-rag ask "hello there"
# Expect: route→direct, no retrieval step in trace
```

---

## Incident response

### `connection refused` on a port

```bash
# Check if the server is running
pgrep -la llama-server

# Tail the log for the failing server
tail -50 storage/logs/chat.log     # or embed.log / vlm.log

# Common causes:
# - Model still downloading (wait, re-tail)
# - Port already in use (lsof -i :8080)
# - Not enough RAM to load the model (reduce context with -c 2048 in serve.sh)
```

### Chat replies but ignores system prompt / instructions

The server must be started with `--jinja` for correct chat templating. Check
`serve.sh` — it already passes this flag. Reinstall llama.cpp if the flag is
unrecognised (`brew upgrade llama.cpp`).

### Embeddings dimension mismatch

```bash
rm -rf storage/chroma
uv run local-rag ingest
```

### Vision model returns empty captions

```bash
tail -50 storage/logs/vlm.log
# Verify the model is a true VLM (has an mmproj file alongside the GGUF).
# Qwen2.5-VL and LLaVA families work; text-only models silently return empty.
```

### ColPali out-of-memory

```bash
# Lower render DPI  →  edit src/local_rag/ingest/pdf.py  _RENDER_DPI = 100
# Or fall back to CPU  →  edit .env  COLPALI_DEVICE=cpu
```

### Full reset

```bash
./scripts/stop.sh
rm -rf storage/
# Then re-ingest
```
