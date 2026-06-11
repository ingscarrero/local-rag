# 04 — Setup & Operation

Step-by-step, reproducible setup. Tested on **macOS (Apple Silicon)** with
**Homebrew**, **uv**, and **miniforge Python 3.12**.

There are two interchangeable ways to serve the local models. They expose the
**same** OpenAI-compatible API, so the application code is identical for both —
only the `*_BASE_URL` / `*_MODEL` values in `.env` differ.

- **Path A — LM Studio** (GUI, recommended if you already use it): one server on
  port `:1234` serves *all three* models, routed by the `model` id in each request.
- **Path B — llama.cpp** (CLI, fully scripted): three `llama-server` processes on
  ports `:8080/:8081/:8082`, one model each.

Pick one. Section 1A/3A cover LM Studio; 1B/3B cover llama.cpp. Everything from
section 2 (Python deps) and section 4 onward is shared.

## 0. Prerequisites

```bash
# uv (https://docs.astral.sh/uv) installed; Homebrew only needed for Path B.
uv --version
```

Apple Silicon is strongly recommended: both runtimes use the Metal backend and
ColPali uses the PyTorch **MPS** device. The system runs on Intel/CUDA too — set
`COLPALI_DEVICE=cuda` or `cpu` in `.env`.

## 1A. LM Studio (recommended)

Install LM Studio (https://lmstudio.ai), then download one model of each kind —
either from the in-app search, or via the `lms` CLI:

```bash
lms get ministral-3-14b-instruct-2512                 # chat / reasoning
lms get text-embedding-nomic-embed-text-v1.5          # text embeddings
lms get zai-org/glm-4.6v-flash                        # vision (figure captions)
```

Start the local server (**Developer** tab → *Start Server*, or the CLI):

```bash
lms server start            # serves the OpenAI-compatible API on :1234
lms ps                      # shows currently loaded models + their exact ids
```

LM Studio loads models just-in-time on first request, but pre-loading avoids a
cold-start stall on the first call:

```bash
lms load text-embedding-nomic-embed-text-v1.5 -y
lms load zai-org/glm-4.6v-flash -y
```

> The `*_MODEL` ids in `.env` **must match** what `lms ps` reports exactly —
> LM Studio routes each request by that id. This is the active configuration.

Skip to **section 2**.

## 1B. llama.cpp (CLI alternative)

```bash
brew install llama.cpp
llama-server --version    # confirm it's on PATH
```

This provides `llama-server` (the OpenAI-compatible server), `llama-mtmd-cli`
(multimodal), and `llama-embedding`. You'll start the servers in **section 3B**.

## 2. Install Python dependencies

```bash
cd local-rag
uv sync                   # creates .venv, installs everything incl. the package
cp .env.example .env      # then set it for your chosen path (below)
```

For **LM Studio (Path A)** `.env` points all three base URLs at the one port and
uses LM Studio's model ids:

```bash
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_MODEL=ministral-3-14b-instruct-2512
EMBED_BASE_URL=http://127.0.0.1:1234/v1
EMBED_MODEL=text-embedding-nomic-embed-text-v1.5
VLM_BASE_URL=http://127.0.0.1:1234/v1
VLM_MODEL=zai-org/glm-4.6v-flash
OPENAI_API_KEY=lm-studio
```

For **llama.cpp (Path B)** the base URLs are the three separate ports
(`:8080/:8081/:8082`) and the `*_MODEL` values can be any non-empty string.

## 3A. Start LM Studio's server

Already done in 1A (`lms server start`). Confirm it's reachable:

```bash
curl -s http://127.0.0.1:1234/v1/models | head -c 400; echo
# => a JSON list including your chat, embed, and vision model ids
```

Skip to **section 4**.

## 3B. Start the llama.cpp servers

```bash
./scripts/serve.sh
```

On first run this **downloads three GGUF models** from Hugging Face (cached for
next time) and starts one `llama-server` per model:

| Port | Model (default)              | Role                          |
|------|------------------------------|-------------------------------|
| 8080 | `bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` | chat / reasoning |
| 8081 | `nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M` | text embeddings |
| 8082 | `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF`       | figure captioning (vision) |

Check health:

```bash
for p in 8080 8081 8082; do curl -s http://127.0.0.1:$p/health; echo " :$p"; done
# => {"status":"ok"} :8080  (x3)
```

Logs live in `storage/logs/`. Stop everything with `./scripts/stop.sh`.

#### Picking bigger models (you have the RAM)

On a 512 GB M3 Ultra you can run far larger models for better answers:

```bash
CHAT_HF=bartowski/Qwen2.5-72B-Instruct-GGUF:Q4_K_M \
VLM_HF=ggml-org/Qwen2.5-VL-32B-Instruct-GGUF \
./scripts/serve.sh
```

(For LM Studio, just download and load a larger model and update `*_MODEL`.)

## 4. Add documents and ingest

```bash
# Use your own PDFs …
cp ~/papers/*.pdf data/
# … or generate the bundled sample (text + a real chart figure):
uv run python scripts/make_sample_pdf.py

uv run local-rag ingest
```

Ingestion: parses each PDF, chunks text, captions figures with the VLM, embeds
text+captions into ChromaDB, and embeds page images with ColPali. ColPali's model
(`vidore/colqwen2-v1.0`, ~5 GB) downloads on first ingest.

Flags: `--no-captions` (skip VLM captioning), `--no-visual` (skip ColPali).

## 5. Ask questions

```bash
uv run local-rag ask "What does Figure 1 show about throughput versus batch size?"
uv run local-rag ask "Why does agentic RAG improve faithfulness over naive RAG?"
uv run local-rag status
```

`ask` prints the agent's step trace, the grounded answer, the cited text sources,
and the ColPali visual page matches.

## Smoke test

With the bundled sample PDF ingested, these golden questions exercise both paths:

| Question | Should retrieve | Tests |
|----------|-----------------|-------|
| "What does Figure 1 show about throughput vs batch size?" | page 2 (chart) | caption + ColPali visual |
| "What problems does naive RAG have?" | page 1 prose | text retrieval + grading |
| "hello" | nothing | routing → direct answer |

Expected: throughput question cites page 2 and the visual-matches table lists
page 2; the naive-RAG question cites page 1; "hello" returns a direct answer with
**no** retrieval in the trace.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `connection refused` / `404 model not found` (LM Studio) | server not started or model id mismatch — run `lms server start` and make `*_MODEL` match `lms ps` exactly |
| `connection refused` on a port (llama.cpp) | server still downloading/loading — `tail -f storage/logs/<name>.log` |
| chat replies but ignores instructions | use an instruct model; for llama.cpp add `--jinja` (serve.sh already does) for correct chat templating |
| embeddings dimension mismatch after switching embed models | delete `storage/chroma` and re-ingest (embeddings must be homogeneous) |
| ColPali OOM | lower `_RENDER_DPI` in `ingest/pdf.py`, or set `COLPALI_DEVICE=cpu` |
| `mps` not available | update macOS/PyTorch, or set `COLPALI_DEVICE=cpu` |
| captions are empty | confirm the VLM is a true vision model; for llama.cpp check it has an mmproj (`storage/logs/vlm.log`) |
| captions contain `<\|begin_of_box\|>` markers | harmless — GLM-style box tokens are stripped in `models.py`; update if your VLM uses other markers |

## Resetting

```bash
# llama.cpp only — stop the servers (LM Studio: stop from the app or `lms server stop`)
./scripts/stop.sh
rm -rf storage/          # clears all indexes, page renders, logs
```
