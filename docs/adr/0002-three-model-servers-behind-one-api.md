# ADR-0002: Three model roles behind one OpenAI-compatible API

- **Status:** Accepted
- **Date:** 2026-06-11
- **Affects:** `models.py`, `config.py`, `scripts/serve.sh`, `.env.example`, both runbooks

## Context

The system needs three different models: a chat/reasoning LLM (routing,
grading, rewriting, generation), a text-embedding model, and a vision-language
model for captioning. They must run locally. `llama-server` (llama.cpp) serves
**one model per process**; LM Studio serves many models on one port and routes
by model id.

Options considered:

1. **One process, swapping models per call.** Every embedding → chat transition
   would reload gigabytes of weights; unusable latency.
2. **Everything in-process via `transformers`.** No servers to manage, but the
   application would own three multi-GB models, lose llama.cpp's Metal-tuned
   GGUF inference, and be impossible to unit-test without weights.
3. **A cloud API.** Violates the first goal — privacy by construction.
4. **Three *roles*, each an OpenAI-compatible base URL + model id.** With
   llama.cpp that is three warm processes (`:8080/:8081/:8082`); with LM Studio
   all three URLs point at `:1234` and the model id does the routing.

## Decision

Option 4. `config.Settings` has `{LLM,EMBED,VLM}_BASE_URL` and `*_MODEL`;
`models.py` keeps one cached `OpenAI` client per base URL and exposes three
functions — `chat`, `embed`, `caption_image`. Nothing else in the codebase
knows which runtime is behind them. `scripts/serve.sh` starts the llama.cpp
trio with Metal offload and health-checks each port.

## Consequences

- **Positive:** every model stays loaded and warm; an `ask` is a handful of
  HTTP calls to localhost with no reload cost.
- **Positive:** runtime is a `.env` change — llama.cpp ↔ LM Studio ↔ vLLM ↔
  (if you ever wanted it) a hosted endpoint. The client code and the tests are
  identical.
- **Positive:** the seam is trivially fakeable; all 125 tests run with a
  scripted client and no server ([tests/conftest.py](../../tests/conftest.py)).
- **Negative:** with llama.cpp there are three processes to babysit (pids, logs,
  ports) — mitigated by `serve.sh`/`stop.sh` and the runbook.
- **Negative:** memory: all three models are co-resident (≈ 10 GB with the
  default 7B-class models). Trivial on the reference machine; on a 16 GB machine
  use `--no-captions` after the first ingest or unload the VLM.
- **Note:** ColQwen2 is the one model that does *not* go through this API —
  late-interaction scoring needs the raw multi-vectors, which no
  OpenAI-compatible endpoint exposes, so it runs in-process
  ([ADR-0001](0001-colpali-plus-captions-over-ocr.md)).
