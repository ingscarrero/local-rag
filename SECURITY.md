# Security Policy

## Threat model in one paragraph

local-rag is a single-user, single-machine tool. Every model call goes to an
OpenAI-compatible server bound to `127.0.0.1`; nothing is sent to a cloud API,
and there is no telemetry. The only untrusted input is the PDFs you ingest. The
trust boundary and the privacy posture are described in
[docs/03-architecture.md](docs/03-architecture.md#privacy--security-posture) and
[docs/00-requirements.md](docs/00-requirements.md#privacy--security).

## Supported versions

| Version | Supported |
|---------|-----------|
| `main`  | yes       |
| tagged releases < latest | no — please upgrade |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Use GitHub's private vulnerability reporting on this repository
(*Security → Report a vulnerability*). If that is unavailable, contact the
maintainer through the profile listed at <https://github.com/ingscarrero>.

Include: what you found, how to reproduce it (a minimal PDF if relevant), and
the impact you believe it has. You can expect an acknowledgement within 7 days
and a fix or a documented decision within 30 days for confirmed issues.

## What counts

In scope:

- Anything that causes data to leave the machine (network egress other than
  the configured local base URLs and the one-time Hugging Face model download).
- Malicious PDF content that escapes PyMuPDF parsing into code execution, or
  that writes outside `STORAGE_DIR`.
- Secrets being logged or persisted (there should be none — the API key is a
  dummy value for `llama-server`).

Out of scope:

- Prompt injection *inside* your own documents affecting the answer text. The
  agent is grounded and cites sources, but it does not defend against adversarial
  documents you chose to ingest.
- Vulnerabilities in the model runtimes themselves (LM Studio, llama.cpp) or in
  the models — report those upstream.
- Multi-tenant or authentication concerns: the tool has no users, no auth, and
  no network listener of its own by design.

## Dependency hygiene

- Dependencies are locked in `uv.lock`; `uv sync` reproduces the exact tree.
- `transformers` is pinned to the 4.x line on purpose (see
  [ADR-0005](docs/adr/0005-pin-transformers-4x.md)); bumping it silently breaks
  visual retrieval and must be accompanied by the ranking test described there.

## Known unresolved advisories

Last reviewed 2026-09-11 against the open Dependabot alerts. Every alert that had
a patched version compatible with the `transformers` 4.x pin was resolved by
re-locking (`uv lock --upgrade`) and raising the relevant floors in
`pyproject.toml`. The advisories below have **no fix that this project can
take**; each one is dismissed on GitHub as *tolerable risk* with a pointer to
this section and is re-evaluated whenever the blocking condition changes.

| Package (locked) | Advisories | Why it cannot be fixed | Why the risk is tolerable here |
|---|---|---|---|
| `transformers` 4.53.3 | CVE-2026-4372 (RCE, fixed 5.3.0); CVE-2026-5241 (code execution during model init, fixed 5.5.0); CVE-2026-9856 (`save_pretrained` path traversal, fixed 5.10.0); CVE-2026-1839 (`Trainer` code execution, fixed 5.0.0rc3) | Fixes exist only on the 5.x line. 5.x silently breaks ColQwen2's LoRA adapter — see [ADR-0005](docs/adr/0005-pin-transformers-4x.md). | The only model loaded is the hard-coded `vidore/colqwen2-v1.0` (`COLPALI_MODEL`) from the Hugging Face Hub, with `trust_remote_code` left off. The project never calls `Trainer` or `save_pretrained`, and never loads a model or config from a user-supplied path. The untrusted input (your PDFs) reaches `transformers` only as rasterised page images. |
| `torch` 2.7.1 | CVE-2025-3730 (fixed 2.8.0); CVE-2025-2999 (`unpack_sequence`, fixed 2.9.1); CVE-2025-3001 (`torch.lstm_cell`, fixed 2.10.0); CVE-2025-3000 (`torch.jit.script`, fixed 2.13.0) | `colpali-engine` 0.3.11 — the newest release that works with `transformers` 4.53 — requires `torch<2.8`. Moving to `colpali-engine` 0.3.13 (`torch<2.9`, `transformers<4.58`) would clear only CVE-2025-3730 and must first pass the real-model ranking check in ADR-0005. | None of the affected APIs (`lstm_cell`, `unpack_sequence`, TorchScript) is used. `torch.load` is called only on the embeddings tensor this tool wrote itself under `STORAGE_DIR`; no untrusted checkpoints or scripts are ever loaded. |
| `chromadb` 1.5.9 | CVE-2026-45829 (pre-auth code injection); CVE-2026-45833 (code injection); CVE-2026-45830 (cross-user data access); CVE-2026-45831 (RBAC tenant check) | No patched version exists: 1.5.9 is the latest release on PyPI and is inside the vulnerable range. | All four advisories are in the Chroma **server** and its auth/RBAC providers. This project uses the embedded, in-process `chromadb.PersistentClient` on a local directory: no HTTP listener, no auth layer, no tenants, no remote users. The vulnerable code paths are never started. |
| `ragas` 0.3.1 (`eval` extra only) | CVE-2026-6587 (SSRF in the multi-modal faithfulness metric) | No patched version: every release up to the latest (0.4.3) is in the vulnerable range. | Optional extra, not installed by `uv sync`. Not imported anywhere in `src/`; evaluation scripts run offline against the local model endpoints and do not use the multi-modal faithfulness metric. |
| `diskcache` 5.6.3 (via `ragas`, `eval` extra only) | CVE-2025-69872 (unsafe pickle deserialisation of cache entries) | No patched version. | Same optional extra as `ragas`. The cache directory is local and user-owned; nothing writes untrusted cache entries into it. |

Re-check triggers: a `colpali-engine` release that supports `transformers` 5.x
adapters correctly (re-run the ADR-0005 ranking check before adopting it), a
`chromadb` release above 1.5.9, or a `ragas` release above 0.4.3.
