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
