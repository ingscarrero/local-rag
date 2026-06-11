# 05 — Interview & Showcase Prep

A field guide for talking about this project with confidence. Organized as the
questions an interviewer *will* ask, with answers you can own — plus the honest
limitations, because naming them is what separates a senior engineer from a demo.

---

## The 60-second pitch

> "It's a fully-local agentic RAG system over PDFs. You point it at a folder and
> ask questions; it answers with citations, and crucially it can reason about
> **figures and tables**, not just prose. Everything runs on-device through
> llama.cpp — no data leaves the machine. The 'agentic' part is a LangGraph state
> machine that routes the question, grades whether retrieved evidence is actually
> relevant, and rewrites-and-retries when it isn't, instead of blindly stuffing
> top-k chunks into a prompt. For images I do two things: caption figures with a
> local vision model *and* embed whole pages with ColPali visual retrieval, then
> fuse both."

## The "why does this matter" framing

Three real problems, three deliberate answers:
1. **Privacy / cost** — sensitive docs can't go to a cloud API; local inference
   solves both confidentiality and per-token cost.
2. **Naive RAG is brittle** — it can't tell good retrieval from bad. Agentic
   grading + corrective retry directly attacks the #1 cause of RAG hallucination.
3. **Documents are visual** — the answer is often in a chart. Most pipelines throw
   that away at OCR; ColPali + captioning keep it.

---

## Architecture questions

**Q: Walk me through what happens when I ask a question.**
Route (does this need the corpus?) → if yes, retrieve from *two* indexes in
parallel (ChromaDB for text+captions, ColPali for page images) → grade each text
passage for relevance with the LLM → if nothing's relevant and we're under the
iteration cap, rewrite the query and retry; otherwise generate a cited answer
constrained to the graded context. The visual hits tell the model which pages hold
the relevant figures.

**Q: Why LangGraph and not a plain loop / LangChain chains / CrewAI?**
The control flow *is* a graph with conditional edges and a cycle (rewrite→retrieve).
LangGraph models that explicitly: typed shared state, nodes, conditional edges. It
makes the corrective loop legible and debuggable (the `trace` field). A plain loop
works but obscures the state machine; chains don't express cycles well; multi-agent
frameworks like CrewAI are overkill — this is one agent with a decision process,
not a team of agents.

**Q: Why three separate model servers?**
`llama-server` is one-model-per-process. Three processes keep chat, embeddings, and
vision each loaded and warm, all behind the same OpenAI-compatible API — so the
client is uniform and provider-swappable (llama.cpp ↔ LM Studio ↔ vLLM ↔ OpenAI by
changing base URLs only). With 512 GB unified memory, co-residency is free.

**Q: Why ChromaDB?**
It's local, persistent, pip-installable, and lets me supply my own (local)
embeddings. For a single-node corpus it's the least-friction choice. I called out
in the design where I'd move to Qdrant/Milvus/pgvector for scale.

---

## The interesting / deep questions

**Q: ColPali vs. just captioning — why both? Isn't that redundant?**
They fail differently. Captioning produces *searchable, readable* text but is
bottlenecked by the VLM's description — it can miss detail or hallucinate. ColPali
embeds the raw page, so it finds the right page even when a caption would've been
wrong, but it returns a *page*, not prose to quote. So I let captions feed the
answer text and ColPali feed the "which page to look at" signal. Using both is more
robust and it shows I understand the tradeoff rather than chasing one buzzword.

**Q: How does ColPali actually work?**
It renders each page to an image and encodes it as *many* patch-level vectors via a
vision-LLM — late interaction, the ColBERT idea applied to pixels. A query is
encoded into token vectors; scoring is **MaxSim**: for each query token take its max
similarity over all page patches, then sum. No OCR, so layout/tables survive. Cost:
hundreds of vectors per page, so index size and scoring are the tradeoff.

**Q: What stops the agent from looping forever?**
A hard iteration cap (`MAX_AGENT_ITERATIONS`, default 3). When it's hit with no
relevant evidence, `generate` runs anyway and honestly says it couldn't find
support — failing loud, not silent.

**Q: How do you keep a 7B model reliable for the grading/routing decisions?**
Short, structured prompts that return tiny JSON (`{"relevant": true}`), a forgiving
parser, temperature 0, and *per-passage* grading instead of batch grading. Small
models are far more reliable on narrow yes/no judgements than on long multi-part
instructions.

**Q: How do you prevent hallucination?**
Three layers: (1) the grader removes irrelevant context before generation; (2) the
generation prompt constrains the answer to provided context and requires citations;
(3) if there's no evidence, it says so. Citations make any drift auditable.

**Q: How would you evaluate it properly?**
Golden questions for a smoke test (I have them for the sample doc), then RAGAS for
faithfulness, answer relevancy, and context precision/recall — runnable against the
same local models. I was explicit that the contribution is the architecture, not a
benchmark number, so I didn't overclaim with a leaderboard.

---

## Scaling & production questions

**Q: This is fine for 50 PDFs. What breaks at 50,000?**
ColPali brute-force MaxSim is O(pages)/query — that's the first bottleneck. Fix:
ANN with multi-vector support (Vespa, Qdrant multi-vector, PLAID) and/or pooled
"Light-ColPali" to shrink vectors per page. ChromaDB is fine to ~10M vectors;
beyond that, a distributed store. Add a cross-encoder re-ranker before grading, and
batch the captioning. None of that changes the agent design.

**Q: Latency?**
Dominated by LLM calls: route (1) + grade (k per round) + maybe rewrite + generate.
The grading loop is the cost. Mitigations: batch/parallelize grading, use a smaller
model just for grading, or gate grading behind a retrieval-confidence threshold.

**Q: How do you handle updates / re-ingestion?**
`doc_id` is a content hash, so re-ingesting a file upserts rather than duplicates.
Changed file → new hash → new entries; I'd add a sweep to delete orphaned old-hash
entries for true sync.

---

## Honest limitations (say these before they ask)

- Brute-force visual search won't scale without an ANN backend.
- Captioning quality is capped by the local VLM; a wrong caption can mislead text
  retrieval (ColPali partly compensates).
- Fixed-window chunking is intentionally simple — I lean on the agent loop instead
  of perfect splitting; semantic chunking would help recall.
- No re-ranker yet; the LLM grader does that job more slowly.
- Single-node, single-user. No auth, no multi-tenant isolation.

## "What would you do next?"

1. Cross-encoder re-ranker between retrieve and grade.
2. Swap ColPali brute force for Qdrant multi-vector; add Light-ColPali pooling.
3. A small Streamlit UI that shows the cited page image inline.
4. RAGAS in CI on the golden set to catch regressions.
5. Query decomposition for multi-hop questions (sub-questions → merge).

## Things to demo live

1. `local-rag ask "What does Figure 1 show…"` → point at the **visual-matches table**
   naming the chart's page: "notice it found the figure page even though the answer
   is visual."
2. Ask something **not** in the corpus → show it declines instead of hallucinating.
3. Say "hello" → show the **route → direct** trace: no wasted retrieval.
4. `tail storage/logs/*.log` → "all local, three warm model servers."
