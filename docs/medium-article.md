# I Built an Agentic RAG That Can *Read the Charts* — Entirely on My Own Machine

### No cloud, no API keys, no data leaving the laptop. Here's the architecture, a bug that almost shipped silently, and the code.

---

There are hundreds of "agentic RAG" tutorials. Most of them share three traits: they
call a cloud API, they only look at text, and the "agent" is a single
`retrieve → stuff → generate` call wearing a trench coat. I wanted to build the
opposite of that, and I wanted it to be something I'd be comfortable defending in a
technical interview, line by line.

So I built **local-rag**: an agentic RAG system that

- runs **100% locally** — the LLM, the embedding model, and a vision model all
  serve from `llama.cpp` on my own hardware;
- **actually looks at the figures and tables**, not just the prose, using two
  complementary visual-retrieval techniques;
- is **genuinely agentic** — a LangGraph state machine that routes, grades its own
  retrieved evidence, and rewrites-and-retries when the evidence is weak.

This post walks through the design decisions, shows the real terminal output, and —
because this is the part that actually teaches something — tells you about the bug
that made my fancy visual retrieval silently useless, and how a five-line functional
test caught it.

---

## Why "agentic," and why local

**Naive RAG has no idea when it's wrong.** It embeds your chunks, grabs the top-k
nearest neighbors, and stuffs them into the prompt. If those chunks are irrelevant,
it generates anyway — confidently. There's no step that asks *"is this evidence any
good?"*

Agentic RAG inserts that judgment. The agent decides whether to retrieve at all,
reformulates bad queries, **grades whether retrieved passages are relevant**, and
loops back to try again when they aren't. This corrective loop is the single biggest
lever on faithfulness, and it's the thing most tutorials skip.

**Local matters for two unglamorous reasons:** privacy and cost. If your documents
are contracts, patient records, or unreleased research, they can't go to a cloud
endpoint. And running everything on-device means zero per-token cost — you can let
the agent make ten LLM calls per question without watching a meter. On Apple Silicon
with unified memory, models that used to need a datacenter GPU now run on a
workstation.

Here's the whole system on one screen:

```
              ┌──────────────────────── ingestion ────────────────────────┐
   PDFs  ──►  PyMuPDF  ──►  text chunks ─────────────────────────────┐
                       └─►  figures ─► vision-LLM caption ─► embed ───┤
                       │                                              ▼
                       │                                     ┌──────────────┐
                       │                                     │   ChromaDB   │
                       │                                     └──────────────┘
                       └─►  full-page renders ─► ColPali ─► ┌──────────────┐
                                                            │ ColPali index │
                                                            └──────────────┘

   question ─►  route ─┬─ retrieve (text + visual) ─ grade ─┬─ generate ─► answer
                       │         └──────── rewrite & retry ◄─┘    (+ citations)
                       └─ direct answer
```

---

## The stack (and why each piece)

| Concern | Choice | Why |
|---|---|---|
| Model runtime | **llama.cpp** `llama-server` | OpenAI-compatible API, headless, scriptable, Metal-accelerated |
| Reasoning LLM | Qwen2.5-7B-Instruct (GGUF) | strong instruction-following at a size that's reproducible for readers |
| Text embeddings | nomic-embed-text-v1.5 | solid 768-d retrieval embeddings, tiny |
| Figure captioning | Qwen2.5-VL-7B (GGUF + mmproj) | a local model that can *describe a chart* |
| Visual retrieval | **ColQwen2** via `colpali-engine` | late-interaction page-image retrieval; no OCR |
| Vector DB | **ChromaDB** | local, persistent, lets me supply my own embeddings |
| Orchestration | **LangGraph** | the control flow *is* a graph with a cycle — model it as one |

A detail I like: because `llama-server` speaks the OpenAI API, the entire model
client is provider-agnostic. Swap the base URLs and the same code runs against
LM Studio, vLLM, or OpenAI itself. Each model gets its own server process
(ports 8080/8081/8082) so it loads once and stays warm.

```bash
brew install llama.cpp
./scripts/serve.sh     # starts chat :8080, embeddings :8081, vision :8082
```

---

## Making RAG *see*: two visual strategies, on purpose

This is the part I care most about. In real documents the answer is often **in a
figure** — a chart, a table, a diagram. Standard pipelines OCR the page, mangle the
layout, and lose it. I use two complementary techniques and fuse them.

### 1. Caption-and-embed

During ingestion, every extracted figure is handed to the local vision-LLM with a
prompt to describe it factually. That caption is embedded *alongside the prose* in
the same Chroma collection, tagged `kind="caption"`. Now a plain text query can
retrieve "the bar chart where throughput rises then plateaus."

```python
def caption_image(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    resp = client.chat.completions.create(
        model=VLM, temperature=0.0, max_tokens=512,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Describe this figure in detail: chart type, "
                                     "axes, trends, labels, table contents…"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    )
    return resp.choices[0].message.content
```

### 2. ColPali / ColQwen2 — embed the page, skip OCR entirely

[ColPali](https://arxiv.org/abs/2407.01449) renders each page to an image and embeds
it as a **set of patch-level vectors** with a vision-LLM — the "late interaction"
idea from ColBERT, applied to pixels. At query time, the query's token embeddings are
scored against every page's patch embeddings via **MaxSim**. Because it never
linearizes the page into text, it keeps layout, tables, and chart structure intact,
and it beats OCR-then-chunk on visually complex documents.

**Why both?** They fail differently. A caption is *readable* and quotable, but it's
bottlenecked by what the VLM chose to describe. ColPali finds the *right page* even
when a caption would have been wrong — but it returns a page, not prose. So I let
captions feed the answer text and ColPali feed the "which page should I look at"
signal. Using both is more robust than either, and it forces you to understand the
tradeoff instead of chasing one buzzword.

---

## The agent: a LangGraph state machine, not a chain

The control flow has a **cycle** — retrieve, grade, and if the evidence is weak,
rewrite the query and retrieve again. That's not a chain; it's a state machine.
LangGraph models it directly: a typed shared state, nodes, and conditional edges.

```python
g = StateGraph(AgentState)
g.add_node("route", route_question)
g.add_node("retrieve", retrieve)          # text (Chroma) + visual (ColPali)
g.add_node("grade", grade_documents)      # keep only relevant passages
g.add_node("rewrite", rewrite_query)
g.add_node("generate", generate)          # answer, with citations

g.add_conditional_edges("route", pick_route,
                        {"retrieve": "retrieve", "direct": "answer_directly"})
g.add_conditional_edges("grade", decide_after_grade,
                        {"generate": "generate", "rewrite": "rewrite"})
g.add_edge("rewrite", "retrieve")         # the corrective loop
```

The grader is the heart of it. For each retrieved passage the LLM answers a single
yes/no question — *is this relevant to the question?* — and irrelevant passages are
dropped before generation. If nothing survives and we're under the iteration cap, the
agent rewrites the query and tries again. A hard cap guarantees termination.

A lesson worth stating: **small local models are reliable on narrow decisions, not
sprawling ones.** I grade one passage at a time with a tiny prompt returning
`{"relevant": true}`, at temperature 0, parsed with a forgiving extractor. Ask a 7B
model "which of these five is best?" and it flails; ask "is *this one* relevant?" and
it nails it.

---

## It works — here's the real output

After ingesting a sample PDF (two pages: one of prose about RAG, one with a bar chart
of throughput vs. batch size), I asked it about the figure:

```
Q: What does Figure 1 show about throughput versus batch size?

agent trace:
  route → retrieve
  retrieve → 3 text, 2 visual (iter 1)
  grade → 1/3 relevant
  generate → answer

answer:
  Figure 1 shows an increasing trend in throughput as the batch size increases.
  Throughput rises from 42 tok/s at a batch size of 1 to 232 tok/s at a batch
  size of 32 (source p.2).

text sources:   sample.pdf  p.2  (caption)
visual matches: page 2  score 19.83
                page 1  score  9.22
```

Look at what happened. The vision model captioned the chart well enough that the
reasoning model could read the **exact numbers off the bars** (42 → 232 tok/s). The
grader threw out the two page-1 prose chunks as irrelevant to a figure question. And
ColPali independently ranked the chart's page (19.83) far above the text page (9.22).
Text said *what*, ColPali said *where*.

And the corrective router earns its keep on a question that *sounds* general:

```
Q: What problems does naive RAG have?
  route → retrieve → grade → 1/3 relevant → generate
  answer: …ambiguous queries, evidence spread across passages, and answers
          that live in figures or tables rather than prose (source p.1).
```

Grounded in the actual document, cited to page 1 — not a generic from-memory answer.

---

## The bug that almost shipped silently

Here's the part that's worth more than the happy path.

My first end-to-end run *looked* fine. ColPali loaded, produced embeddings of the
right shape `(1, 92, 128)`, returned a plausible MaxSim score. Green across the board.

But I didn't trust "it ran." I wrote a five-line **functional** test: embed both
sample pages, then query with two prompts and check that the *right* page wins.

```
'a bar chart of throughput versus batch size'  -> p1=12.94 p2=12.88  winner=page1  ✗
'definition and problems of naive RAG'          -> p1=13.50 p2=13.44  winner=page1  ✗
```

The scores were **nearly identical regardless of the query**, and the chart query
picked the *wrong* page. The retriever wasn't retrieving — it was returning noise.

The cause: ColQwen2 ships as a **LoRA adapter** on top of a base vision model. I had
`transformers 5.9` installed, and on that version the adapter weights were silently
*not applied* — the model fell back to the untrained base and produced near-uniform
embeddings. No error, no warning I'd have caught at a glance. Inspecting the weights
confirmed it: `lora_B` was all zeros.

The fix was a version pin to the well-tested `transformers 4.x` line. After that:

```
'a bar chart of throughput versus batch size'  -> p1=6.72  p2=16.12  winner=page2  ✓
'definition and problems of naive RAG'          -> p1=12.06 p2=5.12  winner=page1  ✓
```

Sharp, correct discrimination. **The takeaway isn't "pin your deps."** It's that
*"the code ran without error" is not evidence that an ML component works.* Embeddings
fail silently — they always return a vector. The only way to know a retriever
retrieves is to assert that the right thing wins a ranking. That single habit will
save you from shipping a confidently-broken RAG system.

---

## What I'd build next (and the honest limits)

I'm deliberately upfront about where this stops, because pretending otherwise is how
you get caught in an interview:

- **Brute-force MaxSim** over all pages is O(pages) per query. Fine for a personal
  corpus; past a few thousand pages I'd move to a multi-vector ANN store (Vespa,
  Qdrant multi-vector, PLAID) and pooled "Light-ColPali" embeddings.
- **No re-ranker yet** — the LLM grader does that job, more slowly. A cross-encoder
  between retrieve and grade is the natural next step.
- **Captioning quality is capped by the local VLM**; a wrong caption can mislead text
  retrieval (which is exactly why ColPali runs alongside it).
- **Fixed-window chunking** is intentionally simple — I lean on the agent's
  grade-and-retry loop instead of over-tuning the splitter.

---

## Try it

The whole thing is reproducible in about five commands:

```bash
brew install llama.cpp
uv sync
./scripts/serve.sh
uv run python scripts/make_sample_pdf.py   # generates a demo PDF with a real chart
uv run local-rag ingest
uv run local-rag ask "What does Figure 1 show about throughput vs batch size?"
```

No API keys. No data leaving your machine. An agent that grades its own evidence and
can read the charts.

If you build on it — or you're hiring someone who likes to chase silent ML bugs to
ground — I'd love to hear about it.

---

*Built with llama.cpp, ColQwen2, ChromaDB, and LangGraph. Code and full design docs
(concepts, methodology, architecture, setup) are in the repo.*
