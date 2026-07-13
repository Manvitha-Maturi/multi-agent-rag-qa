# Multi-Agent RAG QA over Lithium-Sulfur Battery Papers

A retrieval-augmented question-answering system over a corpus of 11 lithium-sulfur
battery research papers, built as a **multi-agent pipeline** (router → retrieval →
answerer → critic) with a **real evaluation harness** that measures router accuracy,
retrieval precision/recall, and answer groundedness.

The differentiator versus a typical tutorial RAG project is the evaluation: instead of
"it answers questions," this repo ships an 18-case hand-labeled test set, a three-layer
metric suite, and results measured across three independent runs — including an explicit,
diagnosed account of where and why the system fails.

---

## Quick start

The FAISS index is committed, so no rebuild or OCR tooling is required to run it.

### See the results with no API key

The evaluation has already been run and its outputs are committed to the repo. To
reproduce all headline numbers **without a Groq key and without any API calls**, just
install dependencies and run the Layer 3 aggregation — it reads the committed
`tests/eval_results_run*.json` files:

```bash
git clone https://github.com/Manvitha-Maturi/multi-agent-rag-qa.git
cd multi-agent-rag-qa

python -m venv venv
source venv/Scripts/activate        # Windows Git Bash
# source venv/bin/activate          # macOS / Linux

pip install -r requirements.txt
python -m tests.layer3              # prints the measured metrics, no key needed
```

A Groq key is only required to run **new live queries** (the CLI or a fresh eval).

### Running live queries

Add a Groq API key in a `.env` file at the repo root. The Groq free tier is sufficient
and takes about a minute to set up with no credit card — create a key at
[console.groq.com](https://console.groq.com):

```
GROQ_API_KEY=your_key_here
```

Then run the interactive CLI:

```bash
python -m src.cli --verbose
```

---

## Demo

The CLI exercises all three routes. Every value shown — route, sources, grounded/refused
verdict, retry count, timing — is produced by the pipeline itself; nothing is synthesized
for display.

**Simple factual question** → grounded answer, verified against sources:

```
Q> What does Mistry define the capacity defect as?

Route: simple_factual   GROUNDED — answer verified against retrieved sources
✓ completed in 3.65s | route: simple_factual | grounded: yes | retries: 0
```

**Out-of-scope question** → router rejects it before any retrieval:

```
Q> Who won the 2022 World Cup?

Route: out_of_scope   OUT OF SCOPE — outside the corpus, not answered
✓ completed in 0.27s | route: out_of_scope | grounded: no | retries: 0
```

**Comparison question needing an un-retrievable paper** → the system refuses rather than
fabricate:

```
Q> Compare how Mistry and Kumaresan represent sulfur transport

Route: comparison   REFUSED — retrieved context did not support an answer
✓ completed in 17.93s | route: comparison | grounded: no | retries: 0
```

That third case is not a bug — it is the designed behavior, and the evaluation below
explains exactly why it happens.

---

## Architecture

The system is a four-stage agent pipeline orchestrated by `src/orchestrator.py`
(`run_pipeline()` returns a `QAResult` dataclass carrying route, answer, sources, and the
grounded/refused/retry state).

1. **Router** (`src/router.py`) — classifies each query as `simple_factual`,
   `comparison`, or `out_of_scope`. Out-of-scope queries exit immediately with no
   retrieval or generation.

2. **Route-aware retrieval** (`src/qa.py`) — simple questions use semantic search over a
   FAISS index (k=4); comparison questions use hybrid retrieval (k=8): an LLM extracts the
   two entities being compared, then each is retrieved via combined keyword + semantic
   search.

3. **Answerer** (`src/qa.py`) — generates an answer from retrieved context only. If the
   context does not support an answer, it emits a refusal sentinel rather than using
   outside knowledge.

4. **Critic / verifier** (`src/critic.py`) — decomposes each answer into individual
   claims and checks every claim against the retrieved sources, returning structured JSON.
   If any claim is unsupported, `answer_with_verification()` triggers a targeted retry
   (re-prompting with the specific failed claims), capped at 2 retries.

**Stack:** FAISS (`faiss-cpu`) + `sentence-transformers` (`all-mpnet-base-v2`) embeddings,
Groq / Llama-3.3-70B for routing, generation, and verification, `rich` for the CLI.

---

## Evaluation

The evaluation is the core of this project. It runs over `tests/test_set.json` — 18
manually verified cases (9 simple factual, 5 comparison including 2 adversarial "vague"
queries, 4 out-of-scope including 2 borderline lithium-ion questions).

```bash
python -m tests.run_eval      # runs the pipeline over the test set, writes per-case JSON
python -m tests.layer3        # aggregates groundedness across runs (no API calls)
```

`run_eval.py` writes results after every case with checkpoint/resume, so a run interrupted
by an API rate limit can be resumed without re-spending tokens on completed cases.

### Three metric layers

- **Layer 1 — Router accuracy:** does each query get the correct route?
- **Layer 2 — Retrieval precision/recall:** were the correct source documents retrieved?
  (Scored at the source-document level.)
- **Layer 3 — Groundedness:** every critic-run case is bucketed as *answered-and-grounded*,
  *refused*, or *answered-and-ungrounded* (a hallucination). Computed by aggregating the
  critic's verdicts already stored in the per-case results — no additional API calls.

### Results (mean across 3 independent runs)

| Metric | Result |
|---|---|
| Router accuracy | **94.4%** |
| Retrieval recall | **58.3%** |
| Retrieval precision | 35.0% |
| Answered & grounded | 6.7 [6–7] of 14 |
| Refused | 7.3 [7–8] of 14 |
| **Answered & ungrounded (hallucinations)** | **0** across all runs |
| Refusal rate | 52.4% [50.0–57.1%] |
| Grounding rate (of committed answers) | 100% |
| Retrieval-failure cases that hallucinated | **0 of 4** |

The eval was run three times to check reproducibility rather than reporting a single-run
point estimate. Across the three runs, exactly one case changed verdict — the system is
close to deterministic in practice.

### Reading the numbers honestly

**Zero hallucinations, at the cost of a ~52% refusal rate.** These two facts are linked.
The system refuses when retrieval does not support an answer instead of fabricating one,
and roughly half the test questions trigger that behavior. That refusal rate is not random
over-caution — it is a symptom of one specific, diagnosed retrieval failure (see below).
Of the 4 cases where retrieval fully failed (recall = 0), **none hallucinated**: 2 refused,
2 answered correctly from a sibling document.

**Precision is structurally low and recall is the real signal.** Precision is scored at the
source-document level against a single gold document while retrieval returns 3–5 documents,
so precision is capped near 0.33 by construction regardless of retrieval quality. Recall is
the meaningful retrieval metric here.

---

## Known limitations

These are documented deliberately. Surfacing and explaining them is the point of the
evaluation harness.

1. **One paper is nearly un-retrievable, which drives most refusals.** The Kumaresan
   mathematical-model paper is the gold source for several comparison questions but is
   almost never retrieved. Its author names do not appear in its own body text, so
   keyword search for "Kumaresan" matches *other papers' reference lists* instead of the
   source paper — the one query guaranteed to miss an author's own paper is a search for
   that author's name. This is the root cause of the high refusal rate. **Planned fix:**
   match keyword search against source-document metadata, not just chunk text.

2. **Critic has a lexical-match blind spot.** The critic marks a claim as supported when
   the right words appear in context, without verifying correct source attribution. Found
   via adversarial testing (`adversarial_test()` in `src/qa.py`); measured systematically
   rather than patched off a single example.

3. **Comparison retrieval degrades on vague queries.** When a comparison query names no
   concrete entities, the entity-extraction step fabricates near-duplicate pseudo-entities
   and keyword search returns nothing. The two `cmp_vague_*` test cases exist to measure
   this.

4. **`build_index.py` does not regenerate the shipped index.** The corpus PDFs are
   image-scanned, so `PyPDF2.extract_text()` returns no text; the committed index was built
   with an OCR step not included in that script. The index is committed so the project runs
   without rebuilding; a from-scratch rebuild would require adding Tesseract OCR.

5. **Comparison routing is ~5x slower** (~18s vs ~3s) because it runs an extra LLM
   entity-extraction call and passes double the context through the verifier.

---

## Repository layout

```
src/
  orchestrator.py    # pipeline: run_pipeline() + QAResult
  router.py          # query classification
  qa.py              # retrieval, generation, hybrid retrieval, retry loop
  critic.py          # claim-level groundedness verification
  cli.py             # interactive demo CLI
  build_index.py     # index builder (see limitation 4)
tests/
  test_set.json      # 18 hand-labeled cases
  run_eval.py        # eval harness (Layer 1 + Layer 2), checkpoint/resume
  layer3.py          # groundedness aggregation (Layer 3)
  eval_results_run*.json
data/
  index/             # committed FAISS index + chunk metadata
  pdfs/              # source corpus
```