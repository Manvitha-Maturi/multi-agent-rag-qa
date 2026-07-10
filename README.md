## Critic / Verifier Agent (Day 5)

To catch hallucinations before they reach the user, the system includes a second
LLM call — a critic agent (`src/critic.py`) — that verifies every generated answer
against the retrieved source chunks before returning it.

**How it works:**
1. The Answerer generates a response from retrieved context (`src/qa.py`)
2. The Critic (`verify_answer()`) decomposes the answer into individual factual
   claims and checks each one against the same context, returning structured JSON
   (per-claim `supported: true/false` + reasoning)
3. If any claim is unsupported, `answer_with_verification()` triggers a retry:
   the Answerer is re-prompted with the specific failed claims and the critic's
   reasoning, and asked to revise — not just "try again," but a targeted fix
4. Capped at 2 retries to avoid looping on genuinely unanswerable questions

### Finding: a lexical-match blind spot in the critic

Adversarial testing (loosened prompts, higher temperature, broader/noisier
retrieval) surfaced a real weakness: when context contained a number ("within
ten cycles") unrelated to the actual question subject, the critic marked the
answer as grounded — it matched the phrase in the text but didn't verify the
number was attributed to the correct entity. This is a known limitation of
LLM-as-judge groundedness checking: **lexical presence in the context isn't the
same as correct attribution.**

This is intentionally left unpatched for now — a fix (requiring the critic to
verify explicit entity/source attribution, not just phrase presence) is planned
for Day 7, where it can be validated against a proper evaluation set rather than
a single anecdotal example.
