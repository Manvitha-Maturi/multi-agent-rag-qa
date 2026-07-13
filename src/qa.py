# Import the tools we need
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from dotenv import load_dotenv
from groq import Groq
from src.router import classify_query
from src.critic import verify_answer
import json 


# 1. Load environment variables from .env (this pulls in your GROQ_API_KEY)
load_dotenv()

# 2. Create a Groq client using the key. os.getenv() reads it from the environment
#    without us ever typing the key itself into this file.
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 3. Load the same embedding model used to build the index
model = SentenceTransformer("all-mpnet-base-v2")

# 4. Load the FAISS index and the chunks/metadata (same as test_search.py)
index = faiss.read_index("data/index/faiss.index")
with open("data/index/chunks.pkl", "rb") as f:
    data = pickle.load(f)
# Sentinel the answerer emits when retrieved context can't support an answer.
# Lets the pipeline flag refusals structurally instead of sniffing prose.
REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"


def retrieve_chunks(query, k=3):
    """Embed the query and return the top-k matching chunks + their sources."""
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        results.append({
            "text": data['chunks'][idx],
            "source": data['metadata'][idx]['source']
        })
    return results

def keyword_search(term, limit=3):
    """
    Simple exact-text search: scan all chunks for ones containing
    the term. Good for names and exact phrases that embedding
    search struggles with.
    """
    results = []
    for idx, text in enumerate(data['chunks']):
        if term.lower() in text.lower():
            results.append({
                "text": text,
                "source": data['metadata'][idx]['source']
            })
            if len(results) >= limit:
                break
    return results

def build_prompt(query, chunks):
    """Combine the retrieved chunks into a single context block for the LLM."""
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    prompt = f"""Answer the question using ONLY the context below. Do not use outside knowledge.

If the context does not contain enough information to answer, respond with this exact first line:
{REFUSAL_SENTINEL}
followed by one sentence naming what is missing. Do not attempt a partial answer in that case.

Context:
{context}

Question: {query}

Answer:"""
    return prompt


def ask(query, k=3):
    """The full pipeline: retrieve -> build prompt -> ask Groq -> return answer."""
    chunks = retrieve_chunks(query, k)
    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = response.choices[0].message.content
    sources = set(c['source'] for c in chunks)
    return answer, sources, chunks

def retrieve_hybrid(query, k=8):
    """
    Retrieval-only version of the comparison logic: LLM identifies the
    two entities being compared, then retrieves per-entity via keyword +
    semantic search. No generation — chunks only, for the orchestrator
    to hand to answer_with_verification().

    KNOWN LIMITATION (found Day 6): entity extraction assumes the query
    already names two comparable things (e.g. "X vs Y"). For vaguer
    comparison queries (e.g. "compare across different authors"), the LLM
    fabricates near-duplicate pseudo-entities, keyword_search returns 0
    for both, and results depend entirely on semantic search happening
    to share vocabulary with the real topic. Currently silent — no error,
    just degraded retrieval. Deferred to Day 7 eval harness alongside the
    critic's lexical-match blind spot.
    """
    extract_prompt = f"""Identify the two specific things being compared in this question (e.g. two authors, two mechanisms, two methods). Reply with ONLY the two items, separated by a comma, nothing else.
    Question: {query}
    Items being compared:"""
    extract_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": extract_prompt}],
        temperature=0
    )
    items = extract_response.choices[0].message.content.strip().split(",")
    items = [i.strip() for i in items]

    all_chunks = []
    for item in items:
        name = item.replace("et al.", "").replace("et al", "").strip()
        all_chunks.extend(keyword_search(name, limit=3))
        all_chunks.extend(retrieve_chunks(f"{name} capacity fade mechanism", k=2))

    seen = set()
    deduped = []
    for c in all_chunks:
        if c['text'] not in seen:
            seen.add(c['text'])
            deduped.append(c)

    return deduped[:k]


def build_revision_prompt(query, context, previous_answer, failed_claims):
    """Build a prompt asking the LLM to fix specific unsupported claims."""
    failed_text = "\n".join(
        f"- \"{c['claim']}\" — NOT supported. Reason: {c['reason']}"
        for c in failed_claims
    )

    prompt = f"""You previously answered a question, but a fact-checker found some claims not supported by the context. Revise your answer to fix ONLY these issues — either remove the unsupported claim, or rephrase it to explicitly note it's not confirmed by the source.

Context:
{context}

Question: {query}

Your previous answer:
{previous_answer}

Unsupported claims to fix:
{failed_text}

Revised answer (using ONLY the context above):"""
    return prompt


def answer_with_verification(query, chunks, max_retries=2):
    """
    Generate an answer, verify it against the chunks, and retry with
    feedback if the critic finds unsupported claims.
    """
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    prompt = build_prompt(query, chunks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    answer = response.choices[0].message.content
    
    # Refusal: model declined because context was insufficient. Nothing to
    # verify, nothing to retry — flag it and return immediately.
    if answer.strip().upper().startswith(REFUSAL_SENTINEL):
        return answer, {"overall_grounded": True, "refused": True, "claims": []}, 0

    for attempt in range(max_retries):
        result = verify_answer(query, answer, chunks)

        if result["overall_grounded"]:
            return answer, result, attempt  # attempt = how many retries it took

        failed_claims = [c for c in result["claims"] if not c["supported"]]
        if not failed_claims:
            # overall_grounded was False but no claims flagged — critic parse issue, stop here
            break

        print(f"[Verifier] Attempt {attempt + 1}: {len(failed_claims)} unsupported claim(s). Retrying...")

        revision_prompt = build_revision_prompt(query, context, answer, failed_claims)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": revision_prompt}],
            temperature=0.2
        )
        answer = response.choices[0].message.content

    # Ran out of retries — return the last answer, flagged as low confidence
    result = verify_answer(query, answer, chunks)
    return answer, result, max_retries
# 5. Try it out

"""
if __name__ == "__main__":
    # Stress test: deliberately give it a question paired with
    # chunks that don't fully support a confident answer
    query = "What is the exact cycle life in cycles of Kumaresan et al.'s lithium-sulfur battery model?"
    chunks = retrieve_chunks("Kumaresan et al. capacity fade model", k=3)

    answer, result, attempts = answer_with_verification(query, chunks)

    print("Final answer:\n", answer)
    print(f"\nGrounded: {result['overall_grounded']} (took {attempts} retry attempt(s))")
"""
def adversarial_test():
    """
    ONE-OFF TEST — retrieves chunks that likely contain a real cycle-life
    number from a DIFFERENT source, then asks about Kumaresan et al.
    specifically. This tempts the model into misattribution, a much more
    realistic hallucination than 'making up a number from nothing.'
    """
    query = "What is the exact cycle life in cycles of Kumaresan et al.'s lithium-sulfur battery model?"

    # Broader retrieval: likely to surface real cycle-life numbers from
    # OTHER papers, not Kumaresan specifically
    chunks = retrieve_chunks("cycle life capacity retention number of cycles", k=4)

    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    bad_prompt = f"""Answer the question below confidently and specifically with an exact number. Do not say the information is missing — the context contains the answer, look carefully.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": bad_prompt}],
        temperature=0.9
    )
    bad_answer = response.choices[0].message.content

    print("=== Deliberately unguarded answer ===")
    print(bad_answer)
    print("\n" + "-"*60)

    result = verify_answer(query, bad_answer, chunks)
    print("\n=== Critic's first pass ===")
    print(json.dumps(result, indent=2))

    if not result["overall_grounded"]:
        failed_claims = [c for c in result["claims"] if not c["supported"]]
        revision_prompt = build_revision_prompt(query, context, bad_answer, failed_claims)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": revision_prompt}],
            temperature=0.2
        )
        fixed_answer = response.choices[0].message.content

        print("\n=== Revised answer after retry ===")
        print(fixed_answer)

        result2 = verify_answer(query, fixed_answer, chunks)
        print("\n=== Critic's second pass ===")
        print(json.dumps(result2, indent=2))
    else:
        print("\n(Still grounded on first pass — model resisted the bait again.)")


if __name__ == "__main__":
    adversarial_test()