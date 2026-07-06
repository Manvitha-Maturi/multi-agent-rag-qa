# Import the tools we need
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from dotenv import load_dotenv
from groq import Groq
from router import classify_query

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

    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain enough information to answer, say so — do not use outside knowledge.

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

    return answer, sources 

def answer_comparison(query, k=4):
    """
    For comparison questions, we search separately for each entity 
    being compared (rather than one search on the whole question),
    then combine the results so both sides are represented.
    """
    # Ask the LLM to pull out the 2 things being compared
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
    

    # Search separately for each item, then combine results
    all_chunks = []
    for item in items:
      # Strip "et al." so we search for just the name itself
       name = item.replace("et al.", "").replace("et al", "").strip()

       # 1. Keyword search: find chunks literally containing the name
       all_chunks.extend(keyword_search(name, limit=3))

       # 2. Semantic search: find chunks about the topic
       all_chunks.extend(retrieve_chunks(f"{name} capacity fade mechanism", k=2))


    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in all_chunks
    )

    prompt = f"""Answer the question by explicitly comparing the relevant sources below. Structure your answer to clearly show similarities and differences. Use ONLY the context provided — if the context doesn't cover one side of the comparison, say so explicitly.

Context:
{context}

Question: {query}

Answer:"""


    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = response.choices[0].message.content
    sources = set(c['source'] for c in all_chunks)
    return answer, sources

def route_and_answer(query):
    """
    The main entry point: classify the question, then send it down
    the right path.
    """
    category = classify_query(query)
    print(f"[Router] Classified as: {category}")

    if category == "out_of_scope":
        return "This question isn't covered by the loaded battery research papers.", set()

    elif category == "comparison":
        return answer_comparison(query, k=6)

    else:  # simple_factual (and fallback default)
        return ask(query, k=3)

# 5. Try it out
if __name__ == "__main__":
    test_questions = [
        "What causes capacity fade in lithium-sulfur batteries?",
        "Compare the capacity fade mechanisms proposed by Kumaresan et al. vs Hofmann et al.",
        "What's the cycle life of sodium-sulfur batteries?"
    ]

    for question in test_questions:
        answer, sources = route_and_answer(question)
        print(f"\nQuestion: {question}")
        print(f"Answer:\n{answer}")
        if sources:
            print(f"Sources used: {', '.join(sources)}")
        print("-" * 60)

