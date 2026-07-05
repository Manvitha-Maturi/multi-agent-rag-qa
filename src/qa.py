# Import the tools we need
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from dotenv import load_dotenv
from groq import Groq

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


# 5. Try it out
if __name__ == "__main__":
    question = "What causes capacity fade in lithium-sulfur batteries?"
    answer, sources = ask(question)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")
    print(f"\nSources used: {', '.join(sources)}")