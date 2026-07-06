import os
from dotenv import load_dotenv
from groq import Groq

# Load the same API key we already set up
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def classify_query(query):
    """
    Look at the user's question and decide which route it needs:
    - simple_factual: a normal single-topic question
    - comparison: asks to compare two things (papers, authors, mechanisms)
    - out_of_scope: unrelated to lithium-sulfur batteries
    """
    prompt = f"""Classify the following question into EXACTLY one category. Reply with ONLY the category name, nothing else.

Categories:
- simple_factual: a single, direct question about lithium-sulfur battery research
- comparison: asks to compare two or more things (papers, authors, mechanisms, methods)
- out_of_scope: not related to lithium-sulfur batteries at all

Question: {query}

Category:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # We want this to be deterministic, not creative
    )

    category = response.choices[0].message.content.strip().lower()
    return category


# Quick test
if __name__ == "__main__":
    test_questions = [
        "What causes capacity fade in lithium-sulfur batteries?",
        "Compare the capacity fade mechanisms proposed by Kumaresan et al. vs Hofmann et al.",
        "What's the cycle life of sodium-sulfur batteries?"
    ]

    for q in test_questions:
        category = classify_query(q)
        print(f"Question: {q}")
        print(f"→ Classified as: {category}\n")