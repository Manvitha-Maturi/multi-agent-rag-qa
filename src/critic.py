import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def verify_answer(query, answer, chunks):
    """
    Check an answer against the chunks it was generated from.
    Returns a dict: {"claims": [...], "overall_grounded": bool}
    """
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    prompt = f"""You are a strict fact-checker. Break the ANSWER below into individual factual claims. For each claim, decide if it is DIRECTLY supported by the CONTEXT — not by outside knowledge, only what is written in the context.

Reply with ONLY valid JSON, no other text, in this exact format:
{{
  "claims": [
    {{"claim": "the specific claim text", "supported": true, "reason": "short reason"}}
  ],
  "overall_grounded": true
}}

overall_grounded should be true only if ALL claims are supported.

CONTEXT:
{context}

QUESTION: {query}

ANSWER TO CHECK:
{answer}

JSON:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    # Groq sometimes wraps JSON in ```json ... ``` fences — strip if present
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1) if raw.startswith("json\n") else raw

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print("[Critic] Failed to parse JSON. Raw output:")
        print(raw)
        # Fail safe: treat as ungrounded so the retry loop kicks in
        result = {"claims": [], "overall_grounded": False}

    return result


if __name__ == "__main__":
    from qa import ask

    query = "What causes capacity fade in lithium-sulfur batteries?"
    answer, sources, chunks = ask(query, k=3)

    print("Answer:\n", answer)
    print("\n" + "-"*60)

    result = verify_answer(query, answer, chunks)
    print("\nCritic result:")
    print(json.dumps(result, indent=2))