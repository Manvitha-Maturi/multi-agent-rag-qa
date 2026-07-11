from dataclasses import dataclass, field
from src.router import classify_query
from src.qa import retrieve_chunks, retrieve_hybrid, answer_with_verification


@dataclass
class QAResult:
    question: str
    route: str
    answer: str = ""
    sources: list = field(default_factory=list)
    verified: bool = False
    retries: int = 0
    trace: list = field(default_factory=list)


RETRIEVAL_CONFIG = {
    "simple_factual": {"k": 4, "hybrid": False},
    "comparison":     {"k": 8, "hybrid": True},
}


def run_pipeline(question: str) -> QAResult:
    result = QAResult(question=question, route="")

    # Stage 1: routing
    route = classify_query(question)
    result.route = route
    result.trace.append(f"router: classified as '{route}'")

    # Stage 2: early exit
    if route == "out_of_scope":
        result.answer = (
            "This question is outside the scope of the lithium-sulfur "
            "battery corpus, so I can't answer it from these documents."
        )
        result.trace.append("orchestrator: early exit, no retrieval or generation")
        return result

    # Stage 3: route-aware retrieval
    config = RETRIEVAL_CONFIG.get(route, {"k": 4, "hybrid": False})
    if config["hybrid"]:
        chunks = retrieve_hybrid(question, k=config["k"])
        result.trace.append(f"retrieval: hybrid, k={config['k']}, got {len(chunks)} chunks")
    else:
        chunks = retrieve_chunks(question, k=config["k"])
        result.trace.append(f"retrieval: semantic, k={config['k']}, got {len(chunks)} chunks")

    # Stage 4: generation + verification
    answer, verify_result, retries = answer_with_verification(question, chunks)
    result.answer = answer
    result.verified = verify_result["overall_grounded"]
    result.retries = retries
    result.sources = sorted(set(c["source"] for c in chunks))
    result.trace.append(f"critic: grounded={result.verified}, retries={retries}")

    return result


if __name__ == "__main__":
    for q in ["What is the capital of France?",
              "What cathode materials are discussed?",
              "Compare the capacity fade mechanisms discussed by different authors"]:
        r = run_pipeline(q)
        print(f"\nQ: {q}")
        print(f"Route: {r.route}")
        print(f"Verified: {r.verified}  Retries: {r.retries}")
        print(f"Sources: {r.sources}")
        print(f"Answer: {r.answer[:200]}{'...' if len(r.answer) > 200 else ''}")
        for line in r.trace:
            print(f"  trace: {line}")