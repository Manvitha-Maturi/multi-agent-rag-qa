"""
Evaluation harness for the multi-agent RAG QA system.
Runs every case in test_set.json through the pipeline and scores it.

Usage:
    python -m tests.run_eval
"""
import json
import os
from collections import defaultdict
from src.orchestrator import run_pipeline

TEST_SET_PATH = "tests/test_set.json"
RESULTS_PATH = "tests/eval_results.json"


def load_test_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def category_of(case_id):
    """Derive a category bucket from the id prefix for per-category scoring."""
    if case_id.startswith("sf_"):
        return "simple_factual"
    if case_id.startswith("cmp_vague_"):
        return "comparison_vague"
    if case_id.startswith("cmp_"):
        return "comparison"
    if case_id.startswith("oos_border_"):
        return "out_of_scope_border"
    if case_id.startswith("oos_"):
        return "out_of_scope"
    return "unknown"


def score_retrieval(expected_sources, actual_sources):
    """
    Compare expected vs actual source sets.
    Returns (precision, recall, expected_set, actual_set).
    Precision/recall are None when there's nothing to score against (empty expected).
    """
    expected = set(expected_sources)
    actual = set(actual_sources)

    if not expected:
        return None, None, expected, actual  # caller decides how to handle

    overlap = expected & actual
    recall = len(overlap) / len(expected)
    precision = len(overlap) / len(actual) if actual else 0.0
    return precision, recall, expected, actual


def load_existing_results(path):
    """Load prior results if present, so we can resume instead of restarting."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []  # corrupt/empty file -> start fresh
    return []


def evaluate(test_set, results_path):
    # Resume: load what's already been scored
    results = load_existing_results(results_path)
    done_ids = {r["id"] for r in results}
    if done_ids:
        print(f"Resuming: {len(done_ids)} case(s) already done, skipping those.\n")

    for case in test_set:
        case_id = case["id"]
        if case_id in done_ids:
            print(f"Skipping {case_id} (already done)")
            continue

        print(f"Running {case_id}: {case['question'][:60]}...")

        try:
            pipeline_result = run_pipeline(case["question"])
        except Exception as e:
            print(f"\n  STOPPED on {case_id}: {type(e).__name__}: {e}")
            print(f"  {len(results)} case(s) saved so far. Rerun the same command "
                  f"to resume once the issue clears.\n")
            break

        route_correct = (pipeline_result.route == case["expected_route"])
        precision, recall, exp_set, act_set = score_retrieval(
            case["expected_sources"], pipeline_result.sources
        )

        results.append({
            "id": case_id,
            "category": category_of(case_id),
            "question": case["question"],
            "expected_route": case["expected_route"],
            "actual_route": pipeline_result.route,
            "route_correct": route_correct,
            "expected_sources": sorted(exp_set),
            "actual_sources": sorted(act_set),
            "precision": precision,
            "recall": recall,
            "verified": pipeline_result.verified,
            "refused": pipeline_result.refused,
            "retries": pipeline_result.retries,
            "answer": pipeline_result.answer,
            "claims": pipeline_result.claims,
        })

        # Checkpoint: write after every successful case
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results

def report(results):
    # ===== Router accuracy =====
    print("\n" + "=" * 60)
    print("ROUTER ACCURACY")
    print("=" * 60)

    total = len(results)
    correct = sum(r["route_correct"] for r in results)
    print(f"\nOverall: {correct}/{total} = {correct/total:.1%}")

    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_cat[r["category"]]["correct"] += r["route_correct"]
        by_cat[r["category"]]["total"] += 1

    print("\nPer category:")
    for cat, s in sorted(by_cat.items()):
        print(f"  {cat:24s} {s['correct']}/{s['total']} = {s['correct']/s['total']:.0%}")

    misses = [r for r in results if not r["route_correct"]]
    if misses:
        print("\nMisclassified:")
        for r in misses:
            print(f"  [{r['id']}] expected '{r['expected_route']}', got '{r['actual_route']}'")
    else:
        print("\nNo routing misses.")

    # ===== Retrieval precision / recall =====
    print("\n" + "=" * 60)
    print("RETRIEVAL PRECISION / RECALL")
    print("=" * 60)

    scored = [r for r in results if r["precision"] is not None]

    if scored:
        avg_p = sum(r["precision"] for r in scored) / len(scored)
        avg_r = sum(r["recall"] for r in scored) / len(scored)
        print(f"\nScored on {len(scored)} cases with known expected sources:")
        print(f"  Mean precision: {avg_p:.1%}")
        print(f"  Mean recall:    {avg_r:.1%}")

        print("\nPer case:")
        for r in scored:
            print(f"  [{r['id']}] P={r['precision']:.0%} R={r['recall']:.0%}")
            if r["recall"] < 1.0:
                print(f"       expected: {r['expected_sources']}")
                print(f"       actual:   {r['actual_sources']}")

    vague = [r for r in results if r["category"] == "comparison_vague"]
    if vague:
        print("\nVague comparison cases (known extraction bug -- not scored):")
        for r in vague:
            print(f"  [{r['id']}] retrieved: {r['actual_sources'] or '(nothing)'}")


def main():
    test_set = load_test_set(TEST_SET_PATH)
    results = evaluate(test_set, RESULTS_PATH)
    report(results)
    print(f"\nResults in {RESULTS_PATH} ({len(results)} case(s))")


if __name__ == "__main__":
    main()