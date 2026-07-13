"""
Layer 3 — groundedness / hallucination-resistance aggregation.

Reads tests/eval_results_run*.json and reports the three-bucket breakdown
(answered-grounded / refused / answered-ungrounded) plus router and retrieval
metrics, as mean [min-max] across runs so a single-run number is never mistaken
for a stable one.
"""
import glob
import json
import os
import statistics

OOS_PREFIX = "out_of_scope"


def load_runs(pattern="tests/eval_results_run*.json"):
    return [(p, json.load(open(p, encoding="utf-8"))) for p in sorted(glob.glob(pattern))]


def score_run(cases):
    n = len(cases)
    router_correct = sum(1 for c in cases if c["route_correct"])

    graded = [c for c in cases if c["recall"] is not None]          # gold-bearing
    mean_recall = statistics.mean(c["recall"] for c in graded)
    mean_precision = statistics.mean(c["precision"] for c in graded)

    critic = [c for c in cases if not c["category"].startswith(OOS_PREFIX)]
    refused = [c for c in critic if c["refused"]]
    answered = [c for c in critic if not c["refused"]]
    grounded = [c for c in answered if c["verified"]]
    ungrounded = [c for c in answered if not c["verified"]]

    retr_fail = [c for c in graded if c["recall"] == 0.0]
    hall_on_fail = [c for c in retr_fail if not c["refused"] and not c["verified"]]

    return {
        "router_acc": router_correct / n,
        "recall": mean_recall,
        "precision": mean_precision,
        "n_critic": len(critic),
        "grounded": len(grounded),
        "refused": len(refused),
        "ungrounded": len(ungrounded),
        "refusal_rate": len(refused) / len(critic),
        "grounding_rate": (len(grounded) / len(answered)) if answered else None,
        "n_retr_fail": len(retr_fail),
        "hall_on_fail": len(hall_on_fail),
    }


def agg(vals):
    vals = [v for v in vals if v is not None]
    return (statistics.mean(vals), min(vals), max(vals)) if vals else None


def pct(t):
    if t is None:
        return "n/a"
    m, lo, hi = t
    return f"{m*100:.1f}%" if lo == hi else f"{m*100:.1f}% [{lo*100:.1f}-{hi*100:.1f}%]"


def cnt(vals):
    m, lo, hi = statistics.mean(vals), min(vals), max(vals)
    return f"{m:.1f}" if lo == hi else f"{m:.1f} [{lo}-{hi}]"


def main():
    runs = load_runs()
    if not runs:
        print("No tests/eval_results_run*.json files found.")
        return
    scores = [score_run(cs) for _, cs in runs]

    print("=" * 60)
    print(f"LAYER 3 - GROUNDEDNESS  ({len(runs)} runs)")
    print("=" * 60)
    for (p, _), s in zip(runs, scores):
        print(f"  {os.path.basename(p)}: grounded={s['grounded']} "
              f"refused={s['refused']} ungrounded={s['ungrounded']} "
              f"(of {s['n_critic']} critic-run)")
    print()
    print(f"  Router accuracy:            {pct(agg([s['router_acc'] for s in scores]))}")
    print(f"  Retrieval recall:           {pct(agg([s['recall'] for s in scores]))}")
    print(f"  Retrieval precision:        {pct(agg([s['precision'] for s in scores]))}")
    print()
    print(f"  Answered & grounded:        {cnt([s['grounded'] for s in scores])}")
    print(f"  Refused:                    {cnt([s['refused'] for s in scores])}")
    print(f"  Answered & UNGROUNDED:      {cnt([s['ungrounded'] for s in scores])}   <- hallucinations")
    print(f"  Refusal rate:               {pct(agg([s['refusal_rate'] for s in scores]))}")
    print(f"  Grounding rate (of answered): {pct(agg([s['grounding_rate'] for s in scores]))}")
    print()
    print(f"  Retrieval-failure cases:    {cnt([s['n_retr_fail'] for s in scores])}")
    print(f"  ...that hallucinated:       {cnt([s['hall_on_fail'] for s in scores])}")
    print("=" * 60)


if __name__ == "__main__":
    main()