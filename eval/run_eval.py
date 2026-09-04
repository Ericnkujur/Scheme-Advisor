"""
Eval harness for JanSahayak.

Two tiers, deliberately separated by cost:

1. FREE tier (default) — retrieval recall@k + eligibility accuracy. Both
   computed from retrieve() and evaluate() directly, no LLM call involved.
   Each test case in test_questions.json ships an explicit "profile" dict
   (not parsed from text) specifically so this tier is 100% deterministic
   and repeatable — it tests the part of the system where correctness
   doesn't depend on the LLM at all.

2. LLM-cost tier (--with-generation) — additionally runs parse_profile()
   and generate_answer() for each query, so you can eyeball whether the
   natural-language answer matches what the deterministic layer already
   proved correct. Costs Groq API calls, so it's opt-in.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --with-generation
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rules"))

from retrieval.retrieve import retrieve
from rules.schema import EligibilityRule, UserProfile
from rules.evaluator import evaluate

RULES_DIR = Path("data/rules")
DB_PATH = "data/chroma_db"
TEST_FILE = Path("eval/test_questions.json")


def load_rules() -> dict[str, EligibilityRule]:
    rules = {}
    for path in RULES_DIR.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        rules[path.stem] = EligibilityRule.model_validate(data)
    return rules


def load_test_questions() -> list[dict]:
    return json.loads(TEST_FILE.read_text(encoding="utf-8"))


def run(with_generation: bool = False, top_k: int = 5):
    rules_by_slug = load_rules()
    questions = load_test_questions()

    retrieval_hits = 0
    retrieval_total = 0
    verdict_correct = 0
    verdict_total = 0
    verdict_failures = []

    print(f"Running {len(questions)} eval queries (top_k={top_k})...\n")

    for q in questions:
        print(f"--- {q['id']} ---")
        print(f"Query: {q['query']}")

        # --- Retrieval recall ---
        chunks = retrieve(q["query"], DB_PATH, top_k=top_k)
        retrieved_slugs = {c.scheme_slug for c in chunks}
        expected_slugs = set(q["expected_relevant_slugs"])

        if expected_slugs:
            hit = bool(expected_slugs & retrieved_slugs)
            retrieval_hits += int(hit)
            retrieval_total += 1
            status = "PASS" if hit else "FAIL"
            print(f"  [Retrieval {status}] expected one of {expected_slugs} in top-{top_k}")
        else:
            print(f"  [Retrieval, out-of-scope query] retrieved: {retrieved_slugs or '(none)'}")

        # --- Eligibility accuracy (deterministic, no LLM) ---
        profile = UserProfile.model_validate(q.get("profile", {}))
        for slug, expected_verdict in q.get("expected_verdicts", {}).items():
            rule = rules_by_slug.get(slug)
            if rule is None:
                print(f"  [Eligibility SKIP] no rule file found for slug '{slug}'")
                continue
            result = evaluate(rule, profile)
            actual_verdict = result.verdict.value
            verdict_total += 1
            if actual_verdict == expected_verdict:
                verdict_correct += 1
                print(f"  [Verdict PASS] {slug}: expected={expected_verdict}, got={actual_verdict}")
            else:
                verdict_failures.append((q["id"], slug, expected_verdict, actual_verdict))
                print(f"  [Verdict FAIL] {slug}: expected={expected_verdict}, got={actual_verdict}")
                print(f"    {result.summary()}")

        print()

    if with_generation:
        print("=== Generation output + automated faithfulness judge (costs Groq API calls) ===\n")
        from rules.parse_profile import parse_profile
        from generation.answer import generate_answer
        from eval.faithfulness_judge import judge_faithfulness

        faithfulness_scores = []

        for q in questions:
            chunks = retrieve(q["query"], DB_PATH, top_k=top_k)
            if not chunks:
                continue
            parsed = parse_profile(q["query"])
            seen_slugs = {c.scheme_slug for c in chunks}
            results = [
                evaluate(rules_by_slug[slug], parsed)
                for slug in seen_slugs if slug in rules_by_slug
            ]
            if not results:
                continue
            answer = generate_answer(q["query"], chunks, results)
            print(f"--- {q['id']} ---")
            print(f"notes: {q.get('notes', '')}")
            print(answer)

            verdict = judge_faithfulness(answer, chunks)
            faithfulness_scores.append((q["id"], verdict))
            print(f"\n  [Faithfulness judge] faithful={verdict['faithful']} — {verdict['reasoning']}")
            if verdict.get("unsupported_claims"):
                print(f"  Unsupported claims flagged: {verdict['unsupported_claims']}")
            print()

        if faithfulness_scores:
            faithful_count = sum(1 for _, v in faithfulness_scores if v["faithful"])
            print(f"=== Faithfulness summary: {faithful_count}/{len(faithfulness_scores)} answers judged fully faithful ===\n")

    print("=== Summary ===")
    if retrieval_total:
        print(f"Retrieval recall@{top_k}: {retrieval_hits}/{retrieval_total} "
              f"({100 * retrieval_hits / retrieval_total:.0f}%)")
    if verdict_total:
        print(f"Eligibility accuracy: {verdict_correct}/{verdict_total} "
              f"({100 * verdict_correct / verdict_total:.0f}%)")
    if verdict_failures:
        print("\nFailed verdict checks:")
        for qid, slug, expected, actual in verdict_failures:
            print(f"  - [{qid}] {slug}: expected {expected}, got {actual}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-generation", action="store_true",
                         help="Also run parse_profile + generate_answer for each query (costs API calls)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    run(with_generation=args.with_generation, top_k=args.top_k)


if __name__ == "__main__":
    main()