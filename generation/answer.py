"""
Generate the final answer shown to the user.

This is the ONLY place retrieval output and the evaluator's verdict get
turned into natural language. The LLM is given the evaluator's verdict as
a fact, not asked to re-derive it — it phrases and explains, it does not
decide. This keeps the "never just say you're eligible" product
requirement enforced by code, not by prompting alone.

Uses Groq (free tier, no card required). Requires GROQ_API_KEY env var
(get one free at https://console.groq.com/keys).

Usage (as a library):
    from generation.answer import generate_answer
    text = generate_answer(query, retrieved_chunks, eligibility_results)
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

if TYPE_CHECKING:
    from retrieval.retrieve import RetrievedChunk
    from rules.evaluator import EligibilityResult

MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SYSTEM_PROMPT = """You are JanSahayak, an assistant that explains Indian government \
scholarship eligibility to students.

You will be given:
1. The user's question
2. Retrieved passages from official scheme guideline documents
3. Structured eligibility verdicts already computed by a rule engine (NOT by you)

Your job is only to explain and present this information clearly — you do NOT \
decide eligibility yourself. The verdicts given to you (ELIGIBLE / INELIGIBLE / \
NEEDS_INFO) are final; never override them, never say someone "is eligible" if \
the verdict says NEEDS_INFO or INELIGIBLE.

Rules for your response:
- Keep it conversational, like a back-and-forth chat — not a report.
- If most/all schemes need the same missing detail (e.g. income), don't repeat
  "we need X" for every scheme — ask for it ONCE, clearly, as a direct question
  at the end, and briefly note which schemes it would unlock.
- For schemes that are clearly INELIGIBLE, state that briefly and move on —
  don't dwell on them.
- For ELIGIBLE schemes, celebrate that clearly before anything else.
- Don't dump a formal per-scheme checklist unless the user asks for full details.
- Cite the source document for any factual claim about a scheme's rules.
- Never invent eligibility criteria not present in the retrieved passages.
- End with the standard disclaimer: this is a preliminary assessment, not an \
  official approval — the user should verify directly on scholarships.gov.in \
  or the relevant state portal before applying.
- Keep the tone plain and helpful, not bureaucratic.
- If multiple schemes are missing different fields, ask about whichever single \
  field would resolve the most schemes at once — not just the first one you notice.
"""


def _get_client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("GROQ_API_KEY"), base_url=GROQ_BASE_URL)


def _format_chunks(chunks: list["RetrievedChunk"]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"[Source: {c.source_document} | Scheme: {c.scheme_name}]\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


def _format_verdicts(results: list["EligibilityResult"]) -> str:
    parts = []
    for r in results:
        lines = [f"Scheme: {r.scheme_name}", f"Verdict: {r.verdict.value.upper()}"]
        for c in r.conditions:
            lines.append(f"  - {c.label}: {c.status.value} ({c.detail})")
        if r.missing_fields:
            lines.append(f"  Missing info needed: {', '.join(r.missing_fields)}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def generate_answer(
    query: str,
    retrieved_chunks: list["RetrievedChunk"],
    eligibility_results: list["EligibilityResult"],
    client: OpenAI | None = None,
    model: str = MODEL_NAME,
) -> str:
    client = client or _get_client()

    user_prompt = f"""User's question: {query}

Retrieved passages:
{_format_chunks(retrieved_chunks)}

Computed eligibility verdicts (already decided by the rule engine — do not \
change these, only explain them):
{_format_verdicts(eligibility_results)}

Write the response to the user now."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1200,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Smoke test with fake data (no retrieval/evaluator dependency needed to run this)
    from dataclasses import dataclass

    @dataclass
    class FakeChunk:
        text: str
        scheme_name: str
        source_document: str

    @dataclass
    class FakeCondition:
        label: str
        status: str
        detail: str

    @dataclass
    class FakeVerdict:
        scheme_name: str
        verdict: str
        conditions: list
        missing_fields: list

        class _V:
            def __init__(self, v): self.value = v
        def __post_init__(self):
            self.verdict = FakeVerdict._V(self.verdict)

    chunks = [
        FakeChunk(
            text="Applicants must be permanent residents of Haryana. Age must not "
                 "exceed 25. Annual family income must not exceed Rs. 2,50,000. "
                 "Open to SC, OBC, and EWS categories only.",
            scheme_name="Haryana Post-Matric Scholarship",
            source_document="Guidelines_Haryana_PostMatric_2026.pdf",
        )
    ]
    verdicts = [
        FakeVerdict(
            scheme_name="Haryana Post-Matric Scholarship",
            verdict="needs_info",
            conditions=[
                FakeCondition("age", "pass", "required <= 25, user has 22"),
                FakeCondition("annual_family_income", "pass", "required <= 250000, user has 200000"),
                FakeCondition("category", "missing", "user's category unknown"),
            ],
            missing_fields=["category"],
        )
    ]

    print(generate_answer(
        "Am I eligible for scholarships in Haryana?",
        chunks,
        verdicts,
    ))
