"""
LLM-as-judge faithfulness check.

Given a generated answer, the retrieved passages, AND the structured
eligibility data computed by evaluate(), asks a second LLM call whether
every factual claim in the answer is grounded in one of those two inputs.

Both are legitimate sources of truth: raw passages come from the source
PDFs directly; eligibility_results come from evaluate() acting on rules
that were themselves extracted from those same PDFs in an earlier step
the judge doesn't see directly. A number is only "unsupported" if it
appears in NEITHER.
"""
from __future__ import annotations
import json
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

if TYPE_CHECKING:
    from retrieval.retrieve import RetrievedChunk

MODEL_NAME = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checker for a RAG system. You will be \
given: (1) source passages retrieved from documents, and (2) structured eligibility \
data that was separately computed by a deterministic rule engine (extracted ahead of \
time from the same source documents, via a different pipeline step you don't see here).

Both are legitimate grounding — a number, threshold, or document name is SUPPORTED if \
it appears in EITHER the source passages OR the structured eligibility data. Only flag \
a claim as unsupported if it appears in NEITHER.

Your job: identify any SPECIFIC factual claim in the answer (numbers, thresholds, \
percentages, document names, deadlines, amounts) that is not grounded in either \
input. General reasoning about eligibility verdicts (eligible/ineligible/needs more \
info) is fine to skip if it follows from the structured data — focus on concrete \
facts that don't trace back to either source.

Return ONLY a JSON object, no markdown fences, no commentary:
{
  "faithful": true or false,
  "unsupported_claims": ["list of specific unsupported claims, empty if none"],
  "reasoning": "one or two sentence explanation"
}

Be strict about claims grounded in NEITHER input, but do not flag claims that match \
the structured eligibility data even if they don't appear in the passages — that data \
is a legitimate, separate source of truth for this system.
"""


def _get_client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("GROQ_API_KEY"), base_url=GROQ_BASE_URL)


def _format_eligibility_data(eligibility_results: list) -> str:
    parts = []
    for r in eligibility_results:
        lines = [f"Scheme: {r.scheme_name}", f"Verdict: {r.verdict.value.upper()}"]
        for c in r.conditions:
            lines.append(f"  - {c.label}: {c.status.value} ({c.detail})")
        if r.missing_fields:
            lines.append(f"  Missing info: {', '.join(r.missing_fields)}")
        if r.required_documents:
            lines.append(f"  Required documents: {', '.join(r.required_documents)}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def judge_faithfulness(
    answer: str,
    retrieved_chunks: list["RetrievedChunk"],
    eligibility_results: list | None = None,
    client: OpenAI | None = None,
    model: str = MODEL_NAME,
) -> dict:
    client = client or _get_client()

    passages_text = "\n\n---\n\n".join(
        f"[Source: {c.source_document}]\n{c.text}" for c in retrieved_chunks
    )

    eligibility_text = (
        _format_eligibility_data(eligibility_results)
        if eligibility_results else "(none provided)"
    )

    user_prompt = f"""Source passages:
{passages_text}

Structured eligibility data (separately computed, also legitimate grounding):
{eligibility_text}

Answer to check:
{answer}

Judge this answer now."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    text = response.choices[0].message.content
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "faithful": None,
            "unsupported_claims": [],
            "reasoning": f"Judge response could not be parsed as JSON: {text[:200]}",
        }