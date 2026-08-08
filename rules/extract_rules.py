"""
Extract a structured EligibilityRule from raw scheme guideline text using
an LLM. This is the ONLY place an LLM touches eligibility data — its job
is extraction, not decision-making. Output is validated against the
Pydantic schema, so malformed extractions fail loudly instead of silently
producing a bad eligibility check downstream.

Uses Groq (free tier, no card required, generous daily request limits —
much friendlier than Gemini's free tier for a batch job like this one).
Requires GROQ_API_KEY env var (get one free at https://console.groq.com/keys).
Groq exposes an OpenAI-compatible API, so we use the `openai` client
pointed at Groq's base URL.

Usage:
    python extract_rules.py --input data/processed/scheme_001.txt \
                             --source-doc "Guidelines_XYZ_2026.pdf"
"""

from __future__ import annotations
import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
try:
    from schema import EligibilityRule
except ImportError:
    from rules.schema import EligibilityRule

load_dotenv()

SYSTEM_PROMPT = """You extract structured eligibility rules from Indian government \
scholarship/scheme guideline documents.

Return ONLY a single JSON object matching this shape, with no preamble, no \
markdown fences, no commentary:

{
  "scheme_name": string,
  "state": string or null (null = central/all-India scheme),
  "age": {"operator": "<="|"<"|">="|">"|"==", "value": number} or null,
  "annual_family_income": {"operator": ..., "value": number} or null,
  "eligible_categories": array of "General"|"OBC"|"SC"|"ST"|"EWS"|"Minority"|"Any",
  "education_level": "School"|"Undergraduate"|"Postgraduate"|"PhD"|"Diploma"|"Any",
  "requires_domicile": boolean,
  "requires_disability_certificate": boolean,
  "minimum_percentage": {"operator": ..., "value": number} or null,
  "required_documents": array of strings,
  "source_document": string (use the filename provided),
  "source_section": string or null (which section this came from, if identifiable),
  "application_url": string or null,
  "last_verified_date": string or null (ISO date, use the date provided)
}

Rules:
- If a field is not mentioned in the document, use null (numeric/optional fields)
  or "Any"/[] (categorical fields) — do NOT guess or infer values.
- eligible_categories defaults to ["Any"] only if the document truly does not
  restrict by category. If it restricts to specific categories, list exactly those.
- Be conservative: it is better to leave a field null than to fabricate a number.
"""

MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_INPUT_CHARS = 30000  

def _get_client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("GROQ_API_KEY"), base_url=GROQ_BASE_URL)


def extract_rule(
    raw_text: str,
    source_document: str,
    last_verified_date: str,
    client: OpenAI | None = None,
    model: str = MODEL_NAME,
) -> EligibilityRule:
    client = client or _get_client()

    if len(raw_text) > MAX_INPUT_CHARS:
        raw_text = raw_text[:MAX_INPUT_CHARS] + "\n\n[... document truncated for length ...]"

    user_prompt = (
        f"Source document filename: {source_document}\n"
        f"Last verified date: {last_verified_date}\n\n"
        f"Document text:\n---\n{raw_text}\n---"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    text_block = response.choices[0].message.content
    cleaned = text_block.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    data = json.loads(cleaned)
    return EligibilityRule.model_validate(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw scheme text file")
    parser.add_argument("--source-doc", required=True, help="Filename to record as source_document")
    parser.add_argument("--date", default=None, help="ISO date last verified (default: today)")
    parser.add_argument("--out", default=None, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    import datetime
    date = args.date or datetime.date.today().isoformat()

    with open(args.input, "r", encoding="utf-8") as f:
        raw_text = f.read()

    rule = extract_rule(raw_text, args.source_doc, date)

    output = rule.model_dump_json(indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Wrote {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()


