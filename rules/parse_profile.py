"""
Extract a UserProfile from the user's free-text question, so the assistant
can work from natural language ("I'm an SC student pursuing a master's")
instead of requiring a sidebar form to be filled in first.

This mirrors extract_rules.py's pattern: the LLM's only job is to map text
onto a fixed schema, never to make a decision. Fields it can't find in the
text stay None — evaluator.py already knows how to turn "None" into a
NEEDS_INFO verdict with a named missing field, so no special handling is
needed downstream for an incomplete profile.

Uses Groq (same provider as extract_rules.py / answer.py).
"""
from __future__ import annotations
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

try:
    from schema import UserProfile
except ImportError:
    from rules.schema import UserProfile

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SYSTEM_PROMPT = """You extract a student's profile from their free-text question, \
for an Indian government scholarship eligibility assistant.

Return ONLY a single JSON object matching this shape, with no preamble, no \
markdown fences, no commentary:

{
  "age": number or null,
  "annual_family_income": number or null,
  "category": "General"|"OBC"|"SC"|"ST"|"EWS"|"Minority" or null,
  "education_level": "School"|"Undergraduate"|"Postgraduate"|"PhD"|"Diploma" or null,
  "state": string or null,
  "has_domicile_certificate": boolean or null,
  "has_disability_certificate": boolean or null,
  "last_exam_percentage": number or null
}

Rules:
- Only fill in a field if the user's text actually states or clearly implies it.
  "pursuing a master's" -> education_level: "Postgraduate". "I'm SC" -> category: "SC".
  "income is 1.4 lakh" -> annual_family_income: 140000 (convert lakh/crore to plain numbers).
- If something is not mentioned, use null. Do NOT guess or default to a "typical" value.
- Do not infer has_domicile_certificate or has_disability_certificate unless explicitly
  mentioned — silence on these should stay null, not false.
"""


def _get_client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("GROQ_API_KEY"), base_url=GROQ_BASE_URL)


def parse_profile(
    query: str,
    client: OpenAI | None = None,
    model: str = MODEL_NAME,
) -> UserProfile:
    client = client or _get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    text_block = response.choices[0].message.content
    cleaned = text_block.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    data = json.loads(cleaned)
    return UserProfile.model_validate(data)


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "I'm an SC student pursuing a master's, annual family income is 1.4 lakh"
    profile = parse_profile(query)
    print(profile.model_dump_json(indent=2))