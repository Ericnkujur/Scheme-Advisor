# JanSahayak — v1 scaffold

Evidence-first eligibility assistant for Indian government scholarships.
v1 scope: one category (scholarships), central + Haryana state schemes, no
reranking/multilingual yet — see project notes for the v2/v3 roadmap.

## Pipeline (build/run order)

1. **Collect PDFs** — download ~15-20 scheme guideline PDFs from
   scholarships.gov.in into `data/raw_pdfs/`.
2. **Extract text** — `python ingestion/extract_pdf.py --input <pdf> --out <txt>`
3. **Extract structured rules** (LLM, one-time per scheme) —
   `python rules/extract_rules.py --input <txt> --source-doc <filename>`
   Requires `GEMINI_API_KEY` env var set (free key, no card: https://aistudio.google.com/apikey).
   Review the output JSON by hand — this is your ground truth, worth
   double-checking against the actual PDF.
4. **Evaluate eligibility** (no LLM, pure Python) — see `rules/evaluator.py`,
   run directly (`python rules/evaluator.py`) for a working example.
5. *(not yet built)* **retrieval/** — embed chunks, retrieve top-k relevant
   scheme text for a free-text query.
6. *(not yet built)* **generation/** — combine retrieved chunks + evaluator
   verdict into a final answer with citations.
7. *(not yet built)* **app.py** — Streamlit front end.

## Design principle

`rules/evaluator.py` never calls an LLM. `rules/extract_rules.py` is the
only place an LLM sees eligibility data, and its only job is to fill in
`rules/schema.py` — never to decide eligibility itself. This split is the
project's core differentiator; keep it intact as you build out retrieval
and generation.

## Status

- [x] schema.py — Pydantic models for rules + user profile
- [x] evaluator.py — deterministic eligibility logic (tested, 3 verdict paths)
- [x] extract_rules.py — LLM extraction into schema (untested against real PDFs yet)
- [x] extract_pdf.py — PDF -> text
- [x] chunk.py — paragraph-aware chunking with overlap
- [x] embed.py — BAAI/bge-small-en-v1.5 embeddings into local Chroma
- [x] retrieve.py — top-k retrieval by query
- [x] answer.py — final answer generation with citations (LLM explains, never overrides the verdict)
- [x] app.py — Streamlit UI wiring retrieve -> evaluate -> generate together
- [ ] eval set (10-15 hand-verified questions)

## Running the full app

1. Extract rules for each scheme PDF and save to `data/rules/<scheme>.json`:
   `python rules/extract_rules.py --input <text> --source-doc <name> --out data/rules/<scheme>.json`
2. Chunk + embed all schemes: `python ingestion/chunk.py ...` then `python retrieval/embed.py --chunks data/processed/*.json`
3. Set `GEMINI_API_KEY` in your environment (free tier, no card required)
4. `streamlit run app.py`
