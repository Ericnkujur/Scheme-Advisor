# JanSahayak

An evidence-first eligibility assistant for Indian government scholarships.
Instead of asking an LLM "am I eligible?", JanSahayak splits the problem in
two: an LLM extracts structured eligibility rules from official scheme
documents, and a **deterministic Python rule engine** — not the LLM — decides
whether a given student actually qualifies. The LLM only ever explains and
phrases the answer; it never makes the eligibility call itself.

**v1 scope:** 17 real central-government scholarship schemes, sourced from
scholarships.gov.in. Multi-turn chat interface with profile memory across
turns. No multilingual support, no hybrid search/reranking yet — see
"Roadmap" below.

## Why this design

A generic "chat with your scholarship PDFs" RAG bot can hallucinate an
eligibility decision that sounds plausible but is wrong — a serious problem
when the answer affects whether someone applies for real financial aid.
JanSahayak avoids this by construction: `rules/evaluator.py` contains zero
LLM calls. Every ELIGIBLE / INELIGIBLE / NEEDS_INFO verdict is the output of
plain Python comparisons against a structured rule, extracted once ahead of
time and reviewed by hand. The LLM's job — in `generate_answer()` — is
strictly to explain a verdict that's already been decided, cite its sources,
and ask a good follow-up question. It cannot override the verdict.

## Architecture

```
Scheme PDFs (guideline + FAQ)
        |
extract_pdf.py          - PyMuPDF text extraction
        |
extract_rules.py (LLM)  - raw text -> structured EligibilityRule (Pydantic)
        |
chunk.py + embed.py     - paragraph-aware chunking -> BAAI/bge-small-en-v1.5 -> ChromaDB
        |
User query -> parse_profile.py (LLM, extracts UserProfile from free text)
        |
retrieve.py (embedding search, anchored to the conversation, not just the last message)
        |
evaluator.py (NO LLM) - deterministic verdict per scheme
        |
answer.py (LLM) - explains verdicts, cites sources, asks one consolidated follow-up
        |
app.py - Streamlit chat UI
```

`batch_process.py` runs the whole pipeline (text extraction -> rule
extraction -> chunking) over every downloaded scheme in one pass, with
retry/backoff for rate limits and resumable skip-logic so a partial failure
doesn't cost you re-processing schemes that already succeeded.

## Stack

Python, Pydantic, Groq API (`openai/gpt-oss-120b`, OpenAI-compatible client),
sentence-transformers + ChromaDB for retrieval, Streamlit for the UI.
(Started on Gemini; switched to Groq for a more workable free-tier daily
request budget for iterative development.)

## Evaluation

`eval/run_eval.py` runs a hand-verified set of test queries
(`eval/test_questions.json`) against the real pipeline, with two tiers:

- **Free tier** (default, no API cost): retrieval recall@k and eligibility
  accuracy, computed directly from `retrieve()` and `evaluate()` - the part
  of the system where correctness doesn't depend on the LLM at all.
- **`--with-generation`** (costs API calls): additionally runs
  `parse_profile()` + `generate_answer()`, then scores each answer with an
  automated **LLM-as-judge faithfulness check** (`eval/faithfulness_judge.py`)
  - a second, independent model call that flags any specific factual claim
  (number, threshold, document name) not grounded in either the retrieved
  passages or the structured eligibility data the evaluator produced.

### Results (7 hand-verified test queries, covering eligible/ineligible/
needs-info/category-mismatch/out-of-scope cases)

| Metric | Result |
|---|---|
| Retrieval recall@5 | 6/6 (100%) |
| Eligibility accuracy (deterministic, no LLM) | 10/10 scheme/verdict checks (100%) |
| Generation faithfulness (LLM-as-judge) | 5/7 (71%) |

**Two faithfulness failures found and understood, not hidden:**

1. **Over-generalization** - an answer stated that all three retrieved
   disability schemes "cover master's-level study," when the source
   guidelines actually support this for only two of the three. The model
   correctly judged all three schemes *relevant* to the question but
   overstated a shared property across all of them.
2. **Silence treated as exclusion** - for an out-of-scope query ("scholarships
   for studying abroad"), the answer stated the retrieved schemes
   "do not cover study-abroad programmes" as if that were an explicit
   documented exclusion, when the source documents simply never mention
   study-abroad at all - an inference, not a stated fact.

Both are subtle overreach-in-phrasing issues, not eligibility-logic errors -
every eligibility decision across all 7 test cases was correct. This
distinction (reasoning is sound; prose occasionally overstates certainty) is
deliberately called out rather than smoothed over, since it's the accurate
picture of where the system's remaining risk actually lives.

## Known limitations

- **Retrieval cannot cleanly distinguish "genuinely relevant" from
  "topically adjacent but wrong."** Because the whole corpus is
  scholarship-domain text, an out-of-scope query (e.g. "scholarships for
  studying abroad") still returns the closest available chunks by cosine
  distance, even though none are actually relevant - verified by comparing
  real distance scores (relevant queries: ~0.51-0.59; the abroad query:
  ~0.62-0.64 - a real but narrow gap, not a clean separation). Mitigated at
  the generation layer with an explicit "say so plainly if nothing actually
  matches" instruction, not by filtering retrieval itself.
- **Extraction occasionally omits a real criterion present in the source
  text** - found one case (a disability-percentage threshold) that the LLM
  correctly cited from the raw PDF text but that isn't captured as a
  structured field in `EligibilityRule`, so the deterministic evaluator
  can't independently verify it. Documented as a schema-coverage gap, not
  fixed in v1.
- **Groq's free-tier daily token/request limits** were hit multiple times
  during development; `batch_process.py`'s retry/backoff and skip-on-success
  logic exist specifically to make interrupted batch runs resumable rather
  than needing a full restart.

## Roadmap (not built in v1)

- Hybrid retrieval (BM25 + embeddings) - would likely help the
  out-of-scope-query limitation above, since exact keyword mismatch is
  easier to detect than embedding distance alone on a narrow-domain corpus.
- Multilingual support (Hindi/Hinglish).
- Structured extraction of additional criteria found during eval (e.g.
  disability-percentage thresholds).
- Document-version/change detection.

## Running it

```bash
uv venv && .venv\Scripts\activate      # or source .venv/bin/activate
uv pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`):
```
GROQ_API_KEY=your_key_here
```

**Full pipeline, from scratch:**
```bash
# 1. Fill in data/scheme_manifest.json with real scheme guideline/FAQ URLs
python ingestion/download_manifest.py --manifest data/scheme_manifest.json

# 2. Extract text, rules, and chunks for everything downloaded
python ingestion/batch_process.py

# 3. Build the embedding index
python retrieval/embed.py --chunks data/processed/*_chunks.json --db-path data/chroma_db

# 4. Run the app
streamlit run app.py
```

**Run the eval suite:**
```bash
python eval/run_eval.py                    # free - retrieval + eligibility accuracy
python eval/run_eval.py --with-generation   # costs API calls - adds faithfulness judge
```

## Status

- [x] `rules/schema.py` - Pydantic models for eligibility rules + user profile
- [x] `rules/evaluator.py` - deterministic eligibility logic, zero LLM calls
- [x] `rules/extract_rules.py` - LLM extraction into structured rules
- [x] `rules/parse_profile.py` - LLM extraction of user profile from free text
- [x] `ingestion/` - PDF collection, text extraction, chunking, batch processing
- [x] `retrieval/` - embedding + top-k retrieval, conversation-anchored
- [x] `generation/answer.py` - cited, verdict-respecting answer generation
- [x] `app.py` - multi-turn Streamlit chat interface with profile memory
- [x] `eval/` - scripted retrieval/eligibility metrics + automated faithfulness judge
- [ ] Deployment (Streamlit Community Cloud)
- [ ] Hybrid retrieval, multilingual support (see Roadmap)