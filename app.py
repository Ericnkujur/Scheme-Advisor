"""
Streamlit UI for JanSahayak v1.

Flow: user fills in their profile + a free-text question -> retrieve
relevant chunks -> for each scheme that shows up, load its structured
rule (extracted ahead of time via rules/extract_rules.py) and run the
deterministic evaluator -> pass chunks + verdicts to generation/answer.py
-> display the final answer with sources.

Run with:
    streamlit run app.py

Expects:
- A built Chroma index at data/chroma_db (see retrieval/embed.py)
- One JSON file per scheme under data/rules/, each containing an
  EligibilityRule (see rules/extract_rules.py --out ...)
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import streamlit as st

from rules.schema import EligibilityRule, UserProfile, Category, EducationLevel
from rules.evaluator import evaluate
from retrieval.retrieve import retrieve
from generation.answer import generate_answer

RULES_DIR = Path("data/rules")
DB_PATH = "data/chroma_db"


@st.cache_data
def load_rules() -> dict[str, EligibilityRule]:
    """scheme_name -> EligibilityRule, loaded from data/rules/*.json"""
    rules = {}
    if not RULES_DIR.exists():
        return rules
    for path in RULES_DIR.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rule = EligibilityRule.model_validate(data)
        rules[path.stem] = rule
    return rules


def build_profile_from_sidebar() -> UserProfile:
    st.sidebar.header("Your profile")
    st.sidebar.caption("Leave anything blank if you're not sure — the assistant will tell you what's missing.")

    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=0)
    income = st.sidebar.number_input("Annual family income (₹)", min_value=0, value=0, step=10000)
    category = st.sidebar.selectbox(
        "Category", ["Not specified"] + [c.value for c in Category if c != Category.ANY]
    )
    education_level = st.sidebar.selectbox(
        "Education level", ["Not specified"] + [e.value for e in EducationLevel if e != EducationLevel.ANY]
    )
    state = st.sidebar.text_input("State of residence")
    domicile = st.sidebar.selectbox("Have domicile certificate?", ["Not specified", "Yes", "No"])

    return UserProfile(
        age=age if age > 0 else None,
        annual_family_income=income if income > 0 else None,
        category=Category(category) if category != "Not specified" else None,
        education_level=EducationLevel(education_level) if education_level != "Not specified" else None,
        state=state or None,
        has_domicile_certificate=(
            None if domicile == "Not specified" else domicile == "Yes"
        ),
    )


def main():
    st.set_page_config(page_title="JanSahayak", page_icon="🎓")
    st.title("JanSahayak")
    st.caption("Find government scholarships you may be eligible for — with sources, not guesses.")

    if not os.environ.get("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY is not set — answer generation will fail until it's set in your environment.")

    rules_by_scheme = load_rules()
    if not rules_by_scheme:
        st.error(
            "No extracted scheme rules found in data/rules/. Run rules/extract_rules.py "
            "for at least one scheme first."
        )
        st.stop()

    profile = build_profile_from_sidebar()

    query = st.text_input(
        "Ask a question",
        placeholder="e.g. What scholarships am I eligible for as an OBC student in Haryana?",
    )
    top_k = st.sidebar.slider("Number of results to consider", 1, 10, 5)

    if st.button("Search", type="primary") and query:
        with st.spinner("Retrieving relevant schemes..."):
            try:
                chunks = retrieve(query, DB_PATH, top_k=top_k)
            except Exception as e:
                st.error(f"Retrieval failed — has the index been built? ({e})")
                st.stop()

        if not chunks:
            st.info("No relevant schemes found for that query.")
            st.stop()

        # Evaluate eligibility for each distinct scheme that showed up in retrieval
        seen_schemes = {c.scheme_name for c in chunks}
        results = []
        for scheme_name in seen_schemes:
            rule = rules_by_scheme.get(scheme_name)
            if rule is None:
                continue  # retrieved chunk belongs to a scheme we haven't extracted rules for yet
            results.append(evaluate(rule, profile))

        seen_slugs = {c.scheme_slug for c in chunks}
        results = []
        for slug in seen_slugs:
            rule = rules_by_scheme.get(slug)
            if rule is None:
                continue
            results.append(evaluate(rule, profile))

        if not results:
            st.warning(
                "Found relevant text but no structured rules for these schemes yet — "
                "run extract_rules.py for them first."
            )
            st.stop()

        with st.spinner("Preparing your answer..."):
            try:
                answer = generate_answer(query, chunks, results)
            except Exception as e:
                st.error(f"Answer generation failed: {e}")
                st.stop()

        st.markdown(answer)

        with st.expander("Raw eligibility verdicts (for debugging)"):
            for r in results:
                st.text(r.summary())

        with st.expander("Retrieved source passages"):
            for c in chunks:
                st.markdown(f"**{c.scheme_name}** — `{c.source_document}` (distance: {c.distance:.3f})")
                st.text(c.text[:500])
                st.divider()


if __name__ == "__main__":
    main()
