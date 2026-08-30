"""
Streamlit UI for JanSahayak v1.

Flow: chat-based. Each message is parsed for profile info via parse_profile,
merged into session state, then used to retrieve + evaluate schemes.

Retrieval is anchored to the conversation, not just the latest message:
a short follow-up like "the family income is 7 lakhs" is embedded on its
own very differently than "I want to pursue a master's" — anchoring keeps
retrieval on the same set of schemes across a multi-turn conversation
instead of silently drifting to unrelated ones each turn.

Run with:
    streamlit run app.py
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import streamlit as st

from rules.schema import EligibilityRule, UserProfile, Category, EducationLevel
from rules.evaluator import evaluate
from rules.parse_profile import parse_profile
from retrieval.retrieve import retrieve
from generation.answer import generate_answer

RULES_DIR = Path("data/rules")
DB_PATH = "data/chroma_db"


@st.cache_data
def load_rules() -> dict[str, EligibilityRule]:
    """scheme_slug -> EligibilityRule, loaded from data/rules/*.json"""
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
    st.sidebar.caption("Optional — the assistant reads your question first. Use this to correct or add details it missed.")

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


def merge_profiles(parsed: UserProfile, override: UserProfile) -> UserProfile:
    """Sidebar values win when the user actually set them (non-default);
    otherwise fall back to what was parsed from the question text."""
    merged_data = parsed.model_dump()
    override_data = override.model_dump()
    for key, value in override_data.items():
        if value is not None:
            merged_data[key] = value
    return UserProfile.model_validate(merged_data)


def build_retrieval_query(session_state, latest_query: str) -> str:
    """Anchors retrieval to the conversation's original intent instead of
    embedding only the latest message. Without this, a short follow-up
    like "the family income is 7 lakhs" retrieves whichever schemes happen
    to mention income most strongly — which can be entirely different
    schemes than the ones the conversation was actually about."""
    anchor = session_state.get("anchor_query")
    if not anchor:
        session_state.anchor_query = latest_query
        return latest_query
    if latest_query == anchor:
        return latest_query
    return f"{anchor} {latest_query}"


def main():
    st.set_page_config(page_title="JanSahayak", page_icon="🎓")
    st.title("🎓 JanSahayak")
    st.caption("Find government scholarships you may be eligible for — with sources, not guesses.")

    if not os.environ.get("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY is not set — profile parsing and answer generation will fail until it's set in your environment.")

    rules_by_scheme = load_rules()
    if not rules_by_scheme:
        st.error(
            "No extracted scheme rules found in data/rules/. Run rules/extract_rules.py "
            "for at least one scheme first."
        )
        st.stop()

    if "profile" not in st.session_state:
        st.session_state.profile = UserProfile()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "anchor_query" not in st.session_state:
        st.session_state.anchor_query = None

    sidebar_profile = build_profile_from_sidebar()
    top_k = st.sidebar.slider("Number of results to consider", 1, 10, 5)

    with st.sidebar.expander("What I know about you so far"):
        st.json(st.session_state.profile.model_dump(exclude_none=True))
        if st.button("Reset conversation"):
            st.session_state.profile = UserProfile()
            st.session_state.messages = []
            st.session_state.anchor_query = None
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask about scholarships, or answer a follow-up question...")

    if query and (
        not st.session_state.messages
        or st.session_state.messages[-1].get("content") != query
        or st.session_state.messages[-1].get("role") != "user"
    ):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Understanding..."):
                try:
                    newly_parsed = parse_profile(query)
                except Exception as e:
                    st.warning(f"Couldn't parse profile from your message ({e})")
                    newly_parsed = UserProfile()

            # Accumulate: keep everything already known, fill in new fields, sidebar overrides all
            st.session_state.profile = merge_profiles(
                merge_profiles(newly_parsed, st.session_state.profile),
                sidebar_profile,
            )
            profile = st.session_state.profile

            retrieval_query = build_retrieval_query(st.session_state, query)

            try:
                chunks = retrieve(retrieval_query, DB_PATH, top_k=top_k)
            except Exception as e:
                st.error(f"Retrieval failed — has the index been built? ({e})")
                st.stop()

            results = []
            if chunks:
                seen_slugs = {c.scheme_slug for c in chunks}
                results = [
                    evaluate(rules_by_scheme[slug], profile)
                    for slug in seen_slugs if slug in rules_by_scheme
                ]

            if not chunks:
                answer = "I couldn't find any schemes matching that — try rephrasing your question."
            elif not results:
                answer = "Found relevant text but no structured rules for these schemes yet."
            else:
                try:
                    answer = generate_answer(query, chunks, results)
                except Exception as e:
                    answer = f"Answer generation failed: {e}"

            st.markdown(answer)

            if chunks and results:
                with st.expander("Raw eligibility verdicts (for debugging)"):
                    for r in results:
                        st.text(r.summary())

                with st.expander("Retrieved source passages"):
                    for c in chunks:
                        st.markdown(f"**{c.scheme_name}** — `{c.source_document}` (distance: {c.distance:.3f})")
                        st.text(c.text[:500])
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()