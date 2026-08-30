"""
Streamlit UI for JanSahayak v1.

Flow: chat-based. Each message is parsed for profile info, merged into
session state, used to retrieve + evaluate schemes, and answered
conversationally. Missing-field follow-ups render as quick-pick widgets
instead of requiring the user to type an answer — but only when retrieval
actually found something relevant (gated by RELEVANCE_CUTOFF), so the
widget doesn't show up alongside a "nothing matches" answer.

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
RELEVANCE_CUTOFF = 0.60  # tuned from real distance data on this corpus


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


def top_missing_fields(results: list, n: int = 2) -> list[str]:
    """Top N missing fields by how many schemes they'd unlock — asking about
    more than one per turn means the assistant covers ground faster instead
    of drip-feeding one question at a time."""
    from collections import Counter
    counts = Counter(f for r in results for f in r.missing_fields)
    return [f for f, _ in counts.most_common(n)]


def is_relevant(chunks: list) -> bool:
    """True only when retrieval's best match is close enough to trust —
    keeps the follow-up widget from appearing next to a 'nothing matches'
    answer (e.g. an out-of-scope query like 'study abroad')."""
    if not chunks:
        return False
    return min(c.distance for c in chunks) <= RELEVANCE_CUTOFF


def render_missing_field_widget(field: str, key_prefix: str) -> bool:
    """Renders a quick-pick widget for a single missing profile field.
    Updates st.session_state.profile directly and returns True if it did
    (caller should st.rerun() when this returns True)."""
    profile = st.session_state.profile

    if field == "category":
        st.write("Quick pick — what's your category?")
        cols = st.columns(4)
        options = ["General", "OBC", "SC", "ST"]
        for i, cat in enumerate(options):
            if cols[i].button(cat, key=f"{key_prefix}_cat_{cat}"):
                profile.category = Category(cat)
                return True
        cols2 = st.columns(2)
        for i, cat in enumerate(["EWS", "Minority"]):
            if cols2[i].button(cat, key=f"{key_prefix}_cat_{cat}"):
                profile.category = Category(cat)
                return True

    elif field == "annual_family_income":
        income = st.number_input(
            "Annual family income (₹)", min_value=0, step=10000, key=f"{key_prefix}_income"
        )
        if st.button("Submit income", key=f"{key_prefix}_income_submit"):
            profile.annual_family_income = income
            return True

    elif field == "age":
        age = st.number_input("Your age", min_value=0, max_value=100, key=f"{key_prefix}_age")
        if st.button("Submit age", key=f"{key_prefix}_age_submit"):
            profile.age = age
            return True

    elif field == "domicile_certificate":
        st.write("Do you have a domicile certificate?")
        c1, c2 = st.columns(2)
        if c1.button("Yes", key=f"{key_prefix}_domicile_yes"):
            profile.has_domicile_certificate = True
            return True
        if c2.button("No", key=f"{key_prefix}_domicile_no"):
            profile.has_domicile_certificate = False
            return True

    elif field == "disability_certificate":
        st.write("Do you have a disability certificate?")
        c1, c2 = st.columns(2)
        if c1.button("Yes", key=f"{key_prefix}_disability_yes"):
            profile.has_disability_certificate = True
            return True
        if c2.button("No", key=f"{key_prefix}_disability_no"):
            profile.has_disability_certificate = False
            return True

    elif field == "minimum_percentage":
        pct = st.number_input(
            "Your percentage in the qualifying exam", min_value=0.0, max_value=100.0,
            step=0.5, key=f"{key_prefix}_pct"
        )
        if st.button("Submit percentage", key=f"{key_prefix}_pct_submit"):
            profile.last_exam_percentage = pct
            return True

    return False


def merge_profiles(parsed: UserProfile, override: UserProfile) -> UserProfile:
    """Sidebar values win when the user actually set them (non-default);
    otherwise fall back to what was parsed from the question text."""
    merged_data = parsed.model_dump()
    override_data = override.model_dump()
    for key, value in override_data.items():
        if value is not None:
            merged_data[key] = value
    return UserProfile.model_validate(merged_data)


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
    if "pending_fields" not in st.session_state:
        st.session_state.pending_fields = []

    sidebar_profile = build_profile_from_sidebar()
    top_k = st.sidebar.slider("Number of results to consider", 1, 10, 5)

    with st.sidebar.expander("What I know about you so far"):
        st.json(st.session_state.profile.model_dump(exclude_none=True))
        if st.button("Reset conversation"):
            st.session_state.profile = UserProfile()
            st.session_state.messages = []
            st.session_state.pending_fields = []
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

            try:
                chunks = retrieve(query, DB_PATH, top_k=top_k)
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

            if chunks and results and is_relevant(chunks):
                st.session_state.pending_fields = top_missing_fields(results, n=2)
            else:
                st.session_state.pending_fields = []
            st.session_state.last_query = query

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

    if st.session_state.get("pending_fields"):
        with st.container(border=True):
            st.markdown("**📋 A couple more details could unlock more schemes:**")
            remaining = st.session_state.pending_fields
            answered_any = False
            still_pending = []
            for field in remaining:
                if render_missing_field_widget(field, key_prefix=f"pending_{field}"):
                    answered_any = True
                else:
                    still_pending.append(field)

            if answered_any:
                st.session_state.pending_fields = still_pending
                requery = st.session_state.last_query
                st.session_state.messages.append({"role": "user", "content": requery})
                try:
                    chunks = retrieve(requery, DB_PATH, top_k=top_k)
                    seen_slugs = {c.scheme_slug for c in chunks}
                    results = [
                        evaluate(rules_by_scheme[slug], st.session_state.profile)
                        for slug in seen_slugs if slug in rules_by_scheme
                    ]
                    answer = generate_answer(requery, chunks, results) if results else "No matching schemes."
                    if results and is_relevant(chunks):
                        st.session_state.pending_fields = top_missing_fields(results, n=2)
                    else:
                        st.session_state.pending_fields = []
                except Exception as e:
                    answer = f"Error: {e}"
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()


if __name__ == "__main__":
    main()