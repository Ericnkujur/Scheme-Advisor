"""
Structured representation of a scholarship/scheme's eligibility rules.

This is the extraction TARGET: an LLM reads a scheme guideline document
and fills in this schema. Once filled, all eligibility decisions are made
by plain Python logic in evaluator.py — the LLM never decides eligibility
directly. This separation is the core design choice of JanSahayak.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Operator(str, Enum):
    LTE = "<="
    LT = "<"
    GTE = ">="
    GT = ">"
    EQ = "=="


class NumericCondition(BaseModel):
    """A numeric eligibility condition, e.g. age <= 25, income <= 250000."""
    operator: Operator
    value: float


class Category(str, Enum):
    GENERAL = "General"
    OBC = "OBC"
    SC = "SC"
    ST = "ST"
    EWS = "EWS"
    MINORITY = "Minority"
    ANY = "Any"  # scheme does not restrict by category


class EducationLevel(str, Enum):
    SCHOOL = "School"
    UNDERGRADUATE = "Undergraduate"
    POSTGRADUATE = "Postgraduate"
    PHD = "PhD"
    DIPLOMA = "Diploma"
    ANY = "Any"


class EligibilityRule(BaseModel):
    """
    Structured eligibility rules for a single scheme, extracted from its
    official guideline document. Every field is Optional because a scheme
    may not specify all of them — 'not specified' must stay distinguishable
    from 'no restriction', which is why we don't default to permissive
    values like None==no-limit silently. See evaluator.py for how this
    ambiguity is handled (it produces a "missing information" outcome,
    never a silent pass).
    """

    scheme_name: str
    state: Optional[str] = Field(
        default=None,
        description="State the scheme applies to. None means central/all-India scheme.",
    )
    age: Optional[NumericCondition] = None
    annual_family_income: Optional[NumericCondition] = None
    eligible_categories: list[Category] = Field(default_factory=lambda: [Category.ANY])
    education_level: EducationLevel = EducationLevel.ANY
    requires_domicile: bool = False
    requires_disability_certificate: bool = False
    minimum_percentage: Optional[NumericCondition] = Field(
        default=None,
        description="Minimum academic percentage/marks required in the previous qualifying exam.",
    )
    required_documents: list[str] = Field(default_factory=list)

    # Provenance — required for the "evidence-first" answer requirement.
    source_document: str
    source_section: Optional[str] = None
    application_url: Optional[str] = None
    last_verified_date: Optional[str] = Field(
        default=None,
        description="ISO date the source document was last checked/scraped, e.g. '2026-08-06'.",
    )


class UserProfile(BaseModel):
    """What we know about the applicant. Any field left as None is treated
    as 'unknown', not 'does not apply' — this is what drives the
    missing-information branch in the evaluator."""

    age: Optional[float] = None
    annual_family_income: Optional[float] = None
    category: Optional[Category] = None
    education_level: Optional[EducationLevel] = None
    state: Optional[str] = None
    has_domicile_certificate: Optional[bool] = None
    has_disability_certificate: Optional[bool] = None
    last_exam_percentage: Optional[float] = None
