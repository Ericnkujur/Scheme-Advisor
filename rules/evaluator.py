"""
Deterministic eligibility evaluation.

No LLM call happens in this file. Given a structured EligibilityRule
(extracted once, offline, from a scheme document) and a UserProfile,
this module decides — with plain comparisons — whether the user:

  - is ELIGIBLE on that condition
  - is INELIGIBLE on that condition
  - can't be determined because we don't know the relevant fact (MISSING)

The overall verdict for a scheme is:
  - ELIGIBLE      -> every condition checked out
  - INELIGIBLE    -> at least one condition definitively failed
  - NEEDS_INFO    -> no condition failed, but at least one couldn't be
                     checked due to missing user data

This mirrors the product requirement: never say "you are eligible"
outright — always show what was checked and what's still unverified.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

try:
    from schema import EligibilityRule, UserProfile, NumericCondition, Operator, Category
except ImportError:
    from rules.schema import EligibilityRule, UserProfile, NumericCondition, Operator, Category


class ConditionStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    MISSING = "missing"


class Verdict(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    NEEDS_INFO = "needs_info"


@dataclass
class ConditionResult:
    label: str
    status: ConditionStatus
    detail: str


@dataclass
class EligibilityResult:
    scheme_name: str
    verdict: Verdict
    conditions: list[ConditionResult] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    required_documents: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"{self.scheme_name}: {self.verdict.value.upper()}"]
        for c in self.conditions:
            lines.append(f"  [{c.status.value}] {c.label} — {c.detail}")
        return "\n".join(lines)


def _check_numeric(label: str, condition: NumericCondition | None,
                    actual: float | None) -> ConditionResult | None:
    """Returns None if the scheme has no such condition (nothing to check)."""
    if condition is None:
        return None
    if actual is None:
        return ConditionResult(
            label=label,
            status=ConditionStatus.MISSING,
            detail=f"scheme requires {label} {condition.operator.value} {condition.value}, "
                   f"but user's {label} is unknown",
        )

    ops = {
        Operator.LTE: actual <= condition.value,
        Operator.LT: actual < condition.value,
        Operator.GTE: actual >= condition.value,
        Operator.GT: actual > condition.value,
        Operator.EQ: actual == condition.value,
    }
    passed = ops[condition.operator]
    return ConditionResult(
        label=label,
        status=ConditionStatus.PASS if passed else ConditionStatus.FAIL,
        detail=f"required {condition.operator.value} {condition.value}, user has {actual}",
    )


def _check_category(rule: EligibilityRule, profile: UserProfile) -> ConditionResult | None:
    if Category.ANY in rule.eligible_categories:
        return None
    if profile.category is None:
        return ConditionResult(
            label="category",
            status=ConditionStatus.MISSING,
            detail=f"scheme restricted to {[c.value for c in rule.eligible_categories]}, "
                   f"user's category unknown",
        )
    passed = profile.category in rule.eligible_categories
    return ConditionResult(
        label="category",
        status=ConditionStatus.PASS if passed else ConditionStatus.FAIL,
        detail=f"scheme allows {[c.value for c in rule.eligible_categories]}, "
               f"user is {profile.category.value}",
    )

def _check_disability_certificate(rule: EligibilityRule, profile: UserProfile) -> ConditionResult | None:
    if not rule.requires_disability_certificate:
        return None
    if profile.has_disability_certificate is None:
        return ConditionResult(
            label="disability_certificate",
            status=ConditionStatus.MISSING,
            detail="scheme requires a disability certificate; not confirmed for user",
        )
    return ConditionResult(
        label="disability_certificate",
        status=ConditionStatus.PASS if profile.has_disability_certificate else ConditionStatus.FAIL,
        detail="disability certificate required and " +
               ("available" if profile.has_disability_certificate else "not available"),
    )

def _check_domicile(rule: EligibilityRule, profile: UserProfile) -> ConditionResult | None:
    if not rule.requires_domicile:
        return None
    if profile.has_domicile_certificate is None:
        return ConditionResult(
            label="domicile_certificate",
            status=ConditionStatus.MISSING,
            detail="scheme requires a domicile certificate; not confirmed for user",
        )
    return ConditionResult(
        label="domicile_certificate",
        status=ConditionStatus.PASS if profile.has_domicile_certificate else ConditionStatus.FAIL,
        detail="domicile certificate required and " +
               ("available" if profile.has_domicile_certificate else "not available"),
    )


def evaluate(rule: EligibilityRule, profile: UserProfile) -> EligibilityResult:
    checks = [
        _check_numeric("age", rule.age, profile.age),
        _check_numeric("annual_family_income", rule.annual_family_income,
                        profile.annual_family_income),
        _check_numeric("minimum_percentage", rule.minimum_percentage,
                        profile.last_exam_percentage),
        _check_category(rule, profile),
        _check_domicile(rule, profile),
        _check_disability_certificate(rule, profile),
    ]
    conditions = [c for c in checks if c is not None]

    if any(c.status == ConditionStatus.FAIL for c in conditions):
        verdict = Verdict.INELIGIBLE
    elif any(c.status == ConditionStatus.MISSING for c in conditions):
        verdict = Verdict.NEEDS_INFO
    else:
        verdict = Verdict.ELIGIBLE

    missing = [c.label for c in conditions if c.status == ConditionStatus.MISSING]

    return EligibilityResult(
        scheme_name=rule.scheme_name,
        verdict=verdict,
        conditions=conditions,
        missing_fields=missing,
        required_documents=rule.required_documents,
    )


if __name__ == "__main__":
    # Quick smoke test using a hand-built rule (stand-in for LLM-extracted output)
    rule = EligibilityRule(
        scheme_name="Haryana Post-Matric Scholarship (example)",
        state="Haryana",
        age=NumericCondition(operator=Operator.LTE, value=25),
        annual_family_income=NumericCondition(operator=Operator.LTE, value=250000),
        eligible_categories=[Category.SC, Category.OBC, Category.EWS],
        requires_domicile=True,
        required_documents=["income certificate", "domicile certificate", "student ID"],
        source_document="Guidelines_Haryana_PostMatric_2026.pdf",
        source_section="Section 3: Eligibility",
        application_url="https://scholarships.gov.in",
        last_verified_date="2026-08-06",
    )

    # Case 1: user profile missing category and domicile info
    profile_incomplete = UserProfile(
        age=22,
        annual_family_income=200000,
        state="Haryana",
    )
    print(evaluate(rule, profile_incomplete).summary())
    print()

    # Case 2: complete profile, ineligible on income
    profile_ineligible = UserProfile(
        age=22,
        annual_family_income=400000,
        category=Category.OBC,
        has_domicile_certificate=True,
    )
    print(evaluate(rule, profile_ineligible).summary())
    print()

    # Case 3: complete profile, eligible
    profile_eligible = UserProfile(
        age=22,
        annual_family_income=200000,
        category=Category.OBC,
        has_domicile_certificate=True,
    )
    print(evaluate(rule, profile_eligible).summary())
