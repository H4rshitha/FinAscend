"""Company size -> plan -> entitlements.

ONE SOURCE OF TRUTH
-------------------
This module is the only place the tier matrix is written down. The API serves
it to the frontend from here (`GET /auth/me` carries the resolved capability
list), so the UI never hard-codes which plan gets what. A capability added here
appears in both halves of the product at once, and the two can never drift into
disagreeing about what a customer paid for.

WHAT THE TIERS GATE, AND WHY THAT AXIS
--------------------------------------
Tiers gate **analytical depth**, not modules and not row counts. Every plan
gets the whole product: cash health, the payment plan, customers, receipts.
What changes is how far you can open the working underneath a number.

That axis was chosen because it is the one that tracks who is actually on the
other side of the screen. A sole trader wants to know whether payroll clears
this month; a finance lead at a 200-person firm wants to interrogate the
interval calibration before trusting the runway figure. Gating whole modules
instead would take the payment-priority help away from the smallest businesses,
who need it most — they are the ones with a real shortfall to allocate.

IMPORTANT: gating is a PRODUCT boundary, not a security boundary for the
numbers themselves. Every plan sees honest figures; higher plans see more of
the derivation. Nothing here ever changes a computed value, and no tier is
shown a rosier number than another — that would make the tier a lie about the
business rather than a difference in access.
"""

from __future__ import annotations

from enum import Enum


class CompanySize(str, Enum):
    """Self-declared at signup. Maps 1:1 to a plan today, but kept separate
    because size is a fact about the business and plan is a commercial
    decision — conflating them makes 'upgrade without re-registering' awkward
    the first time someone asks for it."""

    SOLO = "solo"        # 1-4 people
    SMALL = "small"      # 5-49
    MEDIUM = "medium"    # 50-249
    LARGE = "large"      # 250+


COMPANY_SIZE_LABELS: dict[CompanySize, dict[str, str]] = {
    CompanySize.SOLO: {
        "label": "Just me",
        "headcount": "1–4 people",
        "hint": "Sole trader or a very small team",
    },
    CompanySize.SMALL: {
        "label": "Small business",
        "headcount": "5–49 people",
        "hint": "A growing team, usually without a full finance function",
    },
    CompanySize.MEDIUM: {
        "label": "Mid-sized",
        "headcount": "50–249 people",
        "hint": "You have someone who owns the numbers",
    },
    CompanySize.LARGE: {
        "label": "Large",
        "headcount": "250+ people",
        "hint": "A finance team that will want to audit the method",
    },
}


class Plan(str, Enum):
    ESSENTIALS = "essentials"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class Capability(str, Enum):
    """Named capabilities. The UI checks these, never the plan name directly —
    so re-packaging the plans later is a change in this file alone."""

    # Always on, for every plan. Listed explicitly rather than assumed, so the
    # entitlement payload is a complete description of what the user can do.
    CASH_HEALTH = "cash_health"
    ACTION_PLAN = "action_plan"
    RECEIPT_CAPTURE = "receipt_capture"
    CUSTOMER_LIST = "customer_list"

    # Depth: opening the working under a headline number.
    METHOD_PANELS = "method_panels"
    SCENARIO_EXPLORER = "scenario_explorer"
    SOLVER_COMPARISON = "solver_comparison"
    CREDIT_EXPLAINABILITY = "credit_explainability"

    # Depth: the full evidence base.
    BACKTEST_HISTORY = "backtest_history"
    AUDIT_LOG = "audit_log"
    AUDIT_EXPORT = "audit_export"


_BASE: set[Capability] = {
    Capability.CASH_HEALTH,
    Capability.ACTION_PLAN,
    Capability.RECEIPT_CAPTURE,
    Capability.CUSTOMER_LIST,
}

_DEPTH: set[Capability] = {
    Capability.METHOD_PANELS,
    Capability.SCENARIO_EXPLORER,
    Capability.SOLVER_COMPARISON,
    Capability.CREDIT_EXPLAINABILITY,
}

_EVIDENCE: set[Capability] = {
    Capability.BACKTEST_HISTORY,
    Capability.AUDIT_LOG,
    Capability.AUDIT_EXPORT,
}

PLAN_CAPABILITIES: dict[Plan, set[Capability]] = {
    Plan.ESSENTIALS: set(_BASE),
    Plan.PROFESSIONAL: _BASE | _DEPTH,
    Plan.ENTERPRISE: _BASE | _DEPTH | _EVIDENCE,
}

PLAN_LABELS: dict[Plan, dict[str, str]] = {
    Plan.ESSENTIALS: {
        "label": "Essentials",
        "tagline": "The answer, in plain language",
    },
    Plan.PROFESSIONAL: {
        "label": "Professional",
        "tagline": "The answer, plus how it was worked out",
    },
    Plan.ENTERPRISE: {
        "label": "Enterprise",
        "tagline": "Everything, including the evidence it was tested against",
    },
}

# Size -> the plan a new signup starts on.
DEFAULT_PLAN_FOR_SIZE: dict[CompanySize, Plan] = {
    CompanySize.SOLO: Plan.ESSENTIALS,
    CompanySize.SMALL: Plan.ESSENTIALS,
    CompanySize.MEDIUM: Plan.PROFESSIONAL,
    CompanySize.LARGE: Plan.ENTERPRISE,
}


def capabilities_for(plan: Plan) -> list[str]:
    """Sorted capability strings for a plan. Sorted so the payload is stable
    and a diff between two responses is meaningful."""
    return sorted(c.value for c in PLAN_CAPABILITIES[plan])


def plan_for_size(size: CompanySize) -> Plan:
    return DEFAULT_PLAN_FOR_SIZE[size]


def has_capability(plan: Plan, capability: Capability) -> bool:
    return capability in PLAN_CAPABILITIES[plan]


def describe_plans() -> list[dict]:
    """The full matrix, for a pricing/upgrade screen. Served rather than
    duplicated in the frontend."""
    return [
        {
            "plan": p.value,
            "label": PLAN_LABELS[p]["label"],
            "tagline": PLAN_LABELS[p]["tagline"],
            "capabilities": capabilities_for(p),
        }
        for p in (Plan.ESSENTIALS, Plan.PROFESSIONAL, Plan.ENTERPRISE)
    ]
