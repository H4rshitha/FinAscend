"""Shared Pydantic v2 base and primitive types.

Per Section 2 of the architecture plan: ORM-friendly, enums serialize as
values, unknown fields rejected. Rejecting unknown fields is deliberate — a
silently-absorbed typo in a field name is how a schema stops describing what
the code actually does.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# Money: Decimal, 14 significant digits / 2 decimal places. Decimal rather than
# float because binary floats cannot represent 0.01 exactly, and an optimizer
# that allocates cash across obligations must not accumulate representation
# error across a few hundred additions.
Money = Annotated[Decimal, Field(max_digits=14, decimal_places=2)]

# A probability or normalized score on [0, 1].
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]


class BaseSchema(BaseModel):
    """Common base for every FinAscend schema."""

    model_config = ConfigDict(
        from_attributes=True,      # ORM objects -> schema
        use_enum_values=True,      # enums serialize as their value
        extra="forbid",            # unknown fields are an error, not a shrug
        validate_assignment=True,
    )


class Currency(str, Enum):
    INR = "INR"
    USD = "USD"
    EUR = "EUR"


class SourceType(str, Enum):
    BANK_STATEMENT = "bank_statement"
    DIGITAL_INVOICE = "digital_invoice"
    RECEIPT_OCR = "receipt_ocr"
    DECENTRO_API = "decentro_api"
    PLAID_API = "plaid_api"
    MANUAL_ENTRY = "manual_entry"


class ObligationDirection(str, Enum):
    PAYABLE = "payable"
    RECEIVABLE = "receivable"


class ObligationStatus(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    PAID = "paid"
    OVERDUE = "overdue"
    WRITTEN_OFF = "written_off"


class FlexibilityLevel(str, Enum):
    RIGID = "rigid"          # all-or-nothing: partial payment forbidden
    MODERATE = "moderate"
    FLEXIBLE = "flexible"


class ScenarioType(str, Enum):
    BASELINE = "baseline"
    BEST_CASE = "best_case"
    WORST_CASE = "worst_case"
    CUSTOM = "custom"


class DecisionActionType(str, Enum):
    PAY_NOW = "pay_now"
    PAY_PARTIAL = "pay_partial"
    RESCHEDULE = "reschedule"
    NEGOTIATE = "negotiate"
    ESCALATE_COLLECTION = "escalate_collection"
    DEFER = "defer"


class ReviewStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"


class UserRole(str, Enum):
    OWNER = "owner"
    ACCOUNTANT = "accountant"
    VIEWER = "viewer"
