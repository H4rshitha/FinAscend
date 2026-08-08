"""Schemas for statement ingestion and business-level insolvency risk.

Two additions that the rest of the system implied but never carried:

**Statement ingestion.** The receipt path proves a document can be read. A bank
statement is a different problem — it is already structured, but structured
*differently by every bank*, and it carries an internal consistency check no
receipt has: a running balance. These schemas record which column was believed
to mean what, and what the reconciliation residual was, because a statement
parsed under the wrong column mapping produces a perfectly well-formed ledger
that is wrong in every row.

**Insolvency risk.** `RunwayAtRisk` answers "how long until the cash runs out."
That is not the same question as "how likely is this business to fail", and
conflating them is the mistake this module exists to avoid. RaR is a quantile
of a first-passage time; a ruin probability is the probability of the event
itself, over a stated horizon. The two are related but neither implies the
other, and only the second one can be checked against realized outcomes.

The design rule inherited from `quant.py` holds: if a number is reported, the
thing that makes it checkable is reported next to it.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import Field

from app.schemas.base import BaseSchema, Money, UnitInterval


# ==========================================================================
# Statement ingestion
# ==========================================================================

class ColumnRole(str, Enum):
    """What a statement column is believed to mean.

    `SIGNED_AMOUNT` and the `DEBIT`/`CREDIT` pair are alternatives, not
    companions: a statement expresses direction either by the sign of one
    column or by which of two columns is populated. A mapping that claims both
    has misread one of them.
    """

    DATE = "date"
    VALUE_DATE = "value_date"
    DESCRIPTION = "description"
    REFERENCE = "reference"
    DEBIT = "debit"
    CREDIT = "credit"
    SIGNED_AMOUNT = "signed_amount"
    BALANCE = "balance"
    IGNORED = "ignored"


class AmountConvention(str, Enum):
    """How the statement encodes direction of flow."""

    SEPARATE_DEBIT_CREDIT = "separate_debit_credit"
    SIGNED_SINGLE_COLUMN = "signed_single_column"
    # One amount column plus a separate Dr/Cr marker column. Common in Indian
    # bank exports and the case a signed-column parser silently gets backwards:
    # every amount is positive, so nothing looks wrong until the balance fails
    # to reconcile.
    AMOUNT_WITH_INDICATOR = "amount_with_indicator"


class ColumnAssignment(BaseSchema):
    """One column's inferred role, with the evidence for the inference."""

    source_name: str = Field(description="The header text as it appeared in the file.")
    column_index: int
    role: ColumnRole
    confidence: UnitInterval = Field(
        description="Score of the winning role, normalized against the runner-up. "
        "Low confidence with a correct reconciliation is fine; low confidence "
        "with a failed reconciliation is where the mapping is the suspect."
    )
    evidence: str = Field(
        description="Why this role won — header match, value shape, or both."
    )


class ReconciliationReport(BaseSchema):
    """Whether the parse is internally consistent.

    THE INVARIANT
    -------------
    A statement asserts its own arithmetic:

        balance[i] == balance[i-1] + credit[i] - debit[i]

    This is the statement analogue of the receipt's `total == subtotal + tax`
    check, and it is stronger: it holds on every row rather than once per
    document, so a single misparsed row is localizable rather than merely
    detectable. A parse that satisfies it on every row is very unlikely to have
    the debit and credit columns swapped, the sign convention inverted, or a
    thousands separator eaten — all three break the identity immediately.

    A statement with no balance column cannot be checked this way, and that is
    reported as `checkable = False` rather than as a pass. An unchecked parse
    and a passing parse are different things.
    """

    checkable: bool = Field(
        description="False when the statement carried no balance column."
    )
    n_rows: int
    n_rows_reconciled: int
    max_absolute_residual: float = Field(
        description="Largest |balance[i] - (balance[i-1] + credit - debit)| over the file."
    )
    mean_absolute_residual: float
    tolerance: float = Field(description="Absolute currency tolerance applied per row.")
    passed: bool
    failing_row_indices: list[int] = Field(
        default_factory=list,
        description="Row indices that broke the identity, capped for readability. "
        "Localizing the failure is the point — 'the file does not reconcile' is "
        "not actionable, 'row 47 does not' is.",
    )
    diagnosis: Optional[str] = Field(
        default=None,
        description="When the parse fails, the most likely cause given the residual "
        "pattern — e.g. a residual of exactly 2x the amount on every row is a "
        "sign inversion, not noise.",
    )


class ParsedStatementRow(BaseSchema):
    """One transaction line, after mapping and before it becomes a domain record."""

    row_index: int
    posted_date: date
    description: str
    reference: Optional[str] = None
    debit: Money = Field(description="Money out. Zero when the row is a credit.")
    credit: Money = Field(description="Money in. Zero when the row is a debit.")
    balance: Optional[Money] = None
    reconciled: bool = True


class StatementParseResult(BaseSchema):
    """Everything the parser concluded, including how confident it was.

    `rejected` is a first-class outcome. The parser refuses rather than emitting
    a partially-trusted ledger, for the same reason `normalize()` refuses an
    unreadable receipt total: a wrong statement row is indistinguishable from a
    right one once it reaches the optimizer.
    """

    dialect_name: Optional[str] = Field(
        default=None, description="Matched known layout, if any; None when inferred cold."
    )
    convention: AmountConvention
    date_format: str
    column_assignments: list[ColumnAssignment]
    rows: list[ParsedStatementRow]
    reconciliation: ReconciliationReport
    opening_balance: Optional[Money] = None
    closing_balance: Optional[Money] = None
    total_credits: Money
    total_debits: Money
    rejected: bool = False
    rejection_reason: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


# ==========================================================================
# API ingestion pipeline
# ==========================================================================

class ProviderKind(str, Enum):
    """What is actually on the other end of the connector.

    `LOCAL_REFERENCE` is named rather than hidden. A connector talking to a
    reference server that this repository also ships is a real exercise of the
    HTTP path and an honest one, but it is not evidence that a bank integration
    works, and a field that says so keeps the two from being confused.
    """

    LOCAL_REFERENCE = "local_reference"
    HTTP_OPEN_BANKING = "http_open_banking"


class RateCapStatus(BaseSchema):
    """Where the connector stands against its call budget.

    The architecture plan commits to ~30 calls/user/month. A commitment with no
    counter behind it is a plan, not a limit, so the counter is part of the
    response rather than an internal detail.
    """

    window_days: int
    calls_used: int
    calls_allowed: int
    calls_remaining: int
    window_resets_on: date
    exhausted: bool


class SyncPage(BaseSchema):
    """One page of a paginated pull, retained so the sync is auditable."""

    page_number: int
    cursor: Optional[str] = None
    n_rows: int
    http_status: int
    elapsed_ms: float
    retries: int = 0


class StatementSyncResult(BaseSchema):
    """The outcome of one `sync()` against a provider.

    `idempotency_key` is echoed because the whole point of sending one is that
    a retried sync does not double-post the ledger; a caller that cannot see
    which key was used cannot verify that property held.
    """

    provider: str
    provider_kind: ProviderKind
    account_reference: str
    idempotency_key: str
    replayed_from_cache: bool = Field(
        description="True when this key was already served — the request was a retry "
        "and the original result was returned rather than the pull being repeated."
    )
    pages: list[SyncPage]
    rate_cap: RateCapStatus
    parse: Optional[StatementParseResult] = None
    n_inflows: int = 0
    n_outflows: int = 0
    warnings: list[str] = Field(default_factory=list)


# ==========================================================================
# Business-level insolvency risk
# ==========================================================================

class RuinCurvePoint(BaseSchema):
    """P(cash has gone to zero at or before day t), with its Monte Carlo error.

    FIRST PASSAGE, NOT TERMINAL
    ---------------------------
    This is P(min over [0, t] of balance <= 0), not P(balance at t <= 0). The
    distinction is the whole content of the metric: a business that runs out of
    cash in week three and is rescued by a receivable in week six has failed,
    and a terminal-value check would score it as fine. Ruin is absorbing in the
    sense that matters — payroll missed in week three is missed regardless of
    what arrives later.
    """

    horizon_days: int
    ruin_probability: UnitInterval
    standard_error: float = Field(
        description="Binomial SE sqrt(p(1-p)/N) across the Monte Carlo draws."
    )
    ci_lower: UnitInterval
    ci_upper: UnitInterval


class HazardPoint(BaseSchema):
    """Conditional failure rate in one interval, given survival to its start.

    Reported alongside the cumulative curve because the two answer different
    questions. A cumulative curve that rises steadily can hide a hazard that
    spikes at a payroll date and is near zero elsewhere — which is the shape
    that tells an owner *when* to act rather than merely how worried to be.
    """

    interval_start_day: int
    interval_end_day: int
    n_at_risk: int
    n_failed: int
    hazard: UnitInterval


class ZScoreZone(str, Enum):
    """Altman's three zones for the Z''-score."""

    SAFE = "safe"
    GREY = "grey"
    DISTRESS = "distress"


class InputProvenance(str, Enum):
    """Where a balance-sheet input came from.

    This enum is the reason the Z-score in this system is defensible. Altman's
    ratios need balance-sheet items a cash-flow ledger does not contain. Some
    are genuinely derivable from ingested data, some must be supplied by the
    business, and some can only be approximated. Tagging each one means the
    score can never quietly rest on a number nobody actually knows — the
    approximated ones are visible in the response, and the model refuses to
    report a score when a required input is missing entirely.
    """

    DERIVED_FROM_LEDGER = "derived_from_ledger"
    SUPPLIED_BY_BUSINESS = "supplied_by_business"
    APPROXIMATED = "approximated"


class BalanceSheetInput(BaseSchema):
    """One balance-sheet line, carrying where it came from."""

    name: str
    value: float
    provenance: InputProvenance
    note: Optional[str] = None


class AltmanZScore(BaseSchema):
    """Altman Z''-score (the four-variable, non-manufacturer revision).

    WHY Z'' AND NOT THE ORIGINAL Z
    ------------------------------
    The 1968 Z-score's X4 is *market* value of equity over total liabilities.
    A small business has no market capitalization, so the original model is
    simply not computable here — substituting book equity into it and still
    calling it Z would be a misattribution, not a simplification. Altman's
    1995 Z'' revision was built for exactly this case: it drops the
    sales/total-assets term (which is what makes the original industry-
    sensitive) and uses book equity in X4.

        Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

        X1 = working capital / total assets
        X2 = retained earnings / total assets
        X3 = EBIT / total assets
        X4 = book value of equity / total liabilities

    Zones: Z'' > 2.60 safe, 1.10 - 2.60 grey, < 1.10 distress.

    WHAT THIS SCORE IS AND IS NOT
    -----------------------------
    The coefficients are Altman's, fitted on his sample, not refitted here —
    there is no default-labelled panel of small-business balance sheets in this
    project to refit them on. So this is a *published external model applied to
    supplied inputs*, and it is reported as such. It is not validated by this
    repository's backtest and it must not be presented as though it were. Its
    value is that it is a genuinely independent view: it reads the balance
    sheet, where the Monte Carlo ruin probability reads the cash-flow path, and
    two methods disagreeing is information.
    """

    z_score: float
    zone: ZScoreZone
    x1_working_capital_to_assets: float
    x2_retained_earnings_to_assets: float
    x3_ebit_to_assets: float
    x4_equity_to_liabilities: float
    inputs: list[BalanceSheetInput]
    coefficients_source: str = Field(
        default="Altman (1995) Z''-score for non-manufacturers and emerging markets; "
        "coefficients are the published ones, NOT refitted on this project's data.",
    )
    approximated_input_names: list[str] = Field(
        default_factory=list,
        description="Inputs that were approximated rather than derived or supplied. "
        "A score resting on several of these is weak evidence and says so.",
    )


class RuinCalibration(BaseSchema):
    """Does the predicted ruin probability match the realized ruin frequency?

    This is the check that separates a ruin probability from a plausible
    number. The simulator is run over many independently generated businesses
    whose futures are then actually played out, and the predicted probability
    is compared against whether ruin in fact occurred. Reported with a Brier
    score and ROC-AUC because they measure different failures: Brier catches a
    model whose probabilities are systematically off, AUC catches one that
    cannot rank a risky business above a safe one. A model can pass either
    alone while being useless.
    """

    n_businesses: int
    horizon_days: int
    realized_ruin_rate: UnitInterval
    mean_predicted_probability: UnitInterval
    brier_score: float = Field(description="Mean squared error of the probability. Lower is better.")
    brier_skill_score: float = Field(
        description="1 - Brier/Brier_baseline, where the baseline always predicts the "
        "base rate. Positive means the model beats knowing only the base rate; "
        "the raw Brier score alone cannot tell you that."
    )
    roc_auc: float
    buckets: list["RuinCalibrationBucket"]
    seed: int


class RuinCalibrationBucket(BaseSchema):
    """One bin of the calibration curve."""

    lower: float
    upper: float
    n: int
    mean_predicted: float
    observed_frequency: float


class BankruptcyRisk(BaseSchema):
    """The composite insolvency view: cash-flow ruin plus balance-sheet distress.

    DELIBERATELY NOT A SINGLE NUMBER
    --------------------------------
    The two components are reported side by side and are *not* averaged into
    one score. Averaging them would be indefensible: the ruin probability is
    calibrated against realized outcomes in this project's own synthetic world,
    while the Z''-score carries external coefficients fitted on a different
    population. Blending a validated estimate with an unvalidated one produces
    a number with no validity claim at all, and hides which half moved it.

    `agreement` states whether they point the same way, and disagreement is
    surfaced as a finding rather than resolved by arithmetic.
    """

    as_of: date
    horizon_days: int
    ruin_curve: list[RuinCurvePoint]
    hazard: list[HazardPoint]
    headline_ruin_probability: UnitInterval = Field(
        description="P(ruin within horizon_days), the last point of the curve."
    )
    expected_days_to_ruin_given_ruin: Optional[float] = Field(
        default=None,
        description="E[time to ruin | ruin occurs]. None when no path ruined. "
        "Conditional on ruin, because the unconditional mean is dominated by "
        "the censored survivors and is not a time-to-failure.",
    )
    altman: Optional[AltmanZScore] = Field(
        default=None,
        description="None when balance-sheet inputs were not supplied. Absent rather "
        "than defaulted — a Z-score computed from assumed inputs is fiction.",
    )
    agreement: Optional[str] = Field(
        default=None,
        description="Whether the cash-flow and balance-sheet views concur, and what "
        "it means when they do not.",
    )
    calibration: Optional[RuinCalibration] = None
    n_iterations: int
    random_seed: int
    caveats: list[str] = Field(default_factory=list)


RuinCalibration.model_rebuild()
