"""The six core domain models from Section 2, with §2.1 amendments applied.

Amendments visible here:
  - Inflow.certainty     : no 1.0 default; it is an output of the A.2 delay fit
  - Obligation.urgency / penalty_severity : computed, see decision_engine
  - RiskScore            : carries model identity, contributions, baseline lift
  - SimulationResult     : carries RaR/CRaR, a required seed, and the fitted
                           uncertainty model instead of a free-text label
  - DecisionPlan         : carries both solvers' objectives and their agreement
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional
from uuid import UUID

from pydantic import Field, model_validator

from app.schemas.base import (
    BaseSchema,
    Currency,
    DecisionActionType,
    FlexibilityLevel,
    Money,
    ObligationDirection,
    ObligationStatus,
    ReviewStatus,
    ScenarioType,
    SourceType,
    UnitInterval,
)
from app.schemas.quant import (
    BaselineLift,
    ChanceConstrainedResult,
    FeatureContribution,
    RiskModelName,
    RunwayAtRisk,
    SolverAgreement,
    SolverName,
    UncertaintyModelSpec,
)


class Inflow(BaseSchema):
    id: UUID
    business_id: UUID
    amount: Money
    currency: Currency = Currency.INR
    expected_date: date
    received_date: Optional[date] = None
    counterparty_id: Optional[UUID] = None
    counterparty_name: str
    source_type: SourceType
    source_reference: Optional[str] = None
    is_receivable: bool = True
    # §2.1: the 1.0 default was struck. A default of 1.0 asserts perfect
    # certainty about a receivable — the strongest and least defensible claim
    # in the whole schema. This is now P(paid by expected_date) from the
    # counterparty's fitted delay distribution (A.2).
    certainty: UnitInterval
    is_duplicate_of: Optional[UUID] = None
    created_at: datetime


class Outflow(BaseSchema):
    id: UUID
    business_id: UUID
    amount: Money
    currency: Currency = Currency.INR
    due_date: date
    paid_date: Optional[date] = None
    counterparty_id: Optional[UUID] = None
    counterparty_name: str
    category: str
    source_type: SourceType
    source_reference: Optional[str] = None
    is_recurring: bool = False
    is_duplicate_of: Optional[UUID] = None
    created_at: datetime


class Obligation(BaseSchema):
    id: UUID
    business_id: UUID
    direction: ObligationDirection
    linked_inflow_id: Optional[UUID] = None
    linked_outflow_id: Optional[UUID] = None
    counterparty_id: UUID
    amount: Money
    due_date: date
    status: ObligationStatus = ObligationStatus.PENDING
    # §2.1: both computed, never hand-assigned. They feed the prioritizer AND
    # the LP objective, so typed-in values would silently determine the
    # optimizer's answer. See decision_engine.obligation_features.
    urgency: UnitInterval
    penalty_severity: UnitInterval
    flexibility: FlexibilityLevel = FlexibilityLevel.MODERATE
    late_fee_rate_per_day: Optional[float] = None
    priority_rank: Optional[int] = None
    priority_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def _direction_matches_link(self) -> "Obligation":
        """A payable cannot link an Inflow; a receivable cannot link an Outflow."""
        if self.direction == ObligationDirection.PAYABLE.value and self.linked_inflow_id is not None:
            raise ValueError("a payable obligation cannot link an Inflow")
        if self.direction == ObligationDirection.RECEIVABLE.value and self.linked_outflow_id is not None:
            raise ValueError("a receivable obligation cannot link an Outflow")
        return self


class RiskScore(BaseSchema):
    id: UUID
    business_id: UUID
    obligation_id: Optional[UUID] = None
    counterparty_id: Optional[UUID] = None
    urgency_score: UnitInterval
    impact_score: UnitInterval
    default_probability: UnitInterval
    composite_score: UnitInterval
    # §2.1: identifies a fitted artifact, which a bare "v1" string cannot.
    model_name: RiskModelName
    weighting_scheme_version: str = Field(
        description="model id + training-set hash + fit timestamp"
    )
    feature_contributions: list[FeatureContribution] = Field(default_factory=list)
    baseline_comparison: Optional[BaselineLift] = None
    # Generated from feature_contributions, not written by a rule template.
    rationale: str
    computed_at: datetime


class CashflowPercentileBand(BaseSchema):
    as_of_date: date
    p10: Money
    p50: Money
    p90: Money


class SimulationResult(BaseSchema):
    id: UUID
    business_id: UUID
    scenario_type: ScenarioType = ScenarioType.BASELINE
    # §2.1: no 10_000 default. The value is passed in and justified by the
    # convergence study in notebook 02.
    num_iterations: int
    # §2.1: required, not Optional. Reproducibility is not opt-in.
    random_seed: int
    horizon_days: int = 90
    projected_days_to_zero: Optional[int] = None
    probability_of_shortfall: UnitInterval
    percentile_bands: list[CashflowPercentileBand]
    # §2.1: the headline metric, promoted out of the footnotes.
    runway_at_risk: RunwayAtRisk
    # §2.1: replaces receivable_uncertainty_model: str = "beta_delay_distribution"
    uncertainty_model: UncertaintyModelSpec
    generated_at: datetime


class DecisionActionItem(BaseSchema):
    obligation_id: UUID
    action_type: DecisionActionType
    allocated_amount: Money
    proposed_date: Optional[str] = None
    justification: str


class DecisionPlan(BaseSchema):
    id: UUID
    business_id: UUID
    scenario_type: ScenarioType = ScenarioType.BASELINE
    simulation_result_id: Optional[UUID] = None
    available_cash: Money
    total_obligations_amount: Money
    # §2.1: no default. Records which of the solvers produced THIS plan.
    solver_name: SolverName
    solver_status: str
    objective_value: Optional[float] = None
    # §2.1: both solvers plus the chance-constrained variant surface here.
    lp_objective_value: Optional[float] = None
    dp_objective_value: Optional[float] = None
    solver_agreement: Optional[SolverAgreement] = None
    chance_constrained: Optional[ChanceConstrainedResult] = None
    actions: list[DecisionActionItem]
    baseline_plan_id: Optional[UUID] = None
    devils_advocate_challenge_id: Optional[UUID] = None
    review_status: ReviewStatus = ReviewStatus.PENDING_REVIEW
    reviewed_by: Optional[UUID] = None
    audit_log_entry_id: Optional[UUID] = None
    created_at: datetime
