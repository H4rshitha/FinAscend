"""Quant Core output schemas — the §2.1 amendments.

These types exist so that fitted artifacts travel with their evidence. A
distribution choice carries the KS statistic that justified it; a Monte Carlo
estimate carries its standard error; a model score carries the per-feature
contributions and the measured lift over the rules baseline.

The design rule throughout: if a number is reported, the thing that makes it
checkable is reported next to it.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import Field

from app.schemas.base import BaseSchema, Money, UnitInterval


# --------------------------------------------------------------------------
# A.2 — fitted uncertainty model
# --------------------------------------------------------------------------

class DistributionFamily(str, Enum):
    """Candidate delay distributions, all supported on [0, inf)."""

    GAMMA = "gamma"
    WEIBULL = "weibull"
    LOGNORMAL = "lognormal"


class CopulaFamily(str, Enum):
    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"


class GoodnessOfFit(BaseSchema):
    """One candidate's fit quality, retained even when it loses.

    Runners-up are kept so the selection is auditable: "we chose Weibull"
    means little without "and Gamma scored this, log-normal scored this."
    """

    family: DistributionFamily
    params: dict[str, float]
    ks_statistic: float = Field(description="Kolmogorov-Smirnov D statistic; smaller is better")
    ks_pvalue: float = Field(description="P(D this large | data came from this fitted distribution)")
    log_likelihood: float
    aic: float
    selected: bool = False


class CounterpartyDelayFit(BaseSchema):
    """The fitted payment-delay distribution for a single counterparty."""

    counterparty_id: str
    n_observations: int
    candidates: list[GoodnessOfFit] = Field(
        description="Every family tried, with its fit statistics. Losers included."
    )
    selected_family: DistributionFamily
    selected_params: dict[str, float]
    selection_rationale: str
    # P(paid on or before the expected date) under the fitted distribution.
    # This is what feeds Inflow.certainty, replacing the struck 1.0 default.
    prob_on_time: UnitInterval


class CopulaSpec(BaseSchema):
    """Dependence structure across counterparties.

    Independence is the wrong null here — see the docstring in
    monte_carlo_engine.sample_correlated_delays for why it understates the
    tail that actually matters.
    """

    family: CopulaFamily
    df: Optional[float] = Field(
        default=None,
        description="Student-t degrees of freedom; None for Gaussian. Lower df = fatter joint tails.",
    )
    correlation_matrix: list[list[float]]
    counterparty_order: list[str] = Field(
        description="Row/column order of correlation_matrix — without this the matrix is unusable."
    )
    correlation_source: str = Field(
        description="How the matrix was obtained (e.g. 'Spearman rank correlation of historical delays')."
    )


class UncertaintyModelSpec(BaseSchema):
    """Replaces the struck `receivable_uncertainty_model: str` free-text field."""

    fits: list[CounterpartyDelayFit]
    copula: CopulaSpec
    fitted_at: datetime


# --------------------------------------------------------------------------
# A.1 — forecasting
# --------------------------------------------------------------------------

class ForecastModelName(str, Enum):
    SEASONAL_NAIVE = "seasonal_naive"
    HOLT_WINTERS = "holt_winters"
    SARIMAX = "sarimax"


class ForecastPoint(BaseSchema):
    """One horizon step: the point estimate and its interval.

    `lower`/`upper` are not decoration. A.2 consumes the interval width, so a
    model that reports false precision here propagates that error into RaR.
    """

    as_of_date: date
    point: float
    lower: float
    upper: float


class ModelSelectionScore(BaseSchema):
    """Walk-forward score for one candidate model."""

    model_name: ForecastModelName
    mape: float = Field(description="Mean absolute percentage error, out-of-sample")
    rmse: float
    aic: Optional[float] = Field(default=None, description="None for seasonal naive — it fits no likelihood")
    bic: Optional[float] = None
    n_folds: int
    fold_rmses: list[float] = Field(description="Per-fold RMSE; the spread matters as much as the mean")


class ForecastResult(BaseSchema):
    """Output of A.1, consumed by A.2 and by GET /risk/forecast."""

    business_id: str
    generated_at: datetime
    horizon_days: int
    interval_confidence: float = Field(description="e.g. 0.95 for a 95% prediction interval")
    selected_model: ForecastModelName
    selection_rationale: str
    scores: list[ModelSelectionScore] = Field(
        description="All candidates including rejected ones — the rejections are the interesting part."
    )
    path: list[ForecastPoint]
    days_to_zero: Optional[int] = Field(
        default=None,
        description="Point estimate only. The honest version of this number is the RaR in A.2.",
    )
    # The conformal layer travels with the forecast because it is the evidence
    # that the interval is honest. q_hat near 1.0 means the model's own
    # uncertainty estimate needed no correction; a large q_hat means it did,
    # and reporting it is what distinguishes recalibration from blind widening.
    conformal_q_hat: Optional[float] = Field(
        default=None,
        description="Multiplier applied to the model's own scale: interval = point +/- q_hat * scale(h). Read against z_reference, NOT against 1.0 — it multiplies a sigma.",
    )
    conformal_z_reference: Optional[float] = Field(
        default=None,
        description="The Gaussian reference z_{(1+c)/2} a correctly specified model would need (1.96 at 95%).",
    )
    conformal_scale_ratio: Optional[float] = Field(
        default=None,
        description="q_hat / z. 1.0 = the model's own scale was right; 2.0 = it was half what it should have been.",
    )
    conformal_n_scores: Optional[int] = Field(
        default=None,
        description="Held-out nonconformity scores behind q_hat. The level is only expressible if ceil((N+1)*c) <= N.",
    )
    conformal_achieved: Optional[bool] = Field(
        default=None,
        description="False when N was too small to express the requested level; the interval is then an approximation, not a guarantee.",
    )
    scale_gamma: Optional[float] = Field(
        default=None,
        description="Fitted exponent of scale(h) = a*h**gamma. None when the model supplies an analytic variance.",
    )


# --------------------------------------------------------------------------
# A.2 — Runway at Risk (the headline artifact)
# --------------------------------------------------------------------------

class RunwayAtRisk(BaseSchema):
    """RaR / CRaR, the liquidity analogue of VaR / CVaR.

    `runway_at_risk_days = 11` at `confidence_level = 0.95` reads: there is a
    5% chance of hitting zero cash within 11 days.
    """

    confidence_level: float = Field(description="e.g. 0.95")
    runway_at_risk_days: int = Field(
        description="RaR: the alpha-quantile of the days-to-zero distribution."
    )
    conditional_runway_at_risk_days: float = Field(
        description="CRaR: mean days-to-zero conditional on being in the bad tail. "
        "RaR alone says nothing about how bad the bad case is."
    )
    probability_of_shortfall: UnitInterval
    n_iterations: int
    mc_standard_error: float = Field(
        description="Standard error of the RaR estimate. This is what justifies n_iterations."
    )
    random_seed: int


class ConvergencePoint(BaseSchema):
    """One point on the standard-error-vs-N curve (notebook 02)."""

    n_iterations: int
    estimate: float
    standard_error: float


# --------------------------------------------------------------------------
# A.3 — optimization
# --------------------------------------------------------------------------

class SolverName(str, Enum):
    PULP_CBC = "pulp_cbc"
    SCIPY_MILP = "scipy_milp"
    DP_KNAPSACK = "dp_knapsack"
    SAA_CHANCE_CONSTRAINED = "saa_chance_constrained"
    RULE_BASED_BASELINE = "rule_based_baseline"


class AllocationItem(BaseSchema):
    obligation_id: str
    allocated_amount: Money
    fully_funded: bool


class SolverSolution(BaseSchema):
    """One solver's answer to the allocation problem."""

    solver_name: SolverName
    status: str
    objective_value: float = Field(description="Total weighted penalty incurred. Lower is better.")
    allocations: list[AllocationItem]
    solve_seconds: float


class SolverAgreement(BaseSchema):
    """Cross-check between two independent implementations (A.3).

    Two independent solvers agreeing is evidence of correctness. One solver
    returning a plausible number is not.
    """

    lp_objective_value: float
    dp_objective_value: float
    absolute_delta: float
    tolerance: float
    agree: bool
    explanation: str = Field(
        description="Required when agree=False. Divergence is a finding to explain, not to hide."
    )


class ChanceConstrainedResult(BaseSchema):
    """SAA solution respecting P(shortfall) <= epsilon."""

    epsilon: float = Field(description="Maximum tolerated probability of shortfall")
    saa_num_scenarios: int = Field(
        description="Subsample size, NOT the full Monte Carlo draw count — see A.3 scenario cap."
    )
    achieved_shortfall_probability: float = Field(
        description="Realized P(shortfall) on the scenario batch; should be <= epsilon."
    )
    solution: SolverSolution
    stability_across_resamples: Optional[float] = Field(
        default=None,
        description="Std dev of objective across independent scenario resamples. "
        "This is the evidence that the scenario cap is large enough.",
    )


# --------------------------------------------------------------------------
# A.4 — credit risk
# --------------------------------------------------------------------------

class RiskModelName(str, Enum):
    RULE_BASELINE = "rule_baseline"
    LOGISTIC_L2 = "logistic_l2"
    GBM = "gbm"


class FeatureContribution(BaseSchema):
    """Why this score is what it is, in terms a non-technical user can read."""

    feature: str
    value: float
    contribution: float = Field(
        description="SHAP value (GBM) or coefficient x standardized value (logistic)."
    )
    ci_lower: Optional[float] = Field(default=None, description="Logistic only: 95% CI on the coefficient")
    ci_upper: Optional[float] = None
    direction: str = Field(description="'increases_risk' or 'decreases_risk'")


class ModelPerformance(BaseSchema):
    """Accuracy is deliberately absent — on imbalanced default data it is
    uninformative to the point of being misleading."""

    model_name: RiskModelName
    roc_auc: float
    brier_score: float = Field(description="Lower is better; measures calibration, not just ranking")
    log_loss: float
    n_train: int
    n_test: int


class CalibrationBucket(BaseSchema):
    """One point on the calibration curve: predicted vs. observed."""

    bucket_lower: float
    bucket_upper: float
    n: int
    mean_predicted: float
    observed_rate: float


class BaselineLift(BaseSchema):
    """Did the complexity earn its place? Reported as-is either way."""

    baseline_model: RiskModelName = RiskModelName.RULE_BASELINE
    baseline_roc_auc: float
    model_roc_auc: float
    auc_lift: float = Field(description="model - baseline. Negative is reported, not suppressed.")
    verdict: str


# --------------------------------------------------------------------------
# A.5 — anomaly detection
# --------------------------------------------------------------------------

class AnomalyFlag(BaseSchema):
    record_id: str
    robust_z: float = Field(description="MAD-based, not mean/std — see anomaly_detection docstring")
    flagged_by_robust_z: bool
    flagged_by_isolation_forest: bool
    isolation_forest_score: float
    dbscan_label: Optional[int] = Field(default=None, description="-1 means DBSCAN noise")


class DetectorComparison(BaseSchema):
    """Where the detectors agree and disagree — itself a small piece of research."""

    n_records: int
    n_robust_z_only: int
    n_isolation_forest_only: int
    n_both: int
    jaccard_agreement: float
    commentary: str


# --------------------------------------------------------------------------
# Section C — backtesting
# --------------------------------------------------------------------------

class RegretMetrics(BaseSchema):
    """Efficiency loss vs. a plan built with full knowledge of what happened."""

    mean_regret: float
    median_regret: float
    p95_regret: float
    total_realized_penalty: float
    total_hindsight_penalty: float
    relative_regret: float = Field(
        description="(realized - hindsight) / hindsight. 0.0 means we matched perfect foresight."
    )


class CalibrationResult(BaseSchema):
    """Are the prediction intervals honest?

    A model with tight, wrong intervals is more dangerous than one with wide,
    honest ones, and only this check distinguishes them.
    """

    nominal_coverage: float = Field(description="e.g. 0.95")
    empirical_coverage: float = Field(description="Fraction of actuals that fell inside the interval")
    n_observations: int
    mean_interval_width: float
    verdict: str


class BacktestReport(BaseSchema):
    business_id: str
    generated_at: datetime
    n_replay_days: int
    replay_start: date
    replay_end: date
    regret: RegretMetrics
    calibration: list[CalibrationResult]
    plan_generator: SolverName
    notes: str
