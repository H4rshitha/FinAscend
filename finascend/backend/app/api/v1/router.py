"""API v1 — routes backed by the real Quant Core.

Tier-4 endpoints (risk, simulation, decisions, audit) return genuinely
computed values. Tier-5 endpoints (graph, voice, chat) return an honest 501:
they route, authenticate and validate correctly, but they never fabricate an
answer or return canned text that implies a working model.

A demo dataset is fitted once at startup and cached, because refitting SARIMAX
and running 10,000 Monte Carlo iterations per request would make the API
unusable. The cache key includes the seed so results stay reproducible.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.security import TokenPayload, create_access_token, current_user, require_role
from app.schemas.base import UserRole
from app.schemas.quant import RiskModelName, SolverName
from app.services.audit.hash_chain_logger import HashChainLog
from app.services.decision_engine.rule_based_prioritizer import (
    measure_optimizer_lift,
    prioritize,
    solve_rule_based,
)
from app.services.quant_core.forecasting import select_and_forecast
from app.services.quant_core.monte_carlo_engine import run_simulation
from app.services.quant_core.optimization.chance_constrained import (
    solve_chance_constrained,
)
from app.services.quant_core.optimization.cross_validation import cross_validate
from app.services.quant_core.optimization.lp_solver import solve_lp
from app.services.quant_core.pipeline import build_as_of_view
from app.services.quant_core.risk_scoring import (
    build_features,
    compare_to_baseline,
    explain_prediction,
    rationale_from_contributions,
    train_models,
)
from app.services.quant_core.synthetic_data import Regime, generate_dataset
from app.services.backtesting.replay_harness import obligations_as_of

from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.ingestion import router as ingestion_router

router = APIRouter()
router.include_router(auth_router)
router.include_router(ingestion_router)
AUDIT = HashChainLog()

DEMO_SEED = 42
DEMO_DAYS = 1095
DEMO_COUNTERPARTIES = 10


def _not_implemented(feature: str, reason: str) -> HTTPException:
    """The tier-5 contract: decline honestly rather than fake a response."""
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "error_code": "not_implemented",
            "message": f"{feature} is not yet implemented.",
            "details": {
                "reason": reason,
                "status": "endpoint routes, authenticates and validates; no model is behind it yet",
            },
        },
    )


@lru_cache(maxsize=4)
def _world(seed: int = DEMO_SEED, regime: str = "adversarial"):
    return generate_dataset(
        seed=seed,
        regime=Regime(regime),
        n_days=DEMO_DAYS,
        n_counterparties=DEMO_COUNTERPARTIES,
        opening_balance_months=1.2,
    )


# Demo vantage point: 72% through the history. NOT the last day — by then the
# stressed demo business is essentially insolvent, which makes every metric
# degenerate (RaR pins at 1 day, and all four solvers tie at zero lift because
# nothing is affordable). 72% lands after the structural break, where the
# business is under real pressure but the allocation decision still has
# consequences. The point is to demo the engine where it has something to say.
# Measured across the demo world: cash/30-day-obligation ratio runs 1.36 at
# 72% of the history, 0.74 at 85%, 0.17 at 90%, and turns negative past 93%.
# 0.85 is the point where cash and near-term obligations are comparable, which
# is precisely where the allocation decision has consequences and where RaR is
# neither pinned at the horizon nor collapsed to one day.
DEMO_AS_OF_FRACTION = 0.85


@lru_cache(maxsize=4)
def _view(seed: int = DEMO_SEED, regime: str = "adversarial"):
    ds = _world(seed, regime)
    idx = int(len(ds.daily) * DEMO_AS_OF_FRACTION)
    return build_as_of_view(ds, ds.daily["date"].iloc[idx].date())


@lru_cache(maxsize=4)
def _forecast(seed: int = DEMO_SEED, regime: str = "adversarial", horizon: int = 90):
    v = _view(seed, regime)
    return select_and_forecast(
        v.daily,
        horizon_days=horizon,
        value_column="net_ex_receipts",
        opening_balance=v.opening_balance,
    )


@lru_cache(maxsize=2)
def _risk_models(seed: int = DEMO_SEED, regime: str = "adversarial"):
    ds = _world(seed, regime)
    data = build_features(ds.payments)
    return data, train_models(data, seed=seed)


@lru_cache(maxsize=2)
def _simulation(seed: int = DEMO_SEED, regime: str = "adversarial", n: int = 10_000):
    v = _view(seed, regime)
    fc = _forecast(seed, regime)
    return run_simulation(
        opening_balance=v.opening_balance,
        forecast=fc,
        receivables=v.receivables,
        delays_by_cp=v.delays_by_cp,
        panel=v.delay_panel,
        n_iterations=n,
        seed=seed,
        horizon_days=90,
    )


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------

# The former `POST /auth/token?role=owner` endpoint has been REMOVED.
#
# It minted a valid owner token for anyone who asked, with no credentials. That
# was defensible while the API had no user accounts and nothing to protect, but
# once real sign-in exists it is simply an unauthenticated bypass of it — and
# this repository is public, so the route would be the first thing anyone
# tried. Authentication now lives in `endpoints/auth.py`:
#
#   POST /auth/signup   create an organisation and its first owner
#   POST /auth/login    exchange credentials for a token
#   GET  /auth/me       re-resolve the session from the database
#   GET  /auth/options  company sizes and the plan each maps to


# ---------------------------------------------------------------------------
# financial state
# ---------------------------------------------------------------------------

@router.get("/financial-state/summary", tags=["financial-state"])
def financial_summary(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    v = _view()
    fc = _forecast()
    return {
        "as_of": v.as_of.isoformat(),
        "cash_balance": round(v.opening_balance, 2),
        "outstanding_receivables": len(v.receivables),
        "outstanding_receivable_value": round(float(v.receivables["amount"].sum()), 2),
        "days_to_zero_point_estimate": fc.days_to_zero,
        "note": (
            "days_to_zero is a POINT estimate and ignores uncertainty. "
            "Use /simulation/runway-at-risk for the honest version."
        ),
    }


# ---------------------------------------------------------------------------
# forecasting
# ---------------------------------------------------------------------------

@router.get("/risk/forecast", tags=["risk"])
def get_forecast(
    horizon_days: int = Query(90, ge=7, le=180),
    user: TokenPayload = Depends(current_user),
) -> dict[str, Any]:
    """Forecast with prediction intervals, plus every candidate's score."""
    fc = _forecast(DEMO_SEED, "adversarial", horizon_days)
    return {
        "selected_model": fc.selected_model,
        "selection_rationale": fc.selection_rationale,
        "interval_confidence": fc.interval_confidence,
        "days_to_zero": fc.days_to_zero,
        "candidates": [s.model_dump() for s in fc.scores],
        "path": [p.model_dump() for p in fc.path[:horizon_days]],
        # The conformal layer travels with the forecast: it is the evidence
        # that the interval width was measured against held-out error rather
        # than asserted by the model's own likelihood.
        "interval_calibration": {
            "method": "normalized split conformal on |residual| / scale(h)",
            "q_hat": fc.conformal_q_hat,
            "z_reference": fc.conformal_z_reference,
            "scale_ratio": fc.conformal_scale_ratio,
            "n_calibration_scores": fc.conformal_n_scores,
            "level_achievable": fc.conformal_achieved,
            "scale_gamma": fc.scale_gamma,
            "reading": (
                "q_hat multiplies a STANDARD DEVIATION, so its reference is "
                "z = 1.96 at the 95% level, not 1.0 — a model whose stated "
                "scale is exactly right still needs 1.96 of them. Read "
                "scale_ratio = q_hat / z: 1.0 means the model's own scale was "
                "right, 2.0 means it was half what it should have been."
            ),
        },
    }


# ---------------------------------------------------------------------------
# simulation — the headline metric
# ---------------------------------------------------------------------------

@router.get("/simulation/runway-at-risk", tags=["simulation"])
def runway_at_risk(
    confidence: float = Query(0.95, ge=0.5, le=0.999),
    user: TokenPayload = Depends(current_user),
) -> dict[str, Any]:
    """Runway-at-Risk and CRaR — the product's headline metric."""
    from app.services.quant_core.monte_carlo_engine import compute_runway_at_risk

    paths, _rar, spec = _simulation()
    rar = compute_runway_at_risk(paths, confidence=confidence)
    return {
        "runway_at_risk_days": rar.runway_at_risk_days,
        "conditional_runway_at_risk_days": round(rar.conditional_runway_at_risk_days, 2),
        "confidence_level": rar.confidence_level,
        "probability_of_shortfall": round(rar.probability_of_shortfall, 4),
        "n_iterations": rar.n_iterations,
        "mc_standard_error": round(rar.mc_standard_error, 4),
        "random_seed": rar.random_seed,
        "interpretation": (
            f"There is a {(1 - confidence) * 100:.0f}% chance of reaching zero cash "
            f"within {rar.runway_at_risk_days} days. In that worst-case tail, the "
            f"average time to zero is {rar.conditional_runway_at_risk_days:.1f} days."
        ),
    }


@router.get("/simulation/uncertainty-model", tags=["simulation"])
def uncertainty_model(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    """The fitted delay distributions and copula behind the simulation.

    Reports summary statistics of each fitted distribution alongside its
    parameters. Those are computed here, from the frozen distribution, rather
    than left for a caller to derive from `shape`/`scale` — a client that has to
    reimplement "the mean of a Weibull" to show "usually pays in about 9 days"
    is a client that will eventually get it wrong.

    Note on `prob_on_time`: it is P(delay <= 0) and is therefore **structurally
    zero** for every counterparty, because `loc` is pinned at 0 when fitting and
    the candidate families are continuous. It is retained for compatibility but
    carries no information and must not be rendered as a trust signal — use
    `prob_within_7_days` or the delay quantiles, which actually discriminate.
    """
    from app.services.quant_core.monte_carlo_engine import frozen_from_fit

    _paths, _rar, spec = _simulation()

    fits = []
    for f in spec.fits:
        frozen = frozen_from_fit(f)
        fits.append(
            {
                "counterparty_id": f.counterparty_id,
                "n_observations": f.n_observations,
                "selected_family": f.selected_family,
                "selected_params": f.selected_params,
                "prob_on_time": round(f.prob_on_time, 4),
                "prob_on_time_note": (
                    "P(delay <= 0); structurally zero because loc is pinned at 0. "
                    "Not a usable signal — see prob_within_7_days."
                ),
                # The discriminating statistics.
                "mean_delay_days": round(float(frozen.mean()), 2),
                "median_delay_days": round(float(frozen.ppf(0.5)), 2),
                "p90_delay_days": round(float(frozen.ppf(0.9)), 2),
                "prob_within_7_days": round(float(frozen.cdf(7.0)), 4),
                "prob_within_30_days": round(float(frozen.cdf(30.0)), 4),
                "selection_rationale": f.selection_rationale,
                "candidates": [c.model_dump() for c in f.candidates],
            }
        )

    return {
        "fitted_at": spec.fitted_at.isoformat(),
        "copula": spec.copula.model_dump(),
        "fits": fits,
    }


# ---------------------------------------------------------------------------
# bankruptcy / insolvency risk (A.7)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _ruin_calibration(n_businesses: int = 500, n_iterations: int = 1_500):
    """Cached because calibration is a property of the METHOD, not of a business.

    Recomputing it per request would be both slow and misleading about what it
    measures — it would suggest the number describes the business being
    scored, when it describes whether the estimator's probabilities are honest
    across a population.
    """
    from app.services.quant_core.bankruptcy_risk import validate_ruin_calibration

    return validate_ruin_calibration(
        n_businesses=n_businesses, n_iterations=n_iterations, seed=DEMO_SEED,
        parameter_uncertainty=True,
    )


@router.get("/risk/bankruptcy", tags=["risk"])
def bankruptcy_risk(
    include_calibration: bool = Query(
        True, description="Attach the method's measured calibration."
    ),
    total_assets: Optional[float] = Query(None, gt=0),
    total_liabilities: Optional[float] = Query(None, gt=0),
    current_assets: Optional[float] = Query(None, ge=0),
    current_liabilities: Optional[float] = Query(None, ge=0),
    retained_earnings: Optional[float] = Query(None),
    ebit: Optional[float] = Query(None),
    user: TokenPayload = Depends(current_user),
) -> dict[str, Any]:
    """P(the business itself fails), plus an optional balance-sheet view.

    The cash-flow view always computes — it reads the same Monte Carlo paths
    that back `/simulation/runway-at-risk`. The Altman Z''-score computes only
    when the balance-sheet inputs are supplied as query parameters; omit them
    and the response omits the score rather than inventing one.

    This is the business's own failure probability. It is NOT the counterparty
    default probability served by `/risk/models` and `/risk/{id}/explain`.
    """
    from app.services.quant_core.bankruptcy_risk import (
        BalanceSheet,
        assess_bankruptcy_risk,
    )

    paths, _rar, _spec = _simulation()
    v = _view()

    balance_sheet = None
    supplied = (total_assets, total_liabilities, current_assets,
                current_liabilities, retained_earnings, ebit)
    if any(x is not None for x in supplied):
        balance_sheet = BalanceSheet(
            total_assets=total_assets or 0.0,
            total_liabilities=total_liabilities or 0.0,
            current_assets=current_assets or 0.0,
            current_liabilities=current_liabilities or 0.0,
            retained_earnings=retained_earnings,
            ebit=ebit,
        )

    risk = assess_bankruptcy_risk(
        paths.balances,
        as_of=v.as_of,
        random_seed=paths.seed,
        balance_sheet=balance_sheet,
        calibration=_ruin_calibration() if include_calibration else None,
    )
    return risk.model_dump()


@router.get("/risk/bankruptcy/calibration", tags=["risk"])
def bankruptcy_calibration(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    """Are the ruin probabilities honest? Measured against realized outcomes."""
    c = _ruin_calibration()
    return {
        **c.model_dump(),
        "how_to_read": (
            "brier_skill_score > 0 means the model beats predicting the base rate "
            "for everyone; roc_auc > 0.5 means it can rank a failing business above "
            "a surviving one. Both are needed — a model can pass either alone while "
            "being useless."
        ),
    }


# ---------------------------------------------------------------------------
# credit risk + explainability
# ---------------------------------------------------------------------------

@router.get("/risk/models", tags=["risk"])
def risk_model_comparison(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    """Rules vs logistic vs GBM, with the measured lift reported either way."""
    _data, models = _risk_models()
    return {
        "performance": {
            name.value if hasattr(name, "value") else str(name): m.performance.model_dump()
            for name, m in models.items()
        },
        "lift_vs_baseline": {
            cand.value: compare_to_baseline(models, cand).model_dump()
            for cand in (RiskModelName.LOGISTIC_L2, RiskModelName.GBM)
        },
        "note": (
            "Accuracy is deliberately not reported: on imbalanced default data "
            "it is uninformative to the point of being misleading."
        ),
    }


@router.get("/risk/{row_index}/explain", tags=["risk"])
def explain_risk(
    row_index: int,
    model: RiskModelName = Query(RiskModelName.GBM),
    user: TokenPayload = Depends(current_user),
) -> dict[str, Any]:
    """Per-feature attribution — the substantive answer to the trust barrier."""
    data, models = _risk_models()
    if not 0 <= row_index < len(data.X):
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "not_found",
                "message": f"row_index must be in [0, {len(data.X) - 1}]",
                "details": {},
            },
        )
    fitted = models[model]
    row = data.X.iloc[[row_index]]
    prob = float(fitted.predict_proba(row)[0])
    contribs = explain_prediction(fitted, row)
    return {
        "model": model.value,
        "default_probability": round(prob, 4),
        "rationale": rationale_from_contributions(prob, contribs),
        "feature_contributions": [c.model_dump() for c in contribs],
        "calibration": [b.model_dump() for b in fitted.calibration],
        "baseline_comparison": compare_to_baseline(models, model).model_dump()
        if model is not RiskModelName.RULE_BASELINE
        else None,
    }


# ---------------------------------------------------------------------------
# decisions + solver comparison
# ---------------------------------------------------------------------------

@router.get("/decisions/solvers", tags=["decisions"])
def solver_comparison(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    """LP vs DP vs chance-constrained vs the rules baseline, on one instance."""
    ds = _world()
    v = _view()
    rng = np.random.default_rng(DEMO_SEED)
    problem = obligations_as_of(ds, v, rng=rng)
    paths, _rar, _spec = _simulation()

    check = cross_validate(problem)
    cc = solve_chance_constrained(problem, paths.balances, epsilon=0.05, seed=DEMO_SEED)
    baseline = solve_rule_based(problem)

    return {
        "available_cash": round(problem.available_cash, 2),
        "total_obligations": round(problem.total_amount, 2),
        "lp": check.lp.model_dump(),
        "dp": check.dp.model_dump(),
        "solver_agreement": check.agreement.model_dump(),
        "chance_constrained": cc.model_dump(),
        "rules_baseline": baseline.model_dump(),
        "optimizer_lift_vs_baseline": measure_optimizer_lift(problem, check.lp),
    }


@router.get("/decisions/priority", tags=["decisions"])
def priority_ranking(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    """The rules baseline's ranking, kept as the explicit comparison point."""
    ds, v = _world(), _view()
    problem = obligations_as_of(ds, v, rng=np.random.default_rng(DEMO_SEED))
    return {"ranking": [r.__dict__ for r in prioritize(problem)]}


# Plain-language labels for the generated chart-of-accounts categories. The
# obligation_id is built as f"{category}-{as_of}" in `obligations_as_of`, so the
# category is recoverable from it rather than needing a parallel lookup.
_CATEGORY_LABELS = {
    "payroll": "Staff wages",
    "rent": "Rent",
    "loan_emi": "Loan repayment",
    "vendor_payment": "Supplier bill",
    "tax": "Tax payment",
    "utilities": "Electricity and water",
}


def _category_of(obligation_id: str) -> str:
    return obligation_id.rsplit("-", 3)[0] if "-" in obligation_id else obligation_id


def _plain_label(obligation_id: str) -> str:
    cat = _category_of(obligation_id)
    return _CATEGORY_LABELS.get(cat, cat.replace("_", " ").capitalize())


def _due_phrase(days: float) -> str:
    d = int(round(days))
    if d <= 0:
        return "due today"
    if d == 1:
        return "due tomorrow"
    if d <= 7:
        return f"due in {d} days"
    return f"due in about {d // 7} week{'s' if d // 7 > 1 else ''}"


def _justify(item, allocated: float, late_fee_if_unpaid: float) -> tuple[str, str]:
    """Write the plain-language justification for one allocation.

    Returns (action_type, justification). Every number in the sentence comes
    from the solver's own output or the obligation's contract terms — nothing
    is asserted that is not derived. The register is deliberately non-technical:
    this text is what a business owner reads before approving a plan, so it says
    "late fee" rather than "penalty rate" and never mentions the objective
    function.
    """
    due = _due_phrase(item.days_until_due)
    label = _plain_label(item.obligation_id)
    fee = f"₹{late_fee_if_unpaid:,.0f}"

    if allocated >= item.amount - 0.01:
        action = "pay_now"
        why = (
            f"Pay {label} in full ({due}). "
            + (
                "This one cannot be split — it is all or nothing, so paying part "
                "of it would still count as missing it. "
                if item.is_rigid
                else ""
            )
            + f"Missing it would add about {fee} in late fees."
        )
    elif allocated > 0.01:
        short = item.amount - allocated
        action = "pay_partial"
        why = (
            f"Pay ₹{allocated:,.0f} towards {label} now and leave ₹{short:,.0f} "
            f"for later ({due}). There is not enough cash to clear everything "
            f"this period, and this bill charges a lower late fee than the ones "
            f"being paid in full, so delaying part of it costs the least."
        )
    else:
        action = "defer"
        why = (
            f"Hold off on {label} for now ({due}). "
            f"Its late fee of roughly {fee} is smaller than the fees on the "
            f"bills being paid first, so if something has to wait, this costs "
            f"you the least. Worth a call to ask for more time."
        )
    return action, why


@router.get("/decisions/plan", tags=["decisions"])
def decision_plan(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    """The recommended payment plan, with a justification per obligation.

    This is the endpoint the non-technical view reads. It runs the real LP
    solve, then writes `DecisionActionItem.justification` for each obligation
    from that solve's own allocation plus the obligation's contract terms —
    the schema field existed but nothing populated it, so the plain-language
    layer had no source and the UI would have had to invent the sentence.

    The solver comparison stays on `/decisions/solvers`; this route answers the
    different question of *what should I do today, and why*.
    """
    ds, v = _world(), _view()
    problem = obligations_as_of(ds, v, rng=np.random.default_rng(DEMO_SEED))
    lp = solve_lp(problem)
    baseline = solve_rule_based(problem)

    by_id = {o.obligation_id: o for o in problem.obligations}
    allocated_by_id = {a.obligation_id: float(a.allocated_amount) for a in lp.allocations}

    actions = []
    for o in sorted(problem.obligations, key=lambda x: x.days_until_due):
        alloc = allocated_by_id.get(o.obligation_id, 0.0)
        # Tolerance, not `> 0`: the allocation round-trips through Decimal, so a
        # fully-funded obligation leaves a ~1e-10 residue. Treating that as a
        # real shortfall made the justification quote a late fee of zero on
        # every bill that was actually being paid in full.
        raw_unpaid = o.amount - alloc
        unpaid = raw_unpaid if raw_unpaid > 0.01 else 0.0
        fee_if_unpaid = o.penalty_rate * unpaid if unpaid > 0 else o.max_penalty
        action_type, why = _justify(o, alloc, fee_if_unpaid)
        actions.append(
            {
                "obligation_id": o.obligation_id,
                "label": _plain_label(o.obligation_id),
                "category": _category_of(o.obligation_id),
                "action_type": action_type,
                "amount_due": round(o.amount, 2),
                "allocated_amount": round(alloc, 2),
                "shortfall": round(unpaid, 2),
                "days_until_due": round(o.days_until_due, 1),
                "is_rigid": o.is_rigid,
                "late_fee_if_unpaid": round(fee_if_unpaid, 2),
                "justification": why,
            }
        )

    funded = sum(1 for a in actions if a["action_type"] == "pay_now")
    total_fees = sum(a["late_fee_if_unpaid"] for a in actions if a["shortfall"] > 0.01)

    return {
        "as_of": v.as_of.isoformat(),
        "available_cash": round(problem.available_cash, 2),
        "total_obligations_amount": round(problem.total_amount, 2),
        "solver_name": getattr(lp.solver_name, "value", str(lp.solver_name)),
        "solver_status": lp.status,
        "objective_value": round(lp.objective_value, 2),
        "review_status": "pending_review",
        "n_obligations": len(actions),
        "n_paid_in_full": funded,
        "expected_late_fees": round(total_fees, 2),
        "shortfall": round(max(0.0, problem.total_amount - problem.available_cash), 2),
        "actions": actions,
        # Carried so the plain view can state the honest caveat without a second
        # round trip: this plan is what the exact solver recommends, and the
        # replay showed the exact solver losing to the simple rule.
        "baseline_comparison": {
            "rules_objective_value": round(baseline.objective_value, 2),
            "lp_objective_value": round(lp.objective_value, 2),
            "lp_better_on_this_instance": bool(
                lp.objective_value < baseline.objective_value - 1e-6
            ),
            "caveat": (
                "On this one day the exact optimizer plans a lower total late fee "
                "than the simple rule. Over a 49-step replay it did NOT come out "
                "ahead — planning against an imperfect cash forecast made it "
                "commit money that never arrived. See the backtest for the "
                "measured comparison."
            ),
        },
    }


@router.post("/decisions/{decision_id}/approve", tags=["decisions"])
def approve_decision(
    decision_id: str,
    user: TokenPayload = Depends(require_role(UserRole.OWNER)),
) -> dict[str, Any]:
    """Approve a plan. OWNER only — an accountant may prepare but not authorize."""
    entry = AUDIT.append(
        actor_id=user.sub,
        action="approve_decision",
        entity_type="decision_plan",
        entity_id=decision_id,
        payload={"role": user.role, "business_id": user.business_id},
    )
    return {
        "decision_id": decision_id,
        "review_status": "approved",
        "audit_sequence": entry.sequence,
        "audit_hash": entry.entry_hash,
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

@router.get("/audit/chain", tags=["audit"])
def audit_chain(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    return {
        "head_hash": AUDIT.head_hash,
        "n_entries": len(AUDIT.entries),
        "entries": [e.__dict__ for e in AUDIT.entries],
    }


@router.get("/audit/verify", tags=["audit"])
def audit_verify(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    ok, bad_seq, message = AUDIT.verify()
    return {
        "valid": ok,
        "first_broken_sequence": bad_seq,
        "message": message,
        "caveat": (
            "A hash chain proves tamper-EVIDENCE, not tamper-resistance. Anyone "
            "able to rewrite the whole log can recompute every hash. Real "
            "protection requires publishing the head hash somewhere the "
            "attacker does not control."
        ),
    }


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

def _find_artifact(name: str):
    """Locate a generated backtest artifact relative to wherever uvicorn started."""
    from pathlib import Path

    for candidate in (Path(name), Path("..") / name, Path("../..") / name):
        if candidate.exists():
            return candidate
    return None


def _require_summary() -> dict[str, Any]:
    """Load the JSON summary, or 404 with instructions.

    Never synthesizes a fallback series. A dashboard drawing an invented regret
    curve is indistinguishable from one drawing a real one, which is precisely
    the failure this project refuses everywhere else.
    """
    import json

    path = _find_artifact("BACKTEST_REPORT.json")
    if path is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "backtest_not_generated",
                "message": "No backtest summary exists yet. Nothing is served in "
                           "its place, because a placeholder chart cannot be "
                           "told apart from a real one.",
                "details": {
                    "how_to_generate": (
                        "python -m app.services.backtesting.run_backtest "
                        "--step-days 21 --output ../BACKTEST_REPORT.md"
                    )
                },
            },
        )
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/backtest/report", tags=["backtest"])
def backtest_report(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    """Serve the generated backtest report, or say plainly that it is absent."""
    path = _find_artifact("BACKTEST_REPORT.md")
    if path is not None:
        return {"source": str(path), "markdown": path.read_text(encoding="utf-8")}
    raise HTTPException(
        status_code=404,
        detail={
            "error_code": "report_not_generated",
            "message": "No backtest report exists yet.",
            "details": {
                "how_to_generate": "python -m app.services.backtesting.run_backtest"
            },
        },
    )


@router.get("/backtest/summary", tags=["backtest"])
def backtest_summary(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    """The whole measured run: config, strategies, per-step series, calibration.

    Served from the artifact `run_backtest` wrote, because re-running the
    replay per request is tens of minutes of SARIMAX refits. The run's own
    `generated_at` travels with it so the caller can see how stale it is rather
    than having to assume.
    """
    return _require_summary()


@router.get("/backtest/solvers", tags=["backtest"])
def backtest_solver_comparison(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    """Rules vs LP vs DP vs chance-constrained, over the full replay.

    Distinct from `/decisions/solvers`, which compares the four on a SINGLE
    instance and answers "do they agree". This answers the different and more
    important question: over 49 real decision points, what did each one
    actually cost, and how often did it commit money that never arrived.
    """
    s = _require_summary()
    strategies = s["strategies"]
    base = next(x for x in strategies if x["name"] == "rules_baseline")
    best = min(strategies, key=lambda x: x["total_realized_penalty"])

    return {
        "config": s["config"],
        "generated_at": s["generated_at"],
        "strategies": [
            {
                **{k: v for k, v in x.items() if k != "regret_series"},
                "vs_rules_baseline": (
                    (x["total_realized_penalty"] - base["total_realized_penalty"])
                    / base["total_realized_penalty"]
                    if base["total_realized_penalty"] > 0
                    else 0.0
                ),
            }
            for x in strategies
        ],
        "finding": (
            f"The rules baseline was cheapest ({base['total_realized_penalty']:,.0f}); "
            f"the LP optimizer cost "
            f"{(next(x for x in strategies if x['name'] == 'lp_optimizer')['total_realized_penalty'] - base['total_realized_penalty']) / base['total_realized_penalty']:+.1%} "
            "more. Solving the allocation exactly against an imperfect cash "
            "forecast over-commits; the baseline's myopia leaves slack that "
            "absorbs forecast error. The chance-constrained variant eliminated "
            "over-commitment entirely but paid heavily for it. None of the four "
            "dominates."
        )
        if best["name"] == "rules_baseline"
        else (
            f"{best['name']} achieved the lowest realized penalty "
            f"({best['total_realized_penalty']:,.0f})."
        ),
    }


@router.get("/backtest/calibration", tags=["backtest"])
def backtest_calibration(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    """Interval coverage: pooled, by horizon band, and per-step q̂ over time."""
    s = _require_summary()
    return {
        "generated_at": s["generated_at"],
        "calibration": s["calibration"],
        "steps": [
            {
                "as_of": st["as_of"],
                "forecast_model": st["forecast_model"],
                "conformal_q_hat": st["conformal_q_hat"],
                "runway_at_risk_days": st["runway_at_risk_days"],
                "conditional_runway_at_risk_days": st["conditional_runway_at_risk_days"],
                "mc_standard_error": st["mc_standard_error"],
            }
            for st in s["steps"]
        ],
    }


# ---------------------------------------------------------------------------
# Tier 5 — honest stubs
# ---------------------------------------------------------------------------

@router.post("/graph/query", tags=["graph (not implemented)"])
def graph_query(user: TokenPayload = Depends(current_user)):
    raise _not_implemented(
        "Graph RAG over the counterparty knowledge graph",
        "Requires a Neo4j instance and an ingested relationship graph; neither is "
        "provisioned. Ranked below the statistical core by the build priority order.",
    )


@router.post("/chat", tags=["chat (not implemented)"])
def chat(user: TokenPayload = Depends(current_user)):
    raise _not_implemented(
        "Multilingual chatbot",
        "No LLM backend is wired. Returning canned text here would imply a working "
        "model, which is exactly the kind of false capability this project avoids.",
    )


@router.post("/voice/transcribe", tags=["voice (not implemented)"])
def voice(user: TokenPayload = Depends(current_user)):
    raise _not_implemented(
        "Voice interface",
        "Requires a speech-to-text service and a bidirectional audio WebSocket; "
        "neither is provisioned.",
    )
