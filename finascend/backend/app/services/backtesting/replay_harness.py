"""Section C — day-by-day replay with a hard no-look-ahead boundary.

The harness walks a synthetic history forward one step at a time. At each
step it may use ONLY what an observer standing on that date could have known,
generates a forecast (A.1), a risk estimate (A.2) and an action plan (A.3),
and records what it decided. Afterwards, `regret.py` compares those decisions
against what a planner with perfect foresight would have done.

WHY THE BOUNDARY IS ENFORCED IN CODE
------------------------------------
Look-ahead leakage is the single easiest way to produce impressive backtest
numbers that mean nothing, and it is almost invisible in the output: a leaky
backtest looks like a good model. The only defence is to make the leak
structurally impossible rather than relying on discipline, so every step
routes through `pipeline.build_as_of_view`, which slices the world once and
returns an object containing nothing dated after `as_of`. The step function
never sees the full dataset at all — it cannot leak what it was not given.

STEP SPACING
------------
Steps are spaced by `step_days` rather than run daily. Refitting SARIMAX and
running 10,000 Monte Carlo iterations every single day over a multi-month
window is minutes of compute for information that barely moves between
consecutive days. The spacing is a stated cost/resolution trade, not a
shortcut, and `step_days=1` still works if the resolution is wanted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from app.schemas.quant import RunwayAtRisk, SolverName, SolverSolution
from app.services.quant_core.forecasting import select_and_forecast
from app.services.quant_core.monte_carlo_engine import run_simulation
from app.services.quant_core.optimization.problem import (
    AllocationProblem,
    Flexibility,
    ObligationItem,
    penalty_rate_from_terms,
)
from app.services.quant_core.pipeline import AsOfView, build_as_of_view
from app.services.quant_core.synthetic_data import SyntheticDataset


@dataclass(frozen=True)
class ReplayStep:
    """One decision point, with everything needed to score it later."""

    as_of: date
    opening_balance: float
    forecast_path: np.ndarray          # point forecast of net_ex_receipts
    forecast_lower: np.ndarray
    forecast_upper: np.ndarray
    forecast_model: str
    # The conformal multiplier applied at this step. Recorded because it is the
    # per-step evidence that the interval was recalibrated rather than widened:
    # it should sit near 1.0 for a model whose own uncertainty estimate was
    # already honest.
    conformal_q_hat: float
    runway_at_risk: RunwayAtRisk
    plan: SolverSolution
    problem: AllocationProblem
    n_receivables: int


@dataclass
class ReplayResult:
    steps: list[ReplayStep] = field(default_factory=list)
    horizon_days: int = 90
    step_days: int = 7
    plan_generator: SolverName = SolverName.PULP_CBC


def obligations_as_of(
    ds: SyntheticDataset,
    view: AsOfView,
    *,
    rng: np.random.Generator,
    horizon_days: int = 30,
) -> AllocationProblem:
    """Build the payables the business must decide about on `as_of`.

    The synthetic generator models receivables (money in) explicitly but
    aggregates costs into a single daily outflow series. To give the optimizer
    something to allocate across, the next `horizon_days` of outflow are
    decomposed into discrete obligations with contract terms.

    This decomposition is generated from the world's own outflow level rather
    than invented, so the amounts trace back to the generator. The split into
    categories with different late-fee rates and flexibilities reflects real
    structure: payroll and rent are rigid and punitive, vendor payments are
    negotiable.
    """
    upcoming_outflow = float(
        view.daily["outflow"].tail(horizon_days).sum()
    )
    # Category, share of spend, late fee per day, flexibility.
    # Shares sum to 1.0; they are a fixed chart-of-accounts structure, not a
    # tuned parameter.
    categories = [
        ("payroll", 0.35, 0.0020, Flexibility.RIGID),
        ("rent", 0.15, 0.0015, Flexibility.RIGID),
        ("loan_emi", 0.12, 0.0025, Flexibility.RIGID),
        ("vendor_payment", 0.25, 0.0006, Flexibility.FLEXIBLE),
        ("tax", 0.08, 0.0018, Flexibility.MODERATE),
        ("utilities", 0.05, 0.0008, Flexibility.FLEXIBLE),
    ]

    items: list[ObligationItem] = []
    for i, (name, share, fee, flex) in enumerate(categories):
        amount = upcoming_outflow * share
        days_due = float(rng.integers(1, horizon_days))
        items.append(
            ObligationItem(
                obligation_id=f"{name}-{view.as_of.isoformat()}",
                amount=amount,
                penalty_rate=penalty_rate_from_terms(
                    late_fee_rate_per_day=fee,
                    days_overdue_if_unpaid=float(horizon_days - days_due),
                ),
                flexibility=flex,
                days_until_due=days_due,
            )
        )

    return AllocationProblem(
        available_cash=max(0.0, view.opening_balance), obligations=tuple(items)
    )


def replay(
    ds: SyntheticDataset,
    *,
    start: Optional[date] = None,
    end: Optional[date] = None,
    step_days: int = 7,
    horizon_days: int = 90,
    n_iterations: int = 4000,
    seed: int = 42,
    plan_fn: Optional[Callable[[AllocationProblem, np.ndarray], SolverSolution]] = None,
    plan_generator: SolverName = SolverName.PULP_CBC,
    min_history_days: int = 400,
    shared_cache: Optional[dict] = None,
) -> ReplayResult:
    """Walk the history forward, deciding with information available at each step.

    Args:
        ds: the synthetic world.
        start: first decision date; defaults to `min_history_days` in.
        end: last decision date; defaults to leaving `horizon_days` of future
            available for scoring the decision against what actually happened.
        step_days: spacing between decision points.
        horizon_days: forecast and simulation horizon.
        n_iterations: Monte Carlo draws per step. Lower than production's
            10,000 because the harness runs dozens of steps; the convergence
            study says what accuracy this buys.
        seed: base seed; each step derives a distinct seed from it.
        plan_fn: allocation strategy. Defaults to the LP. Swapping in the
            rules baseline is how the backtest measures the optimizer's value.
        min_history_days: history required before the first decision, so the
            forecaster has enough data to fit.
        shared_cache: optional dict for reusing the forecast and simulation
            across strategies. The forecast and the Monte Carlo simulation
            depend only on `as_of`, the data and the seed — never on
            `plan_fn` — so replaying four strategies over the same history
            otherwise refits SARIMAX and redraws the simulation four times for
            identical results. Passing one dict across the four calls is also
            what guarantees every strategy decides from the *same* forecast, so
            any difference between them is attributable to the allocation
            method rather than to Monte Carlo noise.

    Returns:
        `ReplayResult` with one `ReplayStep` per decision point.
    """
    from app.services.quant_core.optimization.lp_solver import solve_lp

    if plan_fn is None:
        plan_fn = lambda problem, _balances: solve_lp(problem)

    dates = pd.to_datetime(ds.daily["date"])
    first_possible = dates.iloc[min_history_days].date()
    last_possible = (dates.iloc[-1] - pd.Timedelta(days=horizon_days)).date()

    cur = start or first_possible
    stop = end or last_possible
    if cur > stop:
        raise ValueError(
            f"no room to replay: need at least {min_history_days + horizon_days} "
            f"days of history, have {len(dates)}"
        )

    result = ReplayResult(
        horizon_days=horizon_days, step_days=step_days, plan_generator=plan_generator
    )
    step_no = 0

    while cur <= stop:
        # THE BOUNDARY: this view contains nothing dated after `cur`.
        view = build_as_of_view(ds, cur)
        rng = np.random.default_rng(seed + step_no)

        cache_key = (cur, horizon_days, n_iterations, seed)
        if shared_cache is not None and cache_key in shared_cache:
            forecast, paths, rar = shared_cache[cache_key]
        else:
            forecast = select_and_forecast(
                view.daily,
                horizon_days=horizon_days,
                value_column="net_ex_receipts",
                opening_balance=view.opening_balance,
            )
            paths, rar, _spec = run_simulation(
                opening_balance=view.opening_balance,
                forecast=forecast,
                receivables=view.receivables,
                delays_by_cp=view.delays_by_cp,
                panel=view.delay_panel,
                n_iterations=n_iterations,
                seed=seed + step_no,
                horizon_days=horizon_days,
            )
            if shared_cache is not None:
                shared_cache[cache_key] = (forecast, paths, rar)

        problem = obligations_as_of(ds, view, rng=rng)
        plan = plan_fn(problem, paths.balances)

        result.steps.append(
            ReplayStep(
                as_of=cur,
                opening_balance=view.opening_balance,
                forecast_path=np.array([p.point for p in forecast.path]),
                forecast_lower=np.array([p.lower for p in forecast.path]),
                forecast_upper=np.array([p.upper for p in forecast.path]),
                forecast_model=str(forecast.selected_model),
                conformal_q_hat=float(forecast.conformal_q_hat or 1.0),
                runway_at_risk=rar,
                plan=plan,
                problem=problem,
                n_receivables=len(view.receivables),
            )
        )

        step_no += 1
        cur = cur + timedelta(days=step_days)

    return result
