"""Section C — the perfect-foresight benchmark that regret is measured against.

The hindsight-optimal plan solves the SAME allocation problem the live system
faced, but with `available_cash` set to what the business ACTUALLY had, rather
than to a forecast of it. Everything else — obligations, penalty rates, rigidity
constraints — is held identical.

WHY THIS IS THE RIGHT BENCHMARK
-------------------------------
It isolates the cost of *uncertainty* from the cost of *bad method*. A planner
with perfect knowledge of future cash still cannot avoid all penalty — the
obligations may simply exceed what the business will ever have. The gap
between the live plan and this benchmark is therefore attributable to not
knowing the future, which is exactly the quantity worth reporting.

Comparing instead against "zero penalty" would conflate an unachievable ideal
with a method failure and would make every system look bad. Comparing against
another heuristic would only say which heuristic is better, not how much is
being left on the table.

This benchmark is deliberately UNACHIEVABLE in production. Reporting a regret
of zero would mean the benchmark is broken, not that the model is perfect.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.schemas.quant import SolverSolution
from app.services.quant_core.optimization.lp_solver import solve_lp
from app.services.quant_core.optimization.problem import AllocationProblem
from app.services.quant_core.synthetic_data import SyntheticDataset


@dataclass(frozen=True)
class HindsightOutcome:
    """What perfect foresight would have achieved at one decision point."""

    as_of: date
    realized_min_cash: float
    hindsight_plan: SolverSolution
    hindsight_penalty: float


def realized_available_cash(
    ds: SyntheticDataset, as_of: date, horizon_days: int
) -> float:
    """The cash the business actually had available over the horizon.

    Defined as the MINIMUM balance realized over the horizon, not the closing
    balance. A plan that spends the closing balance would be infeasible if the
    account dipped below zero mid-horizon and the money was never
    simultaneously available. The minimum is the amount that could have been
    committed on day one without ever going overdrawn — which is precisely
    what the allocation decision commits to.
    """
    ts = pd.Timestamp(as_of)
    window = ds.daily[
        (ds.daily["date"] >= ts)
        & (ds.daily["date"] <= ts + pd.Timedelta(days=horizon_days))
    ]
    if window.empty:
        return 0.0
    return float(max(0.0, window["balance"].min()))


def hindsight_plan(
    ds: SyntheticDataset,
    problem: AllocationProblem,
    as_of: date,
    horizon_days: int,
) -> HindsightOutcome:
    """Solve the same problem with the realized cash position substituted in.

    Uses the exact LP the live system uses, so the comparison isolates the
    information difference and nothing else. Using a different solver here
    would confound "we did not know the future" with "we used a worse
    algorithm".
    """
    realized = realized_available_cash(ds, as_of, horizon_days)
    perfect = AllocationProblem(
        available_cash=realized, obligations=problem.obligations
    )
    solution = solve_lp(perfect)
    return HindsightOutcome(
        as_of=as_of,
        realized_min_cash=realized,
        hindsight_plan=solution,
        hindsight_penalty=solution.objective_value,
    )


def realized_penalty_of_plan(
    ds: SyntheticDataset,
    problem: AllocationProblem,
    plan: SolverSolution,
    as_of: date,
    horizon_days: int,
) -> float:
    """Score a live plan against what actually happened.

    A plan that committed more cash than the business turned out to have is
    not simply "slightly worse" — the excess payments could not have been
    made. Those obligations are treated as UNPAID and incur their full
    penalty, which is the honest accounting: an over-committed plan fails, it
    does not partially succeed.

    Obligations are truncated in reverse priority order (lowest penalty rate
    first), because a business discovering it is short would rationally skip
    its cheapest-to-miss obligations, not its most punitive ones.
    """
    realized = realized_available_cash(ds, as_of, horizon_days)
    by_id = {o.obligation_id: o for o in problem.obligations}

    committed = [
        (by_id[a.obligation_id], float(a.allocated_amount))
        for a in plan.allocations
        if float(a.allocated_amount) > 0
    ]
    # Keep the most punitive commitments; drop the cheapest when short.
    committed.sort(key=lambda pair: -pair[0].penalty_rate)

    budget = realized
    honoured: dict[str, float] = {}
    for obligation, amount in committed:
        if budget <= 0:
            break
        if obligation.is_rigid:
            if budget + 1e-9 >= amount:
                honoured[obligation.obligation_id] = amount
                budget -= amount
        else:
            take = min(amount, budget)
            honoured[obligation.obligation_id] = take
            budget -= take

    return sum(
        o.penalty_rate * (o.amount - honoured.get(o.obligation_id, 0.0))
        for o in problem.obligations
    )
