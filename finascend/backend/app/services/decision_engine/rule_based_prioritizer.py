"""Section B — the rule-based prioritizer, kept deliberately as the BASELINE.

This module is not legacy code awaiting deletion. It is the comparison point
that gives every claim about the optimizer its meaning. "The LP saves 18% of
penalty" is only informative against something, and this is that something.

The rules encoded here are what a competent human credit controller actually
does: pay the most urgent and most expensive-to-miss obligations first, ranked
by penalty per rupee, and stop when the cash runs out. That is a genuine
attempt, not a straw man — it is greedy on penalty density, which is the
optimal strategy for a *fractional* knapsack (Dantzig 1957).

WHERE IT IS PROVABLY WRONG
--------------------------
Greedy-by-density is exactly optimal when every obligation can be partially
funded. It stops being optimal the moment any obligation is RIGID, because
then the problem is a 0/1 knapsack, and greedy on 0/1 knapsack has no constant
approximation guarantee: it can be made arbitrarily bad. The classic failure
is a high-density item that consumes the whole budget and blocks two
slightly-lower-density items that together were worth far more.

So the baseline is not merely "less clever" — it is wrong in a specific,
demonstrable way, and the backtest quantifies how often that costs real money.
That is a far more useful thing to be able to say than "we used an optimizer".
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.quant import AllocationItem, SolverName, SolverSolution
from app.services.quant_core.optimization.problem import (
    AllocationProblem,
    ObligationItem,
    total_penalty,
)


@dataclass(frozen=True)
class PriorityRanking:
    """Ranked obligations with the score that produced the ranking."""

    obligation_id: str
    rank: int
    score: float
    reason: str


def penalty_density(o: ObligationItem) -> float:
    """Penalty avoided per rupee spent — the greedy ranking key.

    density = w_i * a_i / a_i = w_i

    It reduces to the penalty rate itself, which is worth stating explicitly
    because it makes the rule's blind spot obvious: ranking by density ignores
    obligation SIZE entirely. A tiny obligation with a punitive late fee
    outranks a large one with a moderate fee, even when funding the large one
    would avoid ten times as much total penalty.
    """
    return o.penalty_rate


def prioritize(problem: AllocationProblem) -> list[PriorityRanking]:
    """Rank obligations by the rule: penalty density, then urgency, then size.

    Ties on density are broken by days-until-due (sooner first) and then by
    amount (larger first). Tie-breaks are specified rather than left to sort
    stability so the ranking is deterministic and reproducible across runs.
    """
    ordered = sorted(
        problem.obligations,
        key=lambda o: (-penalty_density(o), o.days_until_due, -o.amount),
    )
    out: list[PriorityRanking] = []
    for rank, o in enumerate(ordered, start=1):
        out.append(
            PriorityRanking(
                obligation_id=o.obligation_id,
                rank=rank,
                score=penalty_density(o),
                reason=(
                    f"penalty rate {o.penalty_rate:.5f}/rupee, due in "
                    f"{o.days_until_due:.0f} days, amount {o.amount:,.0f}"
                    + (" (RIGID: all-or-nothing)" if o.is_rigid else "")
                ),
            )
        )
    return out


def solve_rule_based(problem: AllocationProblem) -> SolverSolution:
    """Greedy allocation following the ranking, for direct comparison with A.3.

    Returns a `SolverSolution` reporting **total penalty incurred**, the same
    quantity the LP and DP report, so the three are directly comparable and
    the baseline's cost can be measured rather than asserted.

    A RIGID obligation is skipped entirely when the remaining cash cannot
    cover it in full — and critically, the rule then moves on to the next
    obligation rather than reconsidering earlier choices. That myopia is the
    behaviour the optimizer improves on.
    """
    import time

    t0 = time.perf_counter()
    ranking = prioritize(problem)
    order = {r.obligation_id: r.rank for r in ranking}
    by_id = {o.obligation_id: o for o in problem.obligations}

    remaining = problem.available_cash
    allocation: dict[str, float] = {o.obligation_id: 0.0 for o in problem.obligations}

    for oid in sorted(order, key=lambda k: order[k]):
        o = by_id[oid]
        if remaining <= 0:
            break
        if o.is_rigid:
            if remaining + 1e-9 >= o.amount:
                allocation[oid] = o.amount
                remaining -= o.amount
            # else: skip and continue — no backtracking. This is the myopia.
        else:
            take = min(o.amount, remaining)
            allocation[oid] = take
            remaining -= take

    alloc_list = [allocation[o.obligation_id] for o in problem.obligations]

    return SolverSolution(
        solver_name=SolverName.RULE_BASED_BASELINE,
        status="Greedy by penalty density (baseline, not optimal under RIGID constraints)",
        objective_value=total_penalty(problem, alloc_list),
        allocations=[
            AllocationItem(
                obligation_id=o.obligation_id,
                allocated_amount=round(a, 2),
                fully_funded=abs(a - o.amount) < 1e-6,
            )
            for o, a in zip(problem.obligations, alloc_list)
        ],
        solve_seconds=time.perf_counter() - t0,
    )


def measure_optimizer_lift(
    problem: AllocationProblem, optimized: SolverSolution
) -> dict[str, float]:
    """Quantify what the optimizer gains over the rules baseline.

    Reported as a penalty *reduction*, positive when the optimizer wins. A
    negative value would mean the optimizer lost to greedy, which should be
    impossible for an exact solver and would indicate a bug — so the sign is
    itself a check.
    """
    baseline = solve_rule_based(problem)
    saved = baseline.objective_value - optimized.objective_value
    pct = (saved / baseline.objective_value * 100.0) if baseline.objective_value > 0 else 0.0
    return {
        "baseline_penalty": baseline.objective_value,
        "optimized_penalty": optimized.objective_value,
        "penalty_saved": saved,
        "penalty_saved_pct": pct,
    }
