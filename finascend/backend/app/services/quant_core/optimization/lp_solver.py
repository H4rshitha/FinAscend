"""A.3 — Linear / mixed-integer program for cash allocation.

Method: `PuLP` modelling the problem in `problem.py` and solving with CBC
(branch-and-cut). PuLP is used rather than `scipy.optimize.milp` because the
model has a natural per-obligation variable structure that reads clearly in
algebraic form, and because CBC handles the mixed continuous/binary case
without the caller having to hand-assemble constraint matrices — which is
where sign errors hide.

The model:

    variables   x_i in [0, a_i]              amount paid to obligation i
                y_i in {0, 1}                for RIGID i only
    minimize    sum_i w_i * (a_i - x_i)
    subject to  sum_i x_i <= C
                x_i = a_i * y_i              for RIGID i

The RIGID linkage is an equality, not the more common pair of big-M
inequalities, because `x_i = a_i * y_i` is exact and needs no big-M constant.
Big-M formulations are a standard source of silent wrongness: too small and
you cut off the optimum, too large and the LP relaxation becomes so loose that
branch-and-bound crawls.

Note this is a genuine MILP whenever any obligation is RIGID; it is a pure LP
only when every obligation permits partial payment.
"""

from __future__ import annotations

import time

import pulp

from app.schemas.quant import AllocationItem, SolverName, SolverSolution
from app.services.quant_core.optimization.problem import (
    AllocationProblem,
    total_penalty,
)


def solve_lp(
    problem: AllocationProblem,
    *,
    msg: bool = False,
    time_limit_seconds: float | None = 30.0,
) -> SolverSolution:
    """Solve the allocation problem as an LP/MILP with CBC.

    Args:
        problem: the instance to solve.
        msg: pass True to let CBC print its log.
        time_limit_seconds: CBC wall-clock cap. A hit limit surfaces in
            `status` as something other than "Optimal" rather than being
            silently reported as a solution.

    Returns:
        `SolverSolution` with the objective expressed as **total penalty
        incurred**, matching `problem.total_penalty`, so it is directly
        comparable to the DP's objective.
    """
    t0 = time.perf_counter()
    model = pulp.LpProblem("finascend_allocation", pulp.LpMinimize)

    x: dict[str, pulp.LpVariable] = {}
    y: dict[str, pulp.LpVariable] = {}

    for o in problem.obligations:
        x[o.obligation_id] = pulp.LpVariable(
            f"x_{o.obligation_id}", lowBound=0.0, upBound=o.amount, cat="Continuous"
        )
        if o.is_rigid:
            y[o.obligation_id] = pulp.LpVariable(
                f"y_{o.obligation_id}", cat="Binary"
            )
            # Exact linkage: paying a rigid obligation means paying all of it.
            model += x[o.obligation_id] == o.amount * y[o.obligation_id]

    # Objective: total penalty incurred. The constant term sum(w_i * a_i) is
    # kept rather than dropped so the reported objective is directly
    # comparable with the DP's, which is the whole point of the cross-check.
    model += pulp.lpSum(
        o.penalty_rate * (o.amount - x[o.obligation_id]) for o in problem.obligations
    )

    model += (
        pulp.lpSum(x[o.obligation_id] for o in problem.obligations)
        <= problem.available_cash
    )

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_seconds)
    model.solve(solver)
    status = pulp.LpStatus[model.status]

    allocation = [
        max(0.0, float(x[o.obligation_id].value() or 0.0)) for o in problem.obligations
    ]

    return SolverSolution(
        solver_name=SolverName.PULP_CBC,
        status=status,
        objective_value=total_penalty(problem, allocation),
        allocations=[
            AllocationItem(
                obligation_id=o.obligation_id,
                allocated_amount=round(a, 2),
                fully_funded=abs(a - o.amount) < 1e-6,
            )
            for o, a in zip(problem.obligations, allocation)
        ],
        solve_seconds=time.perf_counter() - t0,
    )
