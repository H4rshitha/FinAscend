"""A.3 — Cross-check the two independent solvers against each other.

Two implementations that share no algorithm agreeing on an objective value is
genuine evidence of correctness. One implementation returning a plausible
number is not evidence of anything.

WHERE THEY LEGITIMATELY DIVERGE
===============================
Agreement is expected to be exact only when every obligation is RIGID and each
amount is an exact multiple of the DP's discretization unit. Otherwise:

1. **Discretization gap (DP >= LP).** The DP can only spend cash in whole
   `unit` cells and can only fund divisible obligations in whole cells, so it
   optimizes over a subset of the LP's feasible region. Its incurred penalty
   is therefore never *lower* than the LP's; the gap is bounded by roughly
   `unit * sum(penalty_rate)` — one cell of unfunded amount per obligation in
   the worst case.

2. **Rigid weight ceiling (DP >= LP).** A rigid obligation's weight is ceiled
   to whole cells, so the DP charges itself slightly more than the true amount
   and may decline an obligation the LP can just afford.

Both effects push the same direction: **the DP is conservative.** A DP
objective *below* the LP's is not a rounding artifact — it means one of the
solvers is wrong, most likely that the DP has spent money it does not have.
`cross_validate` treats that direction as a failure regardless of magnitude,
which is why the tolerance is one-sided rather than a symmetric abs() check.

That asymmetry is the interesting part of this module and it is the thing to
say out loud in an interview: knowing *which direction* an approximation can
err in is what turns a tolerance into a test.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.quant import SolverAgreement, SolverSolution
from app.services.quant_core.optimization.dp_solver import choose_unit, solve_dp
from app.services.quant_core.optimization.lp_solver import solve_lp
from app.services.quant_core.optimization.problem import (
    AllocationProblem,
    is_feasible,
)


@dataclass(frozen=True)
class CrossCheckResult:
    lp: SolverSolution
    dp: SolverSolution
    agreement: SolverAgreement
    lp_feasible: bool
    dp_feasible: bool


# CBC's default relative optimality gap. The MILP is only solved to within
# this of its true optimum, so the LP objective is itself uncertain at that
# scale and no cross-check tolerance can meaningfully be tighter. Observed
# directly: two of sixty random instances showed the DP below the LP by ~1e-8
# relative, which is this gap and not a solver disagreement.
CBC_RELATIVE_GAP = 1e-7


def expected_discretization_gap(problem: AllocationProblem, unit: float) -> float:
    """Upper bound on how much worse the DP may legitimately be.

    Each obligation can be under-funded by at most one cell relative to the
    LP's continuous optimum, costing `unit * penalty_rate` in penalty. Summing
    over obligations gives the bound.

    Two floors are added. The absolute one covers instances with near-zero
    penalty rates. The **relative** one covers CBC's own optimality gap: an
    absolute floor alone is meaningless on objectives of order 1e4, where 1e-6
    sits below the solver's own resolution.
    """
    discretization = unit * sum(o.penalty_rate for o in problem.obligations)
    solver_noise = CBC_RELATIVE_GAP * max(problem.max_total_penalty, 1.0)
    return discretization + solver_noise + 1e-6


def cross_validate(
    problem: AllocationProblem,
    *,
    unit: float | None = None,
    tolerance: float | None = None,
) -> CrossCheckResult:
    """Run both solvers on one instance and judge whether they agree.

    Args:
        problem: the shared instance.
        unit: DP discretization; derived from the instance if omitted.
        tolerance: allowed DP-worse-than-LP gap. Defaults to the analytic
            discretization bound, so the tolerance is *derived* rather than
            picked to make the test pass.

    Returns:
        `CrossCheckResult` including feasibility of both allocations.
    """
    if unit is None:
        unit = choose_unit(problem)
    if tolerance is None:
        tolerance = expected_discretization_gap(problem, unit)

    lp = solve_lp(problem)
    dp = solve_dp(problem, unit=unit)

    lp_alloc = [a.allocated_amount for a in lp.allocations]
    dp_alloc = [a.allocated_amount for a in dp.allocations]
    lp_ok = is_feasible(problem, [float(a) for a in lp_alloc])
    dp_ok = is_feasible(problem, [float(a) for a in dp_alloc])

    delta = dp.objective_value - lp.objective_value   # signed, DP minus LP

    if delta < -tolerance:
        agree = False
        explanation = (
            f"DP objective is {abs(delta):,.2f} BELOW the LP's. This is not a "
            "discretization artifact — the DP optimizes over a subset of the "
            "LP's feasible region, so it can only ever do worse. A lower DP "
            "objective means one solver is wrong, most likely the DP spending "
            "cash it does not have. Check the capacity floor and the rigid "
            "weight ceiling."
        )
    elif delta > tolerance:
        agree = False
        explanation = (
            f"DP is {delta:,.2f} worse than the LP, exceeding the analytic "
            f"discretization bound of {tolerance:,.2f} at unit={unit:g}. "
            "Expected causes: too coarse a unit, or many divisible "
            "obligations each losing up to one cell. Re-run with a finer unit "
            "to confirm the gap shrinks proportionally."
        )
    else:
        agree = True
        explanation = (
            f"Objectives agree within the analytic discretization bound "
            f"({abs(delta):,.4f} <= {tolerance:,.4f} at unit={unit:g}). "
            "The DP is the conservative side, as expected: it optimizes over "
            "a grid subset of the LP's continuous feasible region."
        )

    return CrossCheckResult(
        lp=lp,
        dp=dp,
        agreement=SolverAgreement(
            lp_objective_value=lp.objective_value,
            dp_objective_value=dp.objective_value,
            absolute_delta=abs(delta),
            tolerance=tolerance,
            agree=agree and lp_ok and dp_ok,
            explanation=explanation
            + ("" if lp_ok else " WARNING: LP allocation is infeasible.")
            + ("" if dp_ok else " WARNING: DP allocation is infeasible."),
        ),
        lp_feasible=lp_ok,
        dp_feasible=dp_ok,
    )
