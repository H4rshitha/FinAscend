"""A.3 — Chance-constrained allocation via Sample Average Approximation.

This is the module that actually connects the optimization layer to the risk
layer. Everywhere else in A.3, `available_cash` is treated as a known
constant — but A.2 has just finished demonstrating that it is not: future cash
depends on when correlated, uncertain receivables arrive. Optimizing against a
single point estimate of available cash and then reporting a 95% Runway-at-
Risk is internally inconsistent.

THE PROBLEM
-----------
Choose an allocation x that minimizes expected penalty subject to

    P( cash goes negative within the horizon | x )  <=  epsilon

A true chance constraint is not tractable directly, because the probability is
an integral over the joint distribution of arrival times. Sample Average
Approximation replaces it with an empirical frequency over S sampled scenarios
drawn from exactly the Monte Carlo machinery in A.2:

    (1/S) * sum_s 1[ shortfall in scenario s ]  <=  epsilon

METHOD: SAFE-CASH REDUCTION RATHER THAN BIG-M SCENARIO BINARIES
--------------------------------------------------------------
The textbook SAA formulation introduces one binary variable per scenario
(z_s = 1 if scenario s violates), giving S binaries and a big-M linking
constraint. With S in the hundreds and rigid-obligation binaries already
present, that MILP becomes slow exactly where the Section C harness re-solves
it on every replay day.

This implementation instead exploits the structure of *this* problem: the
constraint couples to the decision only through total cash spent. Because
spending is monotone — spending more can only make a shortfall more likely,
never less — the chance constraint reduces to a single budget cap:

    spend  <=  Q_epsilon,  where Q_epsilon is the epsilon-quantile of the
                           simulated minimum free cash across scenarios

That is exact for this problem, not an approximation of it, and it collapses
to one deterministic MILP solve. The monotonicity argument is the thing to be
able to defend: it is what makes the reduction legitimate, and it would fail
if paying an obligation could itself generate inflow (e.g. a settlement
discount), which this model does not include.

The scenario cap and its stability check are the honest part: a small S makes
Q_epsilon itself a noisy statistic, so `assess_scenario_stability` measures
how much the answer moves across independent resamples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.schemas.quant import (
    ChanceConstrainedResult,
    SolverName,
    SolverSolution,
)
from app.services.quant_core.optimization.lp_solver import solve_lp
from app.services.quant_core.optimization.problem import AllocationProblem

# Scenario subsample cap. Deliberately NOT the full Monte Carlo draw count:
# the SAA problem is re-solved on every replay day by the Section C harness,
# so cost matters. 200-500 is the design range; the default sits mid-range and
# `assess_scenario_stability` is what justifies it for a given instance rather
# than the number being asserted.
DEFAULT_SAA_SCENARIOS = 300


@dataclass(frozen=True)
class ScenarioBatch:
    """A subsample of Monte Carlo cash paths used for the SAA constraint."""

    min_free_cash: np.ndarray   # (S,) minimum balance reached in each scenario
    n_drawn: int
    seed: int


def draw_scenarios(
    balances: np.ndarray,
    *,
    n_scenarios: int = DEFAULT_SAA_SCENARIOS,
    seed: int = 0,
) -> ScenarioBatch:
    """Subsample Monte Carlo balance paths down to the SAA scenario cap.

    Args:
        balances: (n_iterations, horizon_days) from `simulate_cash_paths`.
        n_scenarios: subsample size; capped at the number available.
        seed: subsample seed, so the SAA solve is reproducible.

    Returns:
        `ScenarioBatch` holding each scenario's minimum balance — the only
        statistic the chance constraint depends on, since a shortfall is
        exactly the event that the running minimum drops below zero.
    """
    rng = np.random.default_rng(seed)
    n_avail = balances.shape[0]
    take = min(n_scenarios, n_avail)
    idx = rng.choice(n_avail, size=take, replace=False)
    return ScenarioBatch(
        min_free_cash=balances[idx].min(axis=1),
        n_drawn=take,
        seed=seed,
    )


def safe_spend_limit(batch: ScenarioBatch, epsilon: float) -> float:
    """The largest spend that keeps P(shortfall) <= epsilon.

    If we spend `s` today, every scenario's balance path shifts down by `s`,
    so scenario i suffers a shortfall exactly when `min_free_cash[i] - s < 0`.
    The fraction of scenarios violating is therefore
    `mean(min_free_cash < s)`, which is the empirical CDF evaluated at s.
    Requiring that to be at most epsilon gives

        s*  =  epsilon-quantile of min_free_cash

    Clipped at zero: a business already projected into shortfall in more than
    epsilon of scenarios has no safe spend, and reporting a negative limit
    would be meaningless.
    """
    q = float(np.quantile(batch.min_free_cash, epsilon))
    return max(0.0, q)


def solve_chance_constrained(
    problem: AllocationProblem,
    balances: np.ndarray,
    *,
    epsilon: float = 0.05,
    n_scenarios: int = DEFAULT_SAA_SCENARIOS,
    seed: int = 0,
    stability_resamples: int = 8,
) -> ChanceConstrainedResult:
    """Solve the allocation subject to P(shortfall) <= epsilon.

    Args:
        problem: the nominal allocation instance. Its `available_cash` is
            treated as an upper bound; the chance constraint may lower it.
        balances: Monte Carlo balance paths from A.2.
        epsilon: maximum tolerated shortfall probability.
        n_scenarios: SAA subsample size (the cap, not the full draw count).
        seed: reproducibility seed for the subsample.
        stability_resamples: independent resamples used to report how much the
            answer depends on which scenarios were drawn.

    Returns:
        `ChanceConstrainedResult` with the solved allocation, the realized
        shortfall probability, and the stability diagnostic.
    """
    batch = draw_scenarios(balances, n_scenarios=n_scenarios, seed=seed)
    limit = safe_spend_limit(batch, epsilon)

    # Is the constraint satisfiable AT ALL? Spending nothing is the safest
    # possible action, so if the business already breaches epsilon at zero
    # spend, no allocation can satisfy the constraint. Reporting that as a
    # bland "spend 0" would be actively misleading: it looks like a cautious
    # recommendation when it is really the model saying the target is
    # unreachable and the business needs financing or renegotiation, not a
    # cleverer payment schedule. This distinction is surfaced, not swallowed.
    baseline_shortfall = float(np.mean(batch.min_free_cash < 0.0))
    infeasible = baseline_shortfall > epsilon

    # The risk-adjusted budget: never more than we hold, never more than is safe.
    risk_adjusted_cash = min(problem.available_cash, limit)
    constrained = AllocationProblem(
        available_cash=risk_adjusted_cash, obligations=problem.obligations
    )
    solution = solve_lp(constrained)
    spend = float(sum(float(a.allocated_amount) for a in solution.allocations))

    achieved = float(np.mean(batch.min_free_cash < spend))

    if infeasible:
        status = (
            f"Infeasible: P(shortfall)={baseline_shortfall:.1%} at ZERO spend, "
            f"already above epsilon={epsilon:.1%}. No allocation satisfies the "
            "chance constraint — the shortfall is driven by the cash position "
            "and receivable timing, not by discretionary payments. The "
            "actionable levers are financing, collection acceleration, or "
            "renegotiating terms, none of which this solver controls."
        )
    else:
        status = f"{solution.status} (chance-constrained, safe spend limit {limit:,.0f})"

    stability = assess_scenario_stability(
        balances,
        epsilon=epsilon,
        n_scenarios=n_scenarios,
        seed=seed,
        resamples=stability_resamples,
    )

    return ChanceConstrainedResult(
        epsilon=epsilon,
        saa_num_scenarios=batch.n_drawn,
        achieved_shortfall_probability=achieved,
        solution=SolverSolution(
            solver_name=SolverName.SAA_CHANCE_CONSTRAINED,
            status=status,
            objective_value=solution.objective_value,
            allocations=solution.allocations,
            solve_seconds=solution.solve_seconds,
        ),
        stability_across_resamples=stability,
    )


def assess_scenario_stability(
    balances: np.ndarray,
    *,
    epsilon: float,
    n_scenarios: int,
    seed: int,
    resamples: int = 8,
) -> float:
    """Std dev of the safe-spend limit across independent scenario subsamples.

    This is the evidence that the scenario cap is large enough, and it is the
    same argument A.2 makes for its iteration count: a number is defensible
    when its sampling error is small relative to what it decides, not when it
    looks like a round figure.

    Returned as a **coefficient of variation** (std / mean) so it can be read
    without knowing the business's scale. A value near zero means the answer
    does not depend on which scenarios happened to be drawn.
    """
    limits = []
    for r in range(resamples):
        batch = draw_scenarios(balances, n_scenarios=n_scenarios, seed=seed + 1000 + r)
        limits.append(safe_spend_limit(batch, epsilon))
    arr = np.array(limits, dtype=float)
    mean = float(arr.mean())
    if mean <= 0:
        return 0.0
    return float(arr.std(ddof=1) / mean)
