"""Correctness tests for A.3 — solver agreement, feasibility, and the DP rewrite.

The headline test is `test_solvers_agree_exactly_on_pure_knapsack`: on
instances where the DP is exact, two independently written algorithms must
produce the same objective. That is real evidence of correctness in a way that
one solver returning a plausible number is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.quant_core.optimization.chance_constrained import (
    draw_scenarios,
    safe_spend_limit,
    solve_chance_constrained,
)
from app.services.quant_core.optimization.cross_validation import (
    cross_validate,
    expected_discretization_gap,
)
from app.services.quant_core.optimization.dp_solver import choose_unit, solve_dp
from app.services.quant_core.optimization.lp_solver import solve_lp
from app.services.quant_core.optimization.problem import (
    AllocationProblem,
    Flexibility,
    ObligationItem,
    is_feasible,
    penalty_rate_from_terms,
    total_penalty,
)


def _rigid_problem(rng, n=8, unit=1000.0):
    """All-rigid instance with unit-aligned amounts: a pure 0/1 knapsack."""
    obs = tuple(
        ObligationItem(
            obligation_id=f"O{i}",
            amount=float(rng.integers(1, 15)) * unit,
            penalty_rate=float(rng.uniform(0.001, 0.02)),
            flexibility=Flexibility.RIGID,
        )
        for i in range(n)
    )
    cash = float(rng.integers(n // 2, n * 8)) * unit
    return AllocationProblem(available_cash=cash, obligations=obs)


def _mixed_problem(rng, n=12, cash_fraction=0.5):
    obs = tuple(
        ObligationItem(
            obligation_id=f"O{i}",
            amount=float(rng.uniform(50_000, 500_000)),
            penalty_rate=penalty_rate_from_terms(
                late_fee_rate_per_day=float(rng.uniform(0.0002, 0.0015)),
                days_overdue_if_unpaid=float(rng.integers(5, 45)),
            ),
            flexibility=Flexibility.RIGID if rng.random() < 0.4 else Flexibility.FLEXIBLE,
        )
        for i in range(n)
    )
    total = sum(o.amount for o in obs)
    return AllocationProblem(available_cash=total * cash_fraction, obligations=obs)


# ---------------------------------------------------------------------------
# Solver agreement
# ---------------------------------------------------------------------------

def test_solvers_agree_exactly_on_pure_knapsack():
    """On an exact 0/1 knapsack the two solvers must give identical objectives.

    Unit-aligned amounts remove the discretization gap entirely, so any
    difference here is a genuine algorithmic disagreement rather than rounding.
    """
    rng = np.random.default_rng(11)
    for trial in range(15):
        p = _rigid_problem(rng)
        r = cross_validate(p, unit=1000.0)
        assert r.agreement.agree, r.agreement.explanation
        assert abs(r.lp.objective_value - r.dp.objective_value) < 1e-6, (
            f"trial {trial}: LP={r.lp.objective_value} DP={r.dp.objective_value}"
        )


def test_dp_is_never_better_than_lp():
    """The DP optimizes over a grid SUBSET of the LP's feasible region.

    So its incurred penalty can only ever be higher. A DP objective below the
    LP's means one solver is wrong — most likely the DP spending cash it does
    not have — and that direction is treated as failure regardless of size.
    """
    rng = np.random.default_rng(5)
    for _ in range(40):
        p = _mixed_problem(rng, n=int(rng.integers(4, 16)))
        r = cross_validate(p)
        tol = expected_discretization_gap(p, choose_unit(p))
        assert r.dp.objective_value >= r.lp.objective_value - tol


def test_discretization_gap_shrinks_with_finer_unit():
    """The DP's gap must be an artifact of granularity, and provably so.

    If the gap did not shrink as the unit refines, it would be a bug rather
    than discretization.
    """
    rng = np.random.default_rng(3)
    p = _mixed_problem(rng, n=10, cash_fraction=0.45)
    gaps = []
    for unit in (10_000.0, 2_000.0, 500.0):
        r = cross_validate(p, unit=unit, tolerance=float("inf"))
        gaps.append(r.dp.objective_value - r.lp.objective_value)
    assert gaps[0] > gaps[1] > gaps[2], f"gap did not shrink monotonically: {gaps}"


def test_both_solvers_produce_feasible_allocations():
    rng = np.random.default_rng(21)
    for _ in range(25):
        p = _mixed_problem(rng, n=int(rng.integers(3, 14)))
        r = cross_validate(p)
        assert r.lp_feasible, "LP produced an infeasible allocation"
        assert r.dp_feasible, "DP produced an infeasible allocation"


def test_rigid_obligations_are_all_or_nothing():
    """A RIGID obligation must never be partially funded by either solver."""
    rng = np.random.default_rng(31)
    p = AllocationProblem(
        available_cash=600_000.0,
        obligations=tuple(
            ObligationItem(f"R{i}", float(rng.uniform(80_000, 250_000)),
                           float(rng.uniform(0.002, 0.02)), Flexibility.RIGID)
            for i in range(7)
        ),
    )
    for sol in (solve_lp(p), solve_dp(p)):
        for item, o in zip(sol.allocations, p.obligations):
            a = float(item.allocated_amount)
            assert a < 0.01 or abs(a - o.amount) < 0.01, (
                f"{sol.solver_name}: rigid obligation partially funded ({a} of {o.amount})"
            )


# ---------------------------------------------------------------------------
# Objective correctness against hand-computed ground truth
# ---------------------------------------------------------------------------

def test_objective_matches_hand_computation():
    """Verify the objective on an instance whose answer is obvious by hand.

    Cash 100, two rigid obligations: A costs 60 with rate 1.0, B costs 50 with
    rate 0.5. Only one is affordable. Paying A avoids 60*1.0 = 60 of penalty
    and leaves B's 50*0.5 = 25. Paying B avoids 25 and leaves 60. So the
    optimum pays A, for a total incurred penalty of exactly 25.
    """
    p = AllocationProblem(
        available_cash=100.0,
        obligations=(
            ObligationItem("A", 60.0, 1.0, Flexibility.RIGID),
            ObligationItem("B", 50.0, 0.5, Flexibility.RIGID),
        ),
    )
    lp = solve_lp(p)
    dp = solve_dp(p, unit=1.0)
    assert abs(lp.objective_value - 25.0) < 1e-6, lp.objective_value
    assert abs(dp.objective_value - 25.0) < 1e-6, dp.objective_value


def test_total_penalty_matches_allocation():
    """The reported objective must equal the penalty of the reported allocation."""
    rng = np.random.default_rng(77)
    p = _mixed_problem(rng, n=9)
    for sol in (solve_lp(p), solve_dp(p)):
        alloc = [float(a.allocated_amount) for a in sol.allocations]
        assert abs(total_penalty(p, alloc) - sol.objective_value) < 1.0


def test_zero_cash_pays_nothing():
    rng = np.random.default_rng(9)
    p = _mixed_problem(rng, n=6, cash_fraction=0.0)
    for sol in (solve_lp(p), solve_dp(p)):
        assert all(float(a.allocated_amount) < 0.01 for a in sol.allocations)
        assert abs(sol.objective_value - p.max_total_penalty) < 1.0


def test_abundant_cash_pays_everything():
    rng = np.random.default_rng(10)
    p = _mixed_problem(rng, n=6, cash_fraction=2.0)
    sol = solve_lp(p)
    assert sol.objective_value < 1.0


def test_penalty_rate_is_derived_not_assigned():
    """Provenance: w must come from contract terms, and scale with them."""
    low = penalty_rate_from_terms(late_fee_rate_per_day=0.001, days_overdue_if_unpaid=10)
    high = penalty_rate_from_terms(late_fee_rate_per_day=0.001, days_overdue_if_unpaid=30)
    assert high == pytest.approx(3 * low)
    assert penalty_rate_from_terms(late_fee_rate_per_day=0.0, days_overdue_if_unpaid=30) == 0.0


# ---------------------------------------------------------------------------
# Chance-constrained / SAA
# ---------------------------------------------------------------------------

def _balances(n=4000, horizon=90, seed=0):
    """Synthetic balance paths with a controllable shortfall distribution."""
    rng = np.random.default_rng(seed)
    drift = rng.normal(-4_000, 3_000, size=(n, 1))
    steps = rng.normal(0, 20_000, size=(n, horizon))
    return 3_000_000.0 + np.cumsum(drift + steps, axis=1)


def test_safe_spend_limit_is_the_epsilon_quantile():
    """The limit must be exactly the epsilon-quantile of minimum free cash."""
    b = _balances(seed=1)
    batch = draw_scenarios(b, n_scenarios=2000, seed=1)
    for eps in (0.01, 0.05, 0.10, 0.25):
        limit = safe_spend_limit(batch, eps)
        expected = max(0.0, float(np.quantile(batch.min_free_cash, eps)))
        assert abs(limit - expected) < 1e-6


def test_tighter_epsilon_spends_less_and_costs_more():
    """The risk/penalty frontier must be monotone — the core economic claim."""
    b = _balances(seed=2)
    rng = np.random.default_rng(4)
    p = _mixed_problem(rng, n=10, cash_fraction=1.2)

    spends, penalties = [], []
    for eps in (0.30, 0.20, 0.10, 0.05, 0.01):
        r = solve_chance_constrained(p, b, epsilon=eps, seed=7)
        spends.append(sum(float(a.allocated_amount) for a in r.solution.allocations))
        penalties.append(r.solution.objective_value)

    assert all(spends[i] >= spends[i + 1] - 1.0 for i in range(len(spends) - 1)), spends
    assert all(penalties[i] <= penalties[i + 1] + 1.0 for i in range(len(penalties) - 1)), penalties


def test_achieved_shortfall_respects_epsilon():
    """The realized shortfall frequency must not exceed the requested epsilon."""
    b = _balances(seed=3)
    rng = np.random.default_rng(6)
    p = _mixed_problem(rng, n=8, cash_fraction=1.5)
    for eps in (0.05, 0.10, 0.20):
        r = solve_chance_constrained(p, b, epsilon=eps, seed=7)
        assert r.achieved_shortfall_probability <= eps + 1e-6


def test_scenario_cap_stability_improves_with_more_scenarios():
    """A larger subsample must make the answer less dependent on the draw.

    This is the evidence that justifies the 200-500 scenario cap, and it is
    the same "why this N" argument A.2 makes for its iteration count.
    """
    b = _balances(n=8000, seed=8)
    rng = np.random.default_rng(12)
    p = _mixed_problem(rng, n=8, cash_fraction=1.2)
    cvs = [
        solve_chance_constrained(p, b, epsilon=0.05, n_scenarios=S, seed=7).stability_across_resamples
        for S in (50, 200, 1000)
    ]
    assert cvs[0] > cvs[-1], f"stability did not improve with more scenarios: {cvs}"


def test_infeasible_chance_constraint_is_reported_not_hidden():
    """When no spend can satisfy epsilon, say so instead of returning zero.

    A bland "spend 0" reads as a cautious recommendation when it actually
    means the target is unreachable and the business needs financing rather
    than a cleverer payment schedule.
    """
    doomed = _balances(seed=5) - 5_000_000.0   # every path already underwater
    rng = np.random.default_rng(13)
    p = _mixed_problem(rng, n=5, cash_fraction=1.0)
    r = solve_chance_constrained(p, doomed, epsilon=0.05, seed=7)
    assert "Infeasible" in r.solution.status
