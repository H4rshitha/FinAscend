"""Shared problem definition for the allocation solvers.

Both solvers in A.3 must be solving *exactly* the same problem for their
cross-check to mean anything, so the problem lives here rather than being
restated in each solver. If the objective were written twice, an agreement
test would be testing that two transcriptions match, not that two algorithms
agree.

THE PROBLEM
-----------
Given available cash C and a set of obligations i = 1..n, each with

    a_i  amount outstanding
    w_i  penalty rate per rupee left unpaid  (>= 0)
    f_i  flexibility: RIGID forbids partial payment

choose an allocation x_i to

    minimize    sum_i w_i * (a_i - x_i)         [total penalty incurred]
    subject to  sum_i x_i <= C                  [cannot spend what we lack]
                0 <= x_i <= a_i
                x_i in {0, a_i}  if f_i == RIGID

Minimizing incurred penalty is equivalent to maximizing avoided penalty
sum_i w_i * x_i, since sum_i w_i * a_i is a constant. The maximization form is
the one the knapsack DP wants, so both are provided and the relationship is
asserted in the tests rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Flexibility(str, Enum):
    RIGID = "rigid"          # all-or-nothing
    MODERATE = "moderate"
    FLEXIBLE = "flexible"


@dataclass(frozen=True)
class ObligationItem:
    """One obligation as the optimizer sees it."""

    obligation_id: str
    amount: float
    penalty_rate: float      # w_i: penalty per rupee unpaid
    flexibility: Flexibility = Flexibility.MODERATE
    days_until_due: float = 0.0

    @property
    def is_rigid(self) -> bool:
        return self.flexibility is Flexibility.RIGID

    @property
    def max_penalty(self) -> float:
        """Penalty if nothing is paid: w_i * a_i."""
        return self.penalty_rate * self.amount


@dataclass(frozen=True)
class AllocationProblem:
    """A single-period allocation instance."""

    available_cash: float
    obligations: tuple[ObligationItem, ...]

    @property
    def total_amount(self) -> float:
        return sum(o.amount for o in self.obligations)

    @property
    def max_total_penalty(self) -> float:
        """Penalty if nothing at all is paid — the constant offset."""
        return sum(o.max_penalty for o in self.obligations)


def total_penalty(problem: AllocationProblem, allocation: Sequence[float]) -> float:
    """Objective value: total penalty incurred by this allocation.

    Both solvers report this quantity, so the cross-check compares like with
    like regardless of which internal form each solver optimized.
    """
    # max(0, ...) per obligation: an allocation rounded to paise can very
    # slightly exceed the amount owed, producing a negative "penalty" for that
    # item. Summed over a book where everything is affordable, that yields a
    # tiny negative total, and any RELATIVE metric computed from it (regret
    # against a hindsight penalty of ~0) becomes nonsense — it produced a
    # -102% relative regret in an early backtest run. Penalty is non-negative
    # by definition; enforce it here rather than patching consumers.
    return sum(
        o.penalty_rate * max(0.0, o.amount - x)
        for o, x in zip(problem.obligations, allocation)
    )


def is_feasible(
    problem: AllocationProblem,
    allocation: Sequence[float],
    item_tol: float = 0.01,
) -> bool:
    """Check budget, bounds, and the all-or-nothing constraint on RIGID items.

    Used by the cross-validation harness: two solvers agreeing on an objective
    value proves nothing if one of them reached it through an infeasible
    allocation.

    TOLERANCE
    ---------
    `item_tol` defaults to 0.01 — one paisa — because allocations are reported
    as `Money`, a Decimal quantized to two places. An allocation of
    123456.789 is reported as 123456.79, so a rigid obligation can never match
    its own amount to 1e-6 and *every* instance would read as infeasible. That
    is a measurement artifact of the display precision, not a solver error.

    The budget check accumulates that rounding across items, so its tolerance
    is `n * item_tol` rather than a single paisa: n independent roundings can
    each push up by half a paisa in the worst case. Using a flat 1e-6 here was
    the original bug and it made 50+ of 60 random instances report as
    infeasible while the solvers were in fact correct.
    """
    n = max(len(problem.obligations), 1)
    budget_tol = n * item_tol
    if sum(allocation) > problem.available_cash + budget_tol:
        return False
    for o, x in zip(problem.obligations, allocation):
        if x < -item_tol or x > o.amount + item_tol:
            return False
        if o.is_rigid and not (abs(x) <= item_tol or abs(x - o.amount) <= item_tol):
            return False
    return True


def penalty_rate_from_terms(
    *,
    late_fee_rate_per_day: float,
    days_overdue_if_unpaid: float,
    relationship_weight: float = 1.0,
) -> float:
    """Derive w_i from contract terms rather than assigning it by hand.

    w_i = late_fee_rate_per_day * days_overdue_if_unpaid * relationship_weight

    This is the §2.1 provenance rule applied to the optimizer's objective. A
    hand-typed `penalty_severity` would silently determine which obligations
    the solver chooses to pay, which means the "optimization" would really be
    the analyst's prior expressed through a weight vector. Deriving it from
    the late-fee clause makes the objective auditable against the contract.

    `relationship_weight` is the one genuinely subjective input (some suppliers
    matter more than their late fee implies). It is isolated as a single named
    multiplier so it can be inspected, rather than blended invisibly into a
    composite score.
    """
    return max(0.0, late_fee_rate_per_day * days_overdue_if_unpaid * relationship_weight)
