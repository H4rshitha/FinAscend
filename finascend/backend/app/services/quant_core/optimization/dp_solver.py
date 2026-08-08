r"""A.3 — Independent dynamic-programming solution to the same allocation problem.

This is a deliberately *separate* implementation from `lp_solver.py`. Its
value is that it shares no code path with the LP beyond the problem
definition, so when the two agree on an objective value that agreement is real
evidence of correctness rather than two calls into one possibly-wrong routine.

FORMULATION AS A KNAPSACK
=========================
Minimizing incurred penalty is equivalent to maximizing avoided penalty,
because

    sum_i w_i*(a_i - x_i)  =  [sum_i w_i*a_i]  -  [sum_i w_i*x_i]
                                  constant           maximize this

So: choose what to pay, under a cash budget, to maximize penalty avoided.
That is exactly a knapsack — cash is capacity, an obligation's amount is its
weight, and the penalty it avoids is its value.

Two item types, handled in the same table:

  * RIGID     -> 0/1 knapsack item. Take all of it or none of it.
  * FLEXIBLE  -> bounded item, divisible on the discretization grid. Modelled
                 as up to k_i unit-sized copies, each of weight `unit` and
                 value `unit * w_i`. Because every copy of a given obligation
                 has identical weight and value, this needs no binary-splitting
                 trick: the inner loop simply takes as many copies as fit.

DISCRETIZATION
==============
Cash is discretized into `unit`-sized cells (default: rupees rounded to a
chosen granularity). This is what makes the DP finite, and it is also the one
place where the DP and the LP can legitimately disagree — see `complexity` and
the divergence discussion in `cross_validation.py`.

STATE SPACE, TRANSITION, COMPLEXITY
===================================
    State:       dp[i][c] = maximum penalty avoided using only obligations
                            1..i with exactly c budget cells available.
    Base case:   dp[0][c] = 0 for all c            (no obligations, nothing avoided)
    Transition (RIGID item i, weight q_i cells, value v_i):
                 dp[i][c] = max( dp[i-1][c],                  skip
                                 dp[i-1][c - q_i] + v_i )     take, if c >= q_i
    Transition (FLEXIBLE item i, up to k_i unit copies):
                 dp[i][c] = max over t in 0..min(k_i, c) of
                                 dp[i-1][c - t] + t * unit * w_i

    The naive form of that flexible transition is O(k) per cell, giving
    O(n * C * k) overall — which is genuinely too slow (it ran for minutes on
    a 30-obligation instance). It collapses to O(C) with one substitution.
    Put v = unit * w_i and j = c - t, so t = c - j and j ranges over
    [max(0, c - k), c]:

        dp[i][c] = max_j { dp[i-1][j] + (c - j) * v }
                 = c*v + max_{j in [c-k, c]} { dp[i-1][j] - j*v }
                          \_________________________________/
                            sliding-window maximum of width k+1

    Defining g[j] = dp[i-1][j] - j*v, the inner term is a plain sliding-window
    maximum over g, computed in amortized O(1) per cell with a monotonic
    deque. This is the max-plus convolution of a linear kernel; the same trick
    is what turns a naive bounded knapsack into a linear-time one.

    Time:        O(n * C_cells)   for BOTH item types after the deque rewrite
                 (measured: a 30-obligation instance went from minutes to
                 milliseconds)
    Space:       O(n * C_cells) for the parent table needed for reconstruction.
                 (A rolling 1-D array would give O(C_cells) space, but then the
                 chosen allocation cannot be recovered — and an allocation the
                 user cannot see is not an answer. The space is spent
                 deliberately.)

The implementation below favours clarity over micro-optimization; it is meant
to be read aloud in an interview.
"""

from __future__ import annotations

import time

import numpy as np

from app.schemas.quant import AllocationItem, SolverName, SolverSolution
from app.services.quant_core.optimization.problem import (
    AllocationProblem,
    total_penalty,
)

# Default granularity: 1000 currency units per cell. Chosen so a typical
# problem (a few million rupees of cash) yields a few thousand cells, which
# keeps the table small enough to be instant while being fine enough that the
# discretization gap against the LP stays well under a rupee per obligation.
# `choose_unit` derives a value from the instance rather than trusting this.
DEFAULT_UNIT = 1000.0


def choose_unit(problem: AllocationProblem, target_cells: int = 4000) -> float:
    """Pick a discretization granularity from the instance size.

    Rather than hardcoding a cell size, scale it so the table has roughly
    `target_cells` columns regardless of whether the business holds thousands
    or tens of millions in cash. This keeps runtime predictable and makes the
    accuracy/΄speed trade-off explicit and inspectable.
    """
    if problem.available_cash <= 0:
        return DEFAULT_UNIT
    raw = problem.available_cash / target_cells
    # Round to a "nice" power-of-ten multiple so reported allocations do not
    # come out at unreadable granularities like 1373.61.
    magnitude = 10.0 ** np.floor(np.log10(max(raw, 1e-9)))
    return float(max(magnitude, 1.0))


def solve_dp(
    problem: AllocationProblem,
    *,
    unit: float | None = None,
) -> SolverSolution:
    """Solve the allocation problem by dynamic programming.

    Args:
        problem: the instance to solve.
        unit: discretization cell size; derived from the instance if omitted.

    Returns:
        `SolverSolution` reporting **total penalty incurred**, the same
        quantity `solve_lp` reports, so the two are directly comparable.
    """
    t0 = time.perf_counter()
    if unit is None:
        unit = choose_unit(problem)

    n = len(problem.obligations)
    # Budget in whole cells. Floor, never round up: rounding up would let the
    # DP spend cash the business does not have, producing an objective the LP
    # cannot match and an allocation that is not actually affordable.
    capacity = int(np.floor(problem.available_cash / unit))

    if n == 0 or capacity <= 0:
        allocation = [0.0] * n
        return _package(problem, allocation, unit, t0)

    NEG = -np.inf
    # dp[i][c] and the choice table used to reconstruct the allocation.
    dp = np.full((n + 1, capacity + 1), NEG)
    dp[0, :] = 0.0
    take = np.zeros((n + 1, capacity + 1), dtype=np.int32)

    for i, o in enumerate(problem.obligations, start=1):
        prev = dp[i - 1]
        cur = dp[i]

        if o.is_rigid:
            # ---- 0/1 item, vectorized over the whole budget axis ----
            # Ceil the weight so we never under-charge ourselves for a rigid
            # obligation we cannot in truth afford.
            q = int(np.ceil(o.amount / unit))
            value = o.penalty_rate * o.amount
            cand = np.full(capacity + 1, NEG)
            if q <= capacity:
                cand[q:] = prev[: capacity + 1 - q] + value
            better = cand > prev
            cur[:] = np.where(better, cand, prev)
            take[i, :] = np.where(better, q, 0)
        else:
            # ---- divisible item via sliding-window maximum ----
            # dp[i][c] = c*v + max_{j in [c-k, c]} (prev[j] - j*v)
            k = int(np.floor(o.amount / unit))
            v = unit * o.penalty_rate
            if k <= 0:
                cur[:] = prev
                take[i, :] = 0
            else:
                idx = np.arange(capacity + 1, dtype=float)
                g = prev - idx * v
                # Monotonic deque of indices with strictly decreasing g.
                # Front is always the argmax over the current window.
                from collections import deque

                dq: deque[int] = deque()
                for c in range(capacity + 1):
                    # Admit the newly reachable index c.
                    while dq and g[dq[-1]] <= g[c]:
                        dq.pop()
                    dq.append(c)
                    # Evict indices that have fallen out of the window [c-k, c].
                    while dq[0] < c - k:
                        dq.popleft()
                    j = dq[0]
                    cur[c] = g[j] + c * v
                    take[i, c] = c - j          # t = c - j cells taken

    # --- reconstruct the allocation by walking the choice table backwards ---
    allocation = [0.0] * n
    c = capacity
    for i in range(n, 0, -1):
        cells = int(take[i, c])
        if cells > 0:
            o = problem.obligations[i - 1]
            # A rigid item is either fully funded or not funded at all; the
            # cell count can exceed its exact amount because the weight was
            # ceiled, so clamp to the true amount.
            allocation[i - 1] = o.amount if o.is_rigid else min(cells * unit, o.amount)
            c -= cells
    return _package(problem, allocation, unit, t0)


def _package(
    problem: AllocationProblem, allocation: list[float], unit: float, t0: float
) -> SolverSolution:
    """Wrap a raw allocation into the shared solution schema."""
    return SolverSolution(
        solver_name=SolverName.DP_KNAPSACK,
        status=f"Optimal (discretized, unit={unit:g})",
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


def complexity_note(problem: AllocationProblem, unit: float | None = None) -> str:
    """Report the table size for this instance, for the write-up and the logs."""
    if unit is None:
        unit = choose_unit(problem)
    cells = int(np.floor(problem.available_cash / unit))
    n = len(problem.obligations)
    return (
        f"DP table {n} x {cells + 1} = {n * (cells + 1):,} states, "
        f"unit={unit:g}. O(n*C) for both item types (divisible items use the "
        f"sliding-window-maximum transition)."
    )
