"""Section C — regret: how much the uncertainty actually cost.

regret_t = realized_penalty(live_plan_t) - penalty(hindsight_optimal_plan_t)

Regret is non-negative by construction, since the hindsight plan optimizes
with strictly more information over the same feasible set. A negative regret
would indicate a bug — most likely that the live plan was scored against a
different cash position than the benchmark — so the sign is a self-check and
is asserted rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.schemas.quant import RegretMetrics
from app.services.backtesting.hindsight_optimal import (
    hindsight_plan,
    realized_penalty_of_plan,
)
from app.services.backtesting.replay_harness import ReplayResult
from app.services.quant_core.synthetic_data import SyntheticDataset


@dataclass(frozen=True)
class StepRegret:
    as_of: str
    realized_penalty: float
    hindsight_penalty: float
    regret: float
    planned_spend: float
    realized_cash: float
    over_committed: bool


def compute_regret(
    ds: SyntheticDataset, replay: ReplayResult
) -> tuple[RegretMetrics, list[StepRegret]]:
    """Score every replay step against its perfect-foresight counterpart.

    Returns:
        (aggregate metrics, per-step detail). The per-step detail is returned
        because the aggregate alone hides the interesting structure: regret is
        usually near zero for long stretches and then spikes at the moments
        the forecast missed a turning point, and an average would smooth away
        exactly the events worth writing about.
    """
    rows: list[StepRegret] = []

    for step in replay.steps:
        hind = hindsight_plan(ds, step.problem, step.as_of, replay.horizon_days)
        realized = realized_penalty_of_plan(
            ds, step.problem, step.plan, step.as_of, replay.horizon_days
        )
        planned_spend = sum(float(a.allocated_amount) for a in step.plan.allocations)
        rows.append(
            StepRegret(
                as_of=step.as_of.isoformat(),
                realized_penalty=realized,
                hindsight_penalty=hind.hindsight_penalty,
                regret=realized - hind.hindsight_penalty,
                planned_spend=planned_spend,
                realized_cash=hind.realized_min_cash,
                over_committed=planned_spend > hind.realized_min_cash + 1.0,
            )
        )

    regrets = np.array([r.regret for r in rows], dtype=float)
    total_realized = float(sum(r.realized_penalty for r in rows))
    total_hindsight = float(sum(r.hindsight_penalty for r in rows))

    metrics = RegretMetrics(
        mean_regret=float(regrets.mean()) if regrets.size else 0.0,
        median_regret=float(np.median(regrets)) if regrets.size else 0.0,
        p95_regret=float(np.quantile(regrets, 0.95)) if regrets.size else 0.0,
        total_realized_penalty=total_realized,
        total_hindsight_penalty=total_hindsight,
        relative_regret=(
            (total_realized - total_hindsight) / total_hindsight
            if total_hindsight > 0
            else 0.0
        ),
    )
    return metrics, rows
