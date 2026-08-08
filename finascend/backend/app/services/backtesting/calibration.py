"""Section C — are the prediction intervals honest?

A 95% prediction interval claims that ~95% of realized outcomes fall inside
it. This module checks that claim against what actually happened.

WHY THIS MATTERS MORE THAN POINT ACCURACY
------------------------------------------
A model with tight, wrong intervals is more dangerous than one with wide,
honest intervals. The whole liquidity engine consumes interval WIDTH as its
measure of uncertainty — A.2 converts it to a standard deviation and
propagates it into Runway-at-Risk. If the intervals are too narrow, RaR is
overconfident and the business is told it has more runway than it does, which
is the exact failure this product exists to prevent.

Only this check distinguishes the two cases. RMSE cannot: a model can have
excellent point accuracy and badly calibrated intervals simultaneously.

READING THE RESULT
------------------
  coverage ~= nominal   intervals are honest
  coverage <  nominal   OVERCONFIDENT — intervals too narrow, the dangerous
                        direction, RaR will be optimistic
  coverage >  nominal   underconfident — intervals too wide, wasteful but safe
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.schemas.quant import CalibrationResult
from app.services.backtesting.replay_harness import ReplayResult
from app.services.quant_core.synthetic_data import SyntheticDataset


@dataclass(frozen=True)
class HorizonCalibration:
    """Coverage measured over one band of forecast horizons.

    BANDS, NOT SINGLE HORIZONS — AND WHY IT CHANGED
    -----------------------------------------------
    This used to report six single horizons: 1, 7, 14, 30, 60, 90. That
    sampling is **aliased**, and the aliasing flattered the result. The replay
    advances `as_of` by a whole number of weeks, so a given horizon h lands on
    the same day of the week at every single step; and coverage genuinely
    varies by weekday, because the generator's flows are near-zero at weekends
    and volatile midweek. The six checkpoints therefore sampled four of the
    seven weekdays and reported 89.5% coverage while the true pooled figure
    over all 90 horizons was 85.9%.

    Bands aggregate every horizon in a range and cannot alias against the step
    spacing, so the breakdown now measures horizon structure rather than the
    interaction between the checkpoint set and the calendar.
    """

    horizon_from: int
    horizon_to: int
    n: int
    coverage: float
    mean_width: float

    @property
    def label(self) -> str:
        return (
            f"{self.horizon_from}"
            if self.horizon_from == self.horizon_to
            else f"{self.horizon_from}–{self.horizon_to}"
        )


def assess_calibration(
    ds: SyntheticDataset,
    replay: ReplayResult,
    *,
    nominal: float = 0.95,
    horizon_buckets: tuple[tuple[int, int], ...] = (
        (1, 7), (8, 14), (15, 30), (31, 45), (46, 60), (61, 75), (76, 90),
    ),
    value_column: str = "net_ex_receipts",
) -> tuple[CalibrationResult, list[HorizonCalibration]]:
    """Measure empirical interval coverage against the nominal level.

    Coverage is reported per horizon band as well as pooled, because they
    routinely differ in an informative way: intervals are often well
    calibrated one day ahead and badly calibrated ninety days ahead, and a
    single pooled number hides that. A forecaster whose 90-day interval covers
    60% of outcomes is unusable for runway even if its pooled coverage looks
    respectable.

    Every horizon from 1 to the forecast length contributes to exactly one
    band, so the pooled figure is the weighted average of the bands and the
    two cannot disagree — which is what went wrong with the previous
    single-horizon checkpoints (see `HorizonCalibration`).

    Args:
        ds: the world, used to look up what actually happened.
        replay: the recorded decision points.
        nominal: the claimed level, e.g. 0.95.
        horizon_buckets: inclusive (from, to) day ranges to report separately.
        value_column: the series that was forecast.

    Returns:
        (pooled result, per-band breakdown)
    """
    actual = ds.daily.set_index("date")[value_column]

    hits: list[bool] = []
    widths: list[float] = []
    per_band: dict[tuple[int, int], list[tuple[bool, float]]] = {
        b: [] for b in horizon_buckets
    }

    for step in replay.steps:
        base = pd.Timestamp(step.as_of)
        n_ahead = len(step.forecast_path)
        for h in range(n_ahead):
            target_date = base + pd.Timedelta(days=h + 1)
            if target_date not in actual.index:
                continue
            y = float(actual.loc[target_date])
            lo, hi = float(step.forecast_lower[h]), float(step.forecast_upper[h])
            inside = lo <= y <= hi
            hits.append(inside)
            widths.append(hi - lo)
            for band in horizon_buckets:
                if band[0] <= h + 1 <= band[1]:
                    per_band[band].append((inside, hi - lo))
                    break

    coverage = float(np.mean(hits)) if hits else 0.0
    mean_width = float(np.mean(widths)) if widths else 0.0

    gap = coverage - nominal
    if abs(gap) <= 0.05:
        verdict = (
            f"Intervals are well calibrated: {coverage:.1%} empirical coverage "
            f"against a {nominal:.0%} nominal level."
        )
    elif gap < 0:
        verdict = (
            f"OVERCONFIDENT: only {coverage:.1%} of outcomes fell inside the "
            f"{nominal:.0%} interval. The intervals are too narrow, so "
            "Runway-at-Risk derived from them is optimistic — the dangerous "
            "direction for a liquidity warning."
        )
    else:
        verdict = (
            f"Underconfident: {coverage:.1%} coverage against a {nominal:.0%} "
            "nominal level. Intervals are wider than they need to be, which "
            "wastes decision room but does not understate risk."
        )

    pooled = CalibrationResult(
        nominal_coverage=nominal,
        empirical_coverage=coverage,
        n_observations=len(hits),
        mean_interval_width=mean_width,
        verdict=verdict,
    )

    breakdown = [
        HorizonCalibration(
            horizon_from=band[0],
            horizon_to=band[1],
            n=len(v),
            coverage=float(np.mean([x[0] for x in v])) if v else 0.0,
            mean_width=float(np.mean([x[1] for x in v])) if v else 0.0,
        )
        for band, v in sorted(per_band.items())
        if v
    ]
    return pooled, breakdown
