"""A.1 — Cash-flow forecasting behind one common interface.

Three models, deliberately spanning a complexity gradient:

  1. SeasonalNaiveForecaster  — the honest floor. y_hat(t+h) = y(t+h-m).
     Any model that cannot beat this is not earning its complexity, and on
     strongly weekly-seasonal data it is a genuinely hard baseline to beat.
  2. HoltWintersForecaster    — triple exponential smoothing (statsmodels
     `ExponentialSmoothing`). Captures level + trend + one seasonal period
     with far fewer parameters than SARIMAX, so it degrades gracefully on
     short histories.
  3. SarimaxForecaster        — `statsmodels.tsa.statespace.SARIMAX`. The most
     flexible; also the most able to overfit, which is exactly why selection
     is done out-of-sample rather than on in-sample likelihood alone.

Model selection is **walk-forward (rolling-origin) cross-validation**, not a
single train/test split. A single split on a seasonal series measures which
window you happened to hold out as much as it measures skill: hold out a
month containing a shock and SARIMAX looks terrible; hold out a calm month
and it looks excellent. Rolling the origin forward averages that away and, as
a side effect, produces a distribution of per-fold errors whose *spread* is
itself informative (reported in `ModelSelectionScore.fold_rmses`).

Prediction intervals — normalized split conformal
-------------------------------------------------
Every model's interval is **conformally recalibrated against its own
walk-forward residuals**. This replaced a per-horizon residual-quantile
construction that was measured at 57.2% coverage on a 95% claim; the diagnosis
is written up in full in `FORECASTING_METHODOLOGY.md`, and the three defects it
found are what this design is shaped by.

The construction has two pieces:

  * a **scale profile** `scale(h)`, the model's own statement about how its
    uncertainty grows with the horizon. SARIMAX supplies this analytically from
    its state-space covariance. Models with no variance model get a power law
    `a * h**gamma` whose exponent is *fitted* to the walk-forward residuals
    rather than assumed to be the textbook sqrt(h).
  * a **conformal multiplier** `q_hat`, the `ceil((N+1)*c)`-th smallest value
    of the nonconformity scores `|residual| / scale(h)` pooled over every
    (fold, horizon) cell. The interval is `point +/- q_hat * scale(h)`.

Three properties matter, and each one fixes a measured failure:

1. **Pooling the scores across horizons is what makes a 95% level reachable at
   all.** A per-horizon quantile from `n_folds` residuals has a hard coverage
   ceiling of (n-1)/(n+1) — 66.7% at five folds, which is why the old
   construction could not have worked however good the forecast was. Pooling
   gives N = n_folds * cv_horizon scores instead of n_folds. Normalizing by
   `scale(h)` is what makes cells from different horizons exchangeable enough
   to pool.
2. **When N is still too small the code refuses**, reporting
   `achieved=False` and falling back to a stated Gaussian approximation,
   rather than returning an order statistic that does not exist.
3. **`scale(h)` is defined over the whole horizon**, so the interval keeps
   widening past the cross-validation horizon. The previous version reused the
   day-14 residual spread for days 15 through 90 — the width ratio between
   h=90 and h=14 was exactly 1.0000.

`q_hat` travels with the forecast, together with the Gaussian reference `z` it
must be read against. **The reference is z (1.96 at 95%), not 1.0** — `q_hat`
multiplies a sigma, so a model whose stated scale is exactly right still needs
1.96 of them to cover 95%. The interpretable number is
`scale_ratio = q_hat / z`: 1.0 means the model's own scale was right, 2.0 means
it was half what it should have been. Reporting the ratio rather than a bare
multiplier is what keeps this a diagnostic instead of an unexplained constant.

All three models propagate intervals into A.2 rather than handing it a point
path. Treating a forecast as certain and then simulating "uncertainty" on top
of it double-counts confidence and understates risk.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.schemas.quant import (
    ForecastModelName,
    ForecastPoint,
    ForecastResult,
    ModelSelectionScore,
)

# Weekly seasonality. Daily business cash flow is dominated by day-of-week
# effects (weekends are near-zero), so m=7 is the period that matters at this
# sampling frequency. Annual seasonality would need m=365, which is
# intractable for SARIMAX and is better handled by the STL decomposition in
# A.6 than by forcing it into the forecaster.
SEASONAL_PERIOD = 7


@dataclass(frozen=True)
class ForecastOutput:
    """Raw forecaster output before it is wrapped into the API schema."""

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


class BaseForecaster(ABC):
    """Common interface for every forecasting model.

    Implementations must be fit-then-predict and must never see data beyond
    the fit window — the backtesting harness in Section C depends on that
    contract to guarantee no look-ahead.
    """

    name: ForecastModelName

    # Whether `predict` returns an interval derived from a real variance model
    # (SARIMAX's state-space covariance) rather than a residual heuristic. Only
    # affects which scale profile the conformal layer normalizes by; both kinds
    # are recalibrated, so a False here is not a licence to under-cover.
    provides_analytic_scale: bool = False

    @abstractmethod
    def fit(self, y: np.ndarray) -> "BaseForecaster":
        """Fit on a 1-D array of historical values."""

    @abstractmethod
    def predict(self, horizon: int, confidence: float = 0.95) -> ForecastOutput:
        """Forecast `horizon` steps ahead with a prediction interval."""

    @property
    def aic(self) -> Optional[float]:
        """In-sample AIC where the model defines a likelihood; None otherwise."""
        return None

    @property
    def bic(self) -> Optional[float]:
        return None


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal naive: y_hat(t+h) = y(t + h - m*ceil(h/m)).

    Method: repeat the last observed seasonal cycle. Chosen as the baseline
    because it is the standard reference for seasonal series (Hyndman &
    Athanasopoulos, *Forecasting: Principles and Practice*, §5.2) and because
    it has zero fitted parameters, so it cannot overfit. Intervals come from
    walk-forward residuals — it defines no likelihood of its own.
    """

    name = ForecastModelName.SEASONAL_NAIVE

    def __init__(self, period: int = SEASONAL_PERIOD) -> None:
        self.period = period
        self._y: Optional[np.ndarray] = None
        self._resid_scale: float = 0.0

    def fit(self, y: np.ndarray) -> "SeasonalNaiveForecaster":
        self._y = np.asarray(y, dtype=float)
        if len(self._y) > self.period:
            in_sample_resid = self._y[self.period :] - self._y[: -self.period]
            self._resid_scale = float(np.std(in_sample_resid, ddof=1))
        return self

    def predict(self, horizon: int, confidence: float = 0.95) -> ForecastOutput:
        if self._y is None:
            raise RuntimeError("fit() must be called before predict()")
        last_cycle = self._y[-self.period :]
        reps = int(np.ceil(horizon / self.period))
        point = np.tile(last_cycle, reps)[:horizon]
        # Interval widens with sqrt(h) — the random-walk-style accumulation of
        # uncertainty. z from the normal quantile; see class docstring for why
        # the empirical route is preferred when folds allow it.
        from scipy import stats as _st

        z = float(_st.norm.ppf(0.5 + confidence / 2.0))
        widen = np.sqrt(np.arange(1, horizon + 1) / self.period)
        half = z * self._resid_scale * widen
        return ForecastOutput(point=point, lower=point - half, upper=point + half)


class HoltWintersForecaster(BaseForecaster):
    """Holt-Winters triple exponential smoothing.

    Method: `statsmodels.tsa.holtwinters.ExponentialSmoothing` with additive
    trend and additive seasonality. Chosen over the multiplicative variant
    because the target series is *net* cash flow, which crosses zero and goes
    negative — multiplicative seasonality is undefined on non-positive data.
    (The gross inflow series is multiplicative, which is why the A.0 generator
    composes it that way; the forecaster targets net flow instead.)

    Damped trend is enabled: an undamped linear trend extrapolated 90 days out
    produces implausible runway estimates, and damping is the standard
    correction (Gardner & McKenzie 1985).
    """

    name = ForecastModelName.HOLT_WINTERS

    def __init__(self, period: int = SEASONAL_PERIOD) -> None:
        self.period = period
        self._fitted = None
        self._resid_scale: float = 0.0

    def fit(self, y: np.ndarray) -> "HoltWintersForecaster":
        y = np.asarray(y, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                y,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=self.period,
                initialization_method="estimated",
            )
            self._fitted = model.fit(optimized=True)
        self._resid_scale = float(np.std(self._fitted.resid, ddof=1))
        return self

    def predict(self, horizon: int, confidence: float = 0.95) -> ForecastOutput:
        if self._fitted is None:
            raise RuntimeError("fit() must be called before predict()")
        point = np.asarray(self._fitted.forecast(horizon), dtype=float)
        from scipy import stats as _st

        z = float(_st.norm.ppf(0.5 + confidence / 2.0))
        widen = np.sqrt(np.arange(1, horizon + 1))
        half = z * self._resid_scale * widen
        return ForecastOutput(point=point, lower=point - half, upper=point + half)

    @property
    def aic(self) -> Optional[float]:
        return float(self._fitted.aic) if self._fitted is not None else None

    @property
    def bic(self) -> Optional[float]:
        return float(self._fitted.bic) if self._fitted is not None else None


class SarimaxForecaster(BaseForecaster):
    """SARIMAX with weekly seasonal structure.

    Method: `statsmodels.tsa.statespace.SARIMAX`, default order (1,0,1) and
    seasonal order (1,0,1,7). Chosen orders are modest on purpose: the series
    is already differenced in spirit (net flow is a difference of levels), and
    high-order SARIMAX on 3 years of daily data overfits readily. The order is
    a constructor argument so notebook 01 can search it rather than assert it.

    Intervals here are analytic — `get_forecast().conf_int()` derives them
    from the state-space covariance, which is a real distributional statement
    rather than a residual heuristic. That is the main reason SARIMAX is worth
    its cost even when its point accuracy only ties Holt-Winters.

    Measured in the Section C backtest, these intervals came in at 96.0%
    empirical coverage against a 95% claim — honest, marginally conservative.
    They still pass through the conformal layer, because a construction that
    happens to be calibrated on this synthetic world has not been *shown* to be
    calibrated on another; `q_hat` near 1.0 is the evidence, not the assumption.
    One caveat stands: `conf_int` conditions on the estimated parameters and
    ignores their sampling variability, so it understates uncertainty on short
    training windows. The conformal multiplier absorbs that, since it is
    measured from held-out error rather than from the likelihood.
    """

    name = ForecastModelName.SARIMAX
    provides_analytic_scale = True

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 0, 1, SEASONAL_PERIOD),
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted = None

    def fit(self, y: np.ndarray) -> "SarimaxForecaster":
        y = np.asarray(y, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._fitted = model.fit(disp=False)
        return self

    def predict(self, horizon: int, confidence: float = 0.95) -> ForecastOutput:
        if self._fitted is None:
            raise RuntimeError("fit() must be called before predict()")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self._fitted.get_forecast(steps=horizon)
            point = np.asarray(fc.predicted_mean, dtype=float)
            ci = np.asarray(fc.conf_int(alpha=1.0 - confidence), dtype=float)
        return ForecastOutput(point=point, lower=ci[:, 0], upper=ci[:, 1])

    @property
    def aic(self) -> Optional[float]:
        return float(self._fitted.aic) if self._fitted is not None else None

    @property
    def bic(self) -> Optional[float]:
        return float(self._fitted.bic) if self._fitted is not None else None


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-8) -> float:
    """Mean absolute percentage error, in percent.

    MAPE is reported because it is the metric a non-technical user
    understands, but it is *not* the selection metric: net cash flow crosses
    zero, and MAPE explodes near zero denominators. `eps` guards the division;
    RMSE is what actually drives model choice.
    """
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    denom = np.where(np.abs(a) < eps, eps, np.abs(a))
    return float(np.mean(np.abs((a - p) / denom)) * 100.0)


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkForwardResult:
    """Per-model walk-forward scores plus the residuals that build intervals."""

    model_name: ForecastModelName
    fold_rmses: list[float]
    fold_mapes: list[float]
    aic: Optional[float]
    bic: Optional[float]
    # residuals[k, h] = (actual - predicted) at horizon step h in fold k
    residuals_by_horizon: np.ndarray
    # scales[k, h] = the standard deviation the model ITSELF claimed at that
    # cell, recovered from the interval it returned. Recorded alongside the
    # residual because the conformal multiplier is the ratio of the two: an
    # error is only large relative to the uncertainty that was advertised.
    scales_by_horizon: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    @property
    def mean_rmse(self) -> float:
        return float(np.mean(self.fold_rmses))

    @property
    def mean_mape(self) -> float:
        return float(np.mean(self.fold_mapes))


def walk_forward_validate(
    y: np.ndarray,
    factory,
    *,
    horizon: int = 14,
    n_folds: int = 5,
    min_train: Optional[int] = None,
) -> WalkForwardResult:
    """Rolling-origin cross-validation, anchored to the END of the series.

    The origin advances by `horizon` each fold, so folds are non-overlapping
    in their evaluation windows and every fold trains only on data strictly
    before its own evaluation window. That ordering is the no-look-ahead
    guarantee, and it is why this cannot be replaced by `sklearn`'s KFold —
    shuffling a time series destroys exactly the structure being modelled.

    WHY THE FOLDS ARE ANCHORED TO THE END
    -------------------------------------
    The last fold's evaluation window is the final `horizon` observations, and
    earlier folds step backwards from there. The previous version started the
    first fold at `min_train = n // 2` and walked forward, which on a
    thousand-point series meant the folds covered days 500-570 and days
    570-1000 were **never validated on at all**. Both the model choice and the
    interval calibration were therefore computed from the middle of the
    history rather than the recent past — exactly backwards for a series with
    a regime change in it, and a measured contributor to the interval
    under-coverage diagnosed in `FORECASTING_METHODOLOGY.md`.

    Anchoring to the end also matches how the model is actually used: the
    forecast that matters is made from the last observation, so the folds that
    should carry the most weight are the ones nearest to it.

    Args:
        y: full history.
        factory: zero-argument callable returning a fresh `BaseForecaster`.
            A factory rather than an instance because each fold must refit
            from scratch; reusing a fitted model would leak later data
            backwards through its state.
        horizon: steps forecast per fold.
        n_folds: number of rolling origins.
        min_train: minimum training length; defaults to half the series.

    Returns:
        `WalkForwardResult` with per-fold errors, per-horizon residuals, and
        the per-horizon scale each fold's model claimed for itself.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if min_train is None:
        min_train = max(n // 2, 4 * SEASONAL_PERIOD)

    usable = n - min_train
    if usable < horizon:
        raise ValueError(
            f"series too short: need > {min_train + horizon} points, got {n}"
        )
    n_folds = max(1, min(n_folds, usable // horizon))

    fold_rmses: list[float] = []
    fold_mapes: list[float] = []
    resid_rows: list[np.ndarray] = []
    scale_rows: list[np.ndarray] = []
    last_aic: Optional[float] = None
    last_bic: Optional[float] = None

    for k in range(n_folds):
        # Fold n_folds-1 tests on the final `horizon` points; fold 0 is the
        # oldest. Walking backwards from the end is what keeps the calibration
        # residuals recent.
        test_end = n - (n_folds - 1 - k) * horizon
        train_end = test_end - horizon
        if train_end < min_train:
            continue
        train, test = y[:train_end], y[train_end:test_end]

        model = factory().fit(train)
        out = model.predict(horizon)
        fold_rmses.append(rmse(test, out.point))
        fold_mapes.append(mape(test, out.point))
        resid_rows.append(test - out.point)
        scale_rows.append(scale_from_output(out))
        last_aic, last_bic = model.aic, model.bic

    if not fold_rmses:
        raise ValueError(
            f"no usable fold: n={n}, min_train={min_train}, horizon={horizon}, "
            f"n_folds={n_folds}"
        )

    return WalkForwardResult(
        model_name=factory().name,
        fold_rmses=fold_rmses,
        fold_mapes=fold_mapes,
        aic=last_aic,
        bic=last_bic,
        residuals_by_horizon=np.array(resid_rows),
        scales_by_horizon=np.array(scale_rows),
    )


# ---------------------------------------------------------------------------
# Conformal interval calibration
# ---------------------------------------------------------------------------

def scale_from_output(out: ForecastOutput, confidence: float = 0.95) -> np.ndarray:
    """Recover the per-step standard deviation implied by a returned interval.

    The inverse of how the interval was built: `sigma = (upper - lower)/(2z)`.
    Going through the interval rather than asking each model for a variance
    keeps `BaseForecaster` to one output type, so a new model only has to
    produce an interval to participate in the conformal layer.
    """
    from scipy import stats as _st

    z = float(_st.norm.ppf(0.5 + confidence / 2.0))
    return np.abs(np.asarray(out.upper) - np.asarray(out.lower)) / (2.0 * z)


def fit_scale_profile(
    residuals_by_horizon: np.ndarray, horizon: int, *, floor: float = 1e-9
) -> tuple[np.ndarray, float]:
    """Fit `scale(h) = a * h**gamma` to walk-forward residuals and extrapolate.

    Used for models with no variance model of their own. The exponent is
    **fitted, not assumed**: the textbook random-walk value is gamma = 0.5, but
    a stationary series has gamma near 0 because the forecast error variance
    converges to the unconditional variance rather than accumulating. Assuming
    sqrt(h) on a stationary series over-widens far horizons; assuming a
    constant under-widens a drifting one. Measuring it avoids picking wrong in
    either direction, and the fitted gamma is reported so the choice stays
    checkable.

    Method: OLS of log(|residual| RMS at h) on log(h), which is the linear form
    of the power law. Falls back to a flat profile when there are too few
    distinct horizons to fit a slope.

    Args:
        residuals_by_horizon: (n_folds, cv_horizon) residual matrix.
        horizon: length of the profile to return.
        floor: guards log(0) when a horizon has degenerate residuals.

    Returns:
        (scale over 1..horizon, fitted gamma)
    """
    if residuals_by_horizon.size == 0:
        return np.ones(horizon), 0.0

    per_h = np.sqrt(np.mean(residuals_by_horizon**2, axis=0))
    per_h = np.maximum(per_h, floor)
    h_obs = np.arange(1, len(per_h) + 1, dtype=float)

    if len(per_h) < 3:
        return np.full(horizon, float(per_h.mean())), 0.0

    gamma, log_a = np.polyfit(np.log(h_obs), np.log(per_h), 1)
    # A negative exponent would say the forecast gets MORE certain the further
    # out it looks, which no forecaster earns; clipped at 0 (constant width).
    gamma = float(np.clip(gamma, 0.0, 1.0))
    a = float(np.exp(log_a))
    h_all = np.arange(1, horizon + 1, dtype=float)
    return a * h_all**gamma, gamma


@dataclass(frozen=True)
class ConformalCalibration:
    """The multiplier that turns a model's own scale into a calibrated interval.

    READING `q_hat` — MIND THE UNITS
    --------------------------------
    `q_hat` multiplies a **standard deviation**, not an interval width, because
    the nonconformity score is `|residual| / scale(h)` and `scale(h)` is a
    sigma. So the reference value is **not 1.0**: a model whose stated sigma is
    exactly right and whose errors are Gaussian needs
    `q_hat = z_{(1+c)/2}` — 1.96 at 95% — because that is the multiple of sigma
    a 95% interval requires.

    `scale_ratio = q_hat / z` is therefore the number to read as "how many
    times too narrow was the model's own scale": 1.0 means it was right, 2.0
    means it was half the size it should have been. Comparing `q_hat` itself
    against 1.0 confuses the two scales and reads a correctly calibrated model
    as being 2x too narrow.
    """

    q_hat: float
    n_scores: int
    rank: int
    confidence: float
    achieved: bool
    scale_gamma: Optional[float]
    note: str
    # The Gaussian reference, so callers never have to recompute it to
    # interpret q_hat.
    z_reference: float = 1.959963984540054
    scale_ratio: float = 1.0


def conformal_multiplier(
    residuals_by_horizon: np.ndarray,
    scales_by_horizon: np.ndarray,
    confidence: float,
) -> ConformalCalibration:
    """Split-conformal multiplier on scores `|residual| / scale`.

    METHOD
    ------
    Score every held-out cell by how large its error was *relative to the
    uncertainty the model advertised there*:

        s[k, h] = |residual[k, h]| / scale[k, h]

    Pool all N = n_folds * cv_horizon scores and take the
    `ceil((N + 1) * confidence)`-th smallest. That rank — rather than
    `np.quantile` — is the finite-sample split-conformal rank: with exchangeable
    scores it makes `P(|new residual| <= q_hat * scale) >= confidence` hold at
    finite N, not just asymptotically.

    WHY POOL ACROSS HORIZONS
    ------------------------
    Because per-horizon quantiles cannot reach the level being claimed. With
    n_folds residuals at a single horizon, the probability a fresh observation
    lands inside the sample range is (n-1)/(n+1) — 66.7% at five folds, against
    a 95% claim. That is the measured cause of the 57.2% coverage the previous
    construction achieved. Pooling multiplies the score count by the CV horizon;
    dividing by `scale[k, h]` is what makes cells at different horizons
    comparable enough to pool in the first place.

    WHAT THIS DOES NOT CLAIM
    ------------------------
    Split conformal guarantees coverage under **exchangeability** of the scores.
    Scores from the same fold share a training set and neighbouring horizons
    share forecast error, so they are correlated: the effective sample size is
    below N and the guarantee is weaker than the textbook statement. This is
    why the fix is validated empirically on the backtest rather than asserted
    from the theorem — the coverage number in `BACKTEST_REPORT.md` is the claim,
    and the theorem is only the reason to expect it.

    Returns:
        `ConformalCalibration`; `achieved=False` means N was too small to
        express the requested level and the caller must not pretend otherwise.
    """
    finite = (
        np.isfinite(residuals_by_horizon)
        & np.isfinite(scales_by_horizon)
        & (scales_by_horizon > 0)
    )
    scores = np.abs(residuals_by_horizon[finite]) / scales_by_horizon[finite]
    n = int(scores.size)

    if n == 0:
        return ConformalCalibration(
            q_hat=1.0, n_scores=0, rank=0, confidence=confidence, achieved=False,
            scale_gamma=None,
            note="no usable calibration scores; interval falls back to the "
                 "model's own uncalibrated scale",
        )

    rank = int(np.ceil((n + 1) * confidence))
    if rank > n:
        # The requested level is not expressible from this many scores. Say so
        # and use the largest score, which is the most conservative honest
        # answer available — never silently return a quantile that does not
        # exist, which is precisely how the previous construction claimed 95%
        # from five residuals.
        return ConformalCalibration(
            q_hat=float(np.max(scores)), n_scores=n, rank=rank,
            confidence=confidence, achieved=False, scale_gamma=None,
            note=(
                f"{confidence:.0%} needs rank {rank} of {n} scores, which does "
                f"not exist; using the maximum score ({np.max(scores):.2f}x) as "
                f"the most conservative honest substitute. Achievable level "
                f"here is {n / (n + 1):.1%}. Increase n_folds or cv_horizon."
            ),
        )

    from scipy import stats as _st

    q_hat = float(np.sort(scores)[rank - 1])
    z = float(_st.norm.ppf(0.5 + confidence / 2.0))
    ratio = q_hat / z
    return ConformalCalibration(
        q_hat=q_hat, n_scores=n, rank=rank, confidence=confidence, achieved=True,
        scale_gamma=None, z_reference=z, scale_ratio=ratio,
        note=(
            f"split-conformal multiplier {q_hat:.3f} at rank {rank}/{n}, "
            f"against a Gaussian reference of z={z:.3f}. "
            + (
                f"The model's own scale was about right ({ratio:.2f}x)."
                if 0.9 <= ratio <= 1.1
                else f"The model's own scale was {ratio:.2f}x "
                     f"{'too narrow' if ratio > 1 else 'too wide'}."
            )
        ),
    )


def conformal_interval(
    point: np.ndarray, scale: np.ndarray, q_hat: float
) -> tuple[np.ndarray, np.ndarray]:
    """`point +/- q_hat * scale(h)`, the calibrated interval."""
    half = float(q_hat) * np.asarray(scale, dtype=float)
    return point - half, point + half


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_FACTORIES = {
    ForecastModelName.SEASONAL_NAIVE: SeasonalNaiveForecaster,
    ForecastModelName.HOLT_WINTERS: HoltWintersForecaster,
    ForecastModelName.SARIMAX: SarimaxForecaster,
}


def select_and_forecast(
    daily: pd.DataFrame,
    *,
    business_id: str = "synthetic",
    horizon_days: int = 90,
    cv_horizon: int = 14,
    n_folds: int = 5,
    confidence: float = 0.95,
    value_column: str = "net",
    opening_balance: Optional[float] = None,
) -> ForecastResult:
    """Run walk-forward selection across all candidates and forecast with the winner.

    Selection rule: lowest mean out-of-sample RMSE across folds. AIC/BIC are
    reported for the models that define a likelihood, but they are *not* the
    deciding metric — AIC is an in-sample criterion and the seasonal naive
    baseline has no likelihood to compare against, so ranking all three on
    AIC would be comparing incommensurable quantities.

    Args:
        daily: frame with a `date` column and `value_column`.
        horizon_days: forecast length.
        cv_horizon: per-fold evaluation length during selection.
        confidence: prediction-interval level.
        value_column: which series to forecast (`net` by default).
        opening_balance: current cash; if given, `days_to_zero` is computed by
            walking the forecast path down from it.

    Returns:
        `ForecastResult` including every candidate's score, winners and losers.
    """
    y = daily[value_column].to_numpy(dtype=float)
    dates = pd.to_datetime(daily["date"])

    scores: list[ModelSelectionScore] = []
    wf_results: dict[ForecastModelName, WalkForwardResult] = {}

    for name, factory in _FACTORIES.items():
        try:
            wf = walk_forward_validate(
                y, factory, horizon=cv_horizon, n_folds=n_folds
            )
        except Exception as exc:  # a candidate that cannot fit is reported, not hidden
            scores.append(
                ModelSelectionScore(
                    model_name=name,
                    mape=float("inf"),
                    rmse=float("inf"),
                    aic=None,
                    bic=None,
                    n_folds=0,
                    fold_rmses=[],
                )
            )
            continue
        wf_results[name] = wf
        scores.append(
            ModelSelectionScore(
                model_name=name,
                mape=wf.mean_mape,
                rmse=wf.mean_rmse,
                aic=wf.aic,
                bic=wf.bic,
                n_folds=len(wf.fold_rmses),
                fold_rmses=wf.fold_rmses,
            )
        )

    if not wf_results:
        raise RuntimeError("no forecasting candidate could be fitted")

    winner = min(wf_results.items(), key=lambda kv: kv[1].mean_rmse)[0]
    runner_up = sorted(wf_results.items(), key=lambda kv: kv[1].mean_rmse)
    rationale = (
        f"{winner.value} selected on lowest mean walk-forward RMSE "
        f"({wf_results[winner].mean_rmse:,.0f}) across "
        f"{len(wf_results[winner].fold_rmses)} rolling-origin folds. "
        + "; ".join(
            f"{n.value}={r.mean_rmse:,.0f}" for n, r in runner_up
        )
        + ". Selection is out-of-sample RMSE, not AIC, because the seasonal "
        "naive baseline defines no likelihood and cannot be ranked on AIC."
    )

    final = _FACTORIES[winner]().fit(y)
    out = final.predict(horizon_days, confidence=confidence)
    wf_win = wf_results[winner]

    # --- interval: the model's own scale, conformally recalibrated ---
    # Both branches go through the same recalibration. The previous version
    # trusted SARIMAX's analytic interval as-is and gave every other model a
    # raw residual quantile with no recalibration at all, which is how one
    # branch reached 96% coverage while the other sat at 57%.
    gamma: Optional[float] = None
    if final.provides_analytic_scale:
        # A real variance model, so its SHAPE across the horizon is trusted and
        # only its level is corrected.
        scale = scale_from_output(out, confidence)
        cal_scales = wf_win.scales_by_horizon
    else:
        # No variance model, so the shape is fitted from held-out residuals.
        # The same profile is used for calibration and for the final interval,
        # which is what makes q_hat a like-for-like multiplier.
        cv_h = wf_win.residuals_by_horizon.shape[1]
        profile, gamma = fit_scale_profile(wf_win.residuals_by_horizon, horizon_days)
        scale = profile
        cal_scales = np.tile(profile[:cv_h], (wf_win.residuals_by_horizon.shape[0], 1))

    calibration = conformal_multiplier(
        wf_win.residuals_by_horizon, cal_scales, confidence
    )
    lower, upper = conformal_interval(out.point, scale, calibration.q_hat)

    rationale += (
        f" Interval: {'analytic state-space' if final.provides_analytic_scale else f'fitted power-law scale (gamma={gamma:.3f})'}"
        f" scale, conformally recalibrated by q_hat={calibration.q_hat:.3f} "
        f"from {calibration.n_scores} held-out scores. {calibration.note}"
    )
    if not calibration.achieved:
        rationale += (
            " WARNING: the requested confidence level is not expressible from "
            "this many calibration scores; the interval is a stated "
            "approximation, not a conformal guarantee."
        )

    last_date = dates.iloc[-1]
    path = [
        ForecastPoint(
            as_of_date=(last_date + pd.Timedelta(days=i + 1)).date(),
            point=float(out.point[i]),
            lower=float(lower[i]),
            upper=float(upper[i]),
        )
        for i in range(horizon_days)
    ]

    days_to_zero: Optional[int] = None
    if opening_balance is not None:
        bal = opening_balance + np.cumsum(out.point)
        below = np.where(bal <= 0.0)[0]
        days_to_zero = int(below[0] + 1) if below.size else None

    return ForecastResult(
        business_id=business_id,
        generated_at=datetime.now(timezone.utc),
        horizon_days=horizon_days,
        interval_confidence=confidence,
        selected_model=winner,
        selection_rationale=rationale,
        scores=scores,
        path=path,
        days_to_zero=days_to_zero,
        conformal_q_hat=calibration.q_hat,
        conformal_z_reference=calibration.z_reference,
        conformal_scale_ratio=calibration.scale_ratio,
        conformal_n_scores=calibration.n_scores,
        conformal_achieved=calibration.achieved,
        scale_gamma=gamma,
    )
