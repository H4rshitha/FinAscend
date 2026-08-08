# Forecasting Methodology (A.1)

A short research memo: what was tried, what was rejected, and why. The
rejections matter as much as the selection.

---

## What is being forecast, and why it is not the obvious series

The target is **`net_ex_receipts`** = cash sales − outflow, *not* net cash flow.

This is the single most consequential choice in the module and it is not
cosmetic. `net` already contains receipts from invoices. A.2 forecasts a series
and then simulates receivable arrivals on top of it. Forecasting `net` would
therefore count every rupee of receivable **twice** — once inside the
forecast's own extrapolation and again in the Monte Carlo arrivals.

The first version of this system did exactly that. The result was tens of
millions of phantom cash and a business that could never run out; Runway-at-Risk
was pinned at the horizon in every configuration tested, which looked like a
healthy business rather than a bug. Splitting the series into a forecastable
operating residual and an explicitly simulated receivable stream is what makes
the two layers compose.

---

## Candidates

### 1. Seasonal naive — the baseline

`y_hat(t+h) = y(t + h - m)`. Repeat the last weekly cycle.

Included because it is the standard reference for seasonal series (Hyndman &
Athanasopoulos §5.2) and has **zero fitted parameters**, so it cannot overfit.
On strongly weekly-seasonal data it is a genuinely hard baseline — daily
business cash flow is dominated by day-of-week effects, and simply repeating
last week captures most of that.

Any model that cannot beat it is not earning its complexity.

### 2. Holt-Winters — damped additive

`statsmodels.tsa.holtwinters.ExponentialSmoothing`, additive trend, additive
seasonality, **damped trend**.

- **Additive, not multiplicative seasonality.** The target is *net* flow, which
  crosses zero and goes negative. Multiplicative seasonality is undefined on
  non-positive data. (The gross inflow series *is* multiplicative — which is why
  the generator composes it that way — but the forecaster does not target it.)
- **Damped trend** (Gardner & McKenzie 1985). An undamped linear trend
  extrapolated 90 days out produces implausible runway estimates. Damping is
  the standard correction and it matters here because the output feeds a
  solvency decision.

### 3. SARIMAX(1,0,1)(1,0,1,7)

`statsmodels.tsa.statespace.SARIMAX`.

Orders are deliberately modest. The series is already differenced in spirit —
net flow is a difference of levels — and high-order SARIMAX on three years of
daily data overfits readily. The order is a constructor argument so notebook 01
can search it rather than assert it.

Its main advantage is **analytic prediction intervals** derived from the
state-space covariance: a real distributional statement rather than a residual
heuristic. That alone justifies its cost even when its point accuracy only ties
Holt-Winters, because A.2 consumes interval *width* as its uncertainty input.

### Seasonal period m = 7

Daily business cash flow is dominated by day-of-week structure (weekends are
near-zero). Annual seasonality would need m = 365, which is intractable for
SARIMAX and is better handled by the STL decomposition in A.6 than by forcing
it into the forecaster.

---

## What was rejected

**A single train/test split.** Rejected outright. On a seasonal series with
injected regime shocks, a single split measures *which window you happened to
hold out* as much as it measures skill: hold out a month containing a shock and
SARIMAX looks terrible, hold out a calm month and it looks excellent.

**AIC/BIC as the selection metric.** Reported, but not decisive, for two
reasons. AIC is an *in-sample* criterion, and the seasonal naive baseline
defines no likelihood at all — ranking all three on AIC compares
incommensurable quantities.

This was not a theoretical concern. Measured on the same data:

| model | walk-forward RMSE | AIC |
|---|---:|---:|
| seasonal_naive | 47,622 | — (no likelihood) |
| holt_winters | 43,339 | **8,697** |
| sarimax | **39,417** | 9,778 |

**SARIMAX won out-of-sample while having the worse AIC.** In-sample fit and
out-of-sample skill disagreed, and selecting on AIC would have picked the wrong
model. This is precisely the case walk-forward validation exists to catch.

**MAPE as the selection metric.** Reported because it is the metric a
non-technical user understands, but never used to select. Net cash flow crosses
zero, and MAPE explodes near zero denominators — observed values of 205% to
390%, which are meaningless as a ranking. RMSE drives selection.

**scikit-learn's `KFold`.** Shuffling a time series destroys exactly the
structure being modelled, and would train on the future to predict the past.

---

## Walk-forward validation

The origin advances by `horizon` each fold, so evaluation windows are
non-overlapping and every fold trains only on data strictly earlier than its
own test window. The model factory is a *callable*, not an instance, because
each fold must refit from scratch — reusing a fitted model would leak later
data backwards through its state.

A side benefit worth keeping: the per-fold RMSE **spread** is as informative as
the mean, and is reported in `ModelSelectionScore.fold_rmses`. A model with a
good average and a terrible worst fold is a different proposition from a
consistently mediocre one.

---

## Prediction intervals

SARIMAX yields analytic intervals from its state-space covariance. The other two
define no such thing, so their intervals come from **empirical quantiles of
walk-forward residuals at each horizon step**:

```
interval(h) = point(h) + quantile(residuals[:, h], [(1-c)/2, (1+c)/2])
```

This is arguably the more honest construction: it assumes no normality and
measures the error the model actually made out-of-sample, rather than the error
its own likelihood claims it should make. The cost is needing enough folds to
estimate a quantile — estimating a 2.5% quantile from three numbers would be
false precision — so it falls back to a normal approximation below
`min_folds_for_intervals` and says so.

---

## The intervals were overconfident — diagnosis before fix

The Section C backtest measured **85.1% empirical coverage against a 95%
nominal level**. This is the **dangerous direction**: A.2 converts interval
width into the uncertainty it propagates into Runway-at-Risk, so intervals that
are too narrow make RaR optimistic — telling a business it has more runway than
it does, which is the exact failure this product exists to prevent.

An earlier version of this memo listed three *suspected* causes — the
structural break, calm-period quantiles, and residual heteroskedasticity — and
proposed widening as a remedy. **All three suspicions were wrong**, and the
proposed remedy would have been the wrong fix applied to the wrong branch. What
follows is what was actually measured, on 6,570 forecast/outcome pairs from a
73-step replay, before any change was made.

### The shortfall is not spread across the model — it is one branch

Coverage split by the model walk-forward selection actually chose:

| selected model | steps | pairs | coverage | mean width | sd of standardized residual |
|---|---:|---:|---:|---:|---:|
| `sarimax` | 54 (74%) | 4,860 | **96.0%** | 187,229 | 0.94 |
| `holt_winters` | 19 (26%) | 1,710 | **57.2%** | 85,807 | 2.73 |
| pooled | 73 | 6,570 | 85.9% | 160,668 | 1.62 |

SARIMAX's analytic state-space intervals are **fine** — 96.0% against a 95%
claim, very slightly conservative. Had the Holt-Winters branch merely matched
it, pooled coverage would have been 96.0%. **The entire shortfall lives in the
26% of steps that did not select SARIMAX**, i.e. in the empirical-quantile
interval path in `empirical_interval`.

That immediately rules out parameter uncertainty as the cause. It is real —
`get_forecast().conf_int()` conditions on the estimated parameters and ignores
their sampling variability — but it is a *small* effect here, and it applies to
the branch that is over-covering, not the one that is failing.

### Cause 1 — five folds cannot express a 95% interval

`empirical_interval` builds the interval as the 2.5% and 97.5% quantiles of the
walk-forward residuals at each horizon step. It runs with `n_folds=5`, so each
quantile is estimated from **five numbers**.

For *n* exchangeable residuals, the probability that a fresh observation lands
inside the sample range is (n−1)/(n+1). At n=5 that ceiling is **66.7%** — and
`np.quantile(col, 0.025)` on five sorted points interpolates at weight 0.1
between the first and second order statistic, i.e. essentially the minimum. So
the construction could not have reached 95% coverage no matter how good the
forecast was. The guard `min_folds_for_intervals=3` was set well below the
level at which the estimate becomes meaningful.

Measured coverage on that branch: **57.2%**, against a structural ceiling of
66.7%. The remaining gap is cause 2.

### Cause 2 — the interval stopped widening at day 14

`residuals_by_horizon` has `cv_horizon=14` columns, but the forecast runs to 90
days. `empirical_interval` indexes it with `min(h, h_avail - 1)`, so every
horizon from 15 to 90 silently reuses the **day-14** residual spread.

The fingerprint is exact:

| h | 1 | 13 | 14 | 15 | 30 | 60 | 90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `holt_winters` width | 97,006 | 63,639 | 85,077 | 85,077 | 85,077 | 85,077 | 85,077 |
| `sarimax` width | 184,099 | 184,961 | 184,966 | 185,442 | 186,384 | 188,202 | 189,999 |

The ratio of the Holt-Winters width at h=90 to h=14 is **1.0000** — not
approximately, exactly. A 90-day-ahead interval was being quoted at 14-day-ahead
width.

### Cause 3 — the calibration window sat in the middle of history

`walk_forward_validate` sets `min_train = n // 2` and advances the origin by
`horizon` for `n_folds` folds. With n≈1,000 and 5 folds of 14 days, the
evaluation windows cover days 500–570 — and days **570–1,000 are never
validated on at all**. The residuals that build the intervals come from the
middle of the history, not the recent past, which is precisely backwards for a
series with a regime change in it.

### What was ruled out, with numbers

- **Point-forecast bias.** Mean standardized residual is **−0.0055**; removing
  the bias entirely moves pooled coverage from 85.9% only to 86.4%. Misses are
  near-symmetric (8.5% below, 5.6% above). Rescaling the width alone by the
  measured 1.62 recovers **94.7%**. This is a width problem, not a centring
  problem.
- **The structural break.** Coverage pre-break 86.7% vs post-break 84.5%, on
  residual SDs of 45,880 and 46,858. A 2-point difference does not explain an
  11-point shortfall, and the break was the leading suspicion.
- **Heteroskedasticity.** Real but second-order and pointing the *other* way:
  weekday/weekend realized residual SD ratio is 1.19 against an interval width
  ratio of 1.00, and weekend coverage is *higher* (90.1% Sunday vs 84.0%
  Monday), so the constant width mildly over-covers the quiet days.

### A reporting artifact worth recording

The report's per-horizon table samples h ∈ {1, 7, 14, 30, 60, 90}. The replay
advances `as_of` by 14 days — an exact multiple of 7 — so a given horizon h
always lands on the **same weekday** at every step. Those six checkpoints
average **89.5%** coverage while true pooled coverage is **85.9%**: the
checkpoint set happens to sample the quieter weekdays. The per-horizon table
now reports bucketed coverage over all 90 horizons rather than six aliased
points.

### The fix that follows from this

Three defects, three targeted changes — not a blanket widening:

1. **Normalized split-conformal intervals** replace the raw residual quantile.
   The nonconformity score is `|residual| / scale(h)`, pooled across horizons,
   and the interval is `point ± q̂ · scale(h)` where `q̂` is the
   `ceil((N+1)·c)`-th order statistic. Pooling across horizons is what makes a
   95% level reachable from 5 folds at all (N = 5 × 14 = 70 scores, rank 68 ≤
   70), and the code **refuses and says so** when N is too small rather than
   returning a quantile it cannot support.
2. **`scale(h)` is defined over the whole horizon**, so widening does not stop
   at the CV horizon. SARIMAX contributes its analytic state-space scale;
   models without one get a power law `a·h^γ` with **γ fitted from the
   walk-forward residuals** rather than the previously assumed √h.
3. **Rolling origins are anchored to the end of the series**, so the
   calibration residuals are the most recent ones.

`q̂` is reported with every forecast, together with the Gaussian reference it
must be read against.

### Reading q̂ — the reference is z, not 1

`q̂` multiplies a **standard deviation**, because the nonconformity score is
`|residual| / scale(h)` and `scale(h)` is a sigma. So a model whose stated
scale is exactly right does *not* produce `q̂ = 1` — it produces
`q̂ = z_{(1+c)/2} = 1.96` at the 95% level, because that is the multiple of
sigma a 95% interval requires. The interpretable quantity is therefore

```
scale_ratio = q̂ / z      1.0 = the model's own scale was right
                         2.0 = it was half the size it should have been
```

Measured over the replay: mean `q̂` = 2.045 for SARIMAX and 2.057 for
Holt-Winters, i.e. `q̂/z` = **1.04** and **1.05**. Both scale profiles are
close to correct once measured against held-out error.

> **Correction.** An earlier draft of this memo read `q̂ ≈ 1` as "already
> honest" and reported the Holt-Winters multiplier of ≈2.3 as "2.3× too
> narrow", concluding that the two branches received visibly different
> treatment. That was a units error, and the measured table does not support
> it: both branches land in the same place. It is recorded here rather than
> silently amended, because misreading the units of one's own diagnostic is
> precisely the failure this memo exists to document.

That both models need only a few percent on top of z is the real result. Each
model's *scale profile* was roughly the right size all along; the failure was
in the interval **construction** wrapped around it — the five-fold quantile,
the clamp at the CV horizon, the mid-series folds — and those are what the fix
removed. Both branches are now recalibrated against held-out error, so neither
can under-cover silently; the previous design left the empirical branch with no
recalibration at all.

Measured before/after coverage is in `BACKTEST_REPORT.md` §2.
