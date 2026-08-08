# FinAscend — Backtest Report

Generated 2026-08-08 12:19 UTC by `app.services.backtesting.run_backtest`. Every figure below is produced by that run; none is written by hand.

## What was tested

A synthetic adversarial business over 1500 days (2022-01-01 to 2026-02-08), seed 42, opening with 1.2 months of cost coverage. That stressed opening position is chosen deliberately: a business holding several months of cash has no allocation problem for one month of bills, and an earlier run of this backtest confirmed it — every strategy paid every obligation, penalty was zero at all 29 steps, and the three strategies were indistinguishable because nothing was actually being decided. The harness replays the history in 21-day steps, and at each step forecasts 90 days ahead, runs 3,000 Monte Carlo iterations, and produces an allocation plan — using only information dated on or before that step.

**49 decision points** were evaluated for each of three strategies.

### The no-look-ahead guarantee

Every step routes through `pipeline.build_as_of_view`, which slices the world once and returns an object containing nothing dated after the decision date. The step function never receives the full dataset, so it cannot leak what it was not given. This is structural rather than a matter of discipline, because a leaky backtest is invisible in its own output — it simply looks like a good model.

## 1. Regret against perfect foresight

Regret is the extra penalty incurred versus a planner who knew exactly what cash would arrive. It isolates the cost of *uncertainty* from the cost of *bad method*: even perfect foresight cannot avoid all penalty when obligations exceed available cash, so the gap to that benchmark is the part attributable to not knowing the future.

| Strategy | Total realized penalty | Hindsight-optimal | Relative regret | Mean | p95 |
|---|---:|---:|---:|---:|---:|
| `rules_baseline` | 1,582,964 | 1,571,211 | 0.7% | 240 | 663 |
| `lp_optimizer` | 1,599,803 | 1,571,211 | 1.8% | 584 | 3,357 |
| `dp_knapsack` | 1,600,372 | 1,571,211 | 1.9% | 595 | 3,365 |
| `chance_constrained` | 3,477,827 | 1,571,211 | 121.3% | 38,911 | 127,227 |

**The LP optimizer changed total realized penalty by -1.1% relative to the rules baseline.**

Over-commitment — planning to spend more than the cash that actually materialized — occurred at **18/49** baseline steps, **19/49** LP steps, **19/49** DP steps and **0/49** chance-constrained steps.

### Side by side: realized penalty and over-commitment

The two columns that matter together. Realized penalty alone rewards whoever spent most aggressively in the periods where the forecast happened to be right; the over-commitment count is what exposes the risk taken to earn it. Reading either column on its own picks the wrong strategy.

| Strategy | Realized penalty | vs rules | Over-commitment | Mean regret | p95 regret |
|---|---:|---:|---:|---:|---:|
| `rules_baseline` | 1,582,964 | — | 18/49 | 240 | 663 |
| `lp_optimizer` | 1,599,803 | +1.1% | 19/49 | 584 | 3,357 |
| `dp_knapsack` | 1,600,372 | +1.1% | 19/49 | 595 | 3,365 |
| `chance_constrained` | 3,477,827 | +119.7% | 0/49 | 38,911 | 127,227 |

```
realized penalty (lower is better)

rules_baseline     |█████████████████████·························|  1.58M  over-commit 18
lp_optimizer       |█████████████████████·························|  1.60M  over-commit 19
dp_knapsack        |█████████████████████·························|  1.60M  over-commit 19
chance_constrained |██████████████████████████████████████████████|  3.48M  over-commit  0
```

The DP is included as a *replayed strategy*, not only as the solver cross-check it also serves. It optimizes over a grid subset of the LP's continuous feasible region, so its planned objective can only be equal or worse — but both solvers optimize against the same imperfect cash forecast, so which one ends up with lower *realized* penalty is not determined by that ordering. Reporting them side by side is what makes the difference between planning quality and outcome quality visible.

### Interpretation

**The optimizer did not beat the naive baseline, and that is the most informative result in this report.** It is reported as measured rather than tuned away.

The mechanism is visible in the over-commitment counts. The LP solves the allocation exactly — against a *forecast* of available cash. When that forecast is optimistic, the LP commits money that never arrives, and the unfunded obligations then incur their full penalty. Solving the wrong problem precisely is worse than solving roughly the right problem approximately: the greedy baseline is myopic, but its myopia happens to leave slack that absorbs forecast error.

This is the specific reason the chance-constrained formulation exists, and its two columns show the trade honestly. It eliminated over-commitment entirely (0/49 versus 19/49 for the LP) — but it paid 2.2x the LP's penalty to do so, because reserving cash against a 5% shortfall probability means declining obligations that would usually have been affordable.

The honest conclusion is that **none of the three dominates**. The right choice depends on whether the business can absorb a missed payment or not, which is a risk-appetite question rather than a modelling one. An engine that silently picked the LP because it is the most sophisticated would be making that decision on the user's behalf without telling them.

## 2. Forecast interval calibration

A 95% prediction interval claims ~95% of outcomes land inside it. This matters more than point accuracy, because A.2 converts interval WIDTH into the uncertainty it propagates into Runway-at-Risk. Intervals that are too narrow make RaR overconfident — telling a business it has more runway than it does, which is the exact failure this product exists to prevent.

**Pooled: 95.6% empirical coverage** against a 95% nominal level, over 4,410 forecast/outcome pairs.

> Intervals are well calibrated: 95.6% empirical coverage against a 95% nominal level.

### Before and after the conformal fix

**This configuration** (21-day steps, 4,410 pairs): **85.1% → 95.6%**. The 85.1% is the figure the previous build of this report published at these exact settings.

The cause was diagnosed before anything was changed, and the write-up is in `FORECASTING_METHODOLOGY.md`. The diagnosis needed a per-branch split, which this report does not compute, so it was run separately at 14-day steps — a denser replay giving 73 steps and 6,570 pairs. Those numbers are quoted below **on their own configuration**, not mixed with the line above; the two agree on the pooled story and differ slightly because they are different replays.

Diagnostic replay, 14-day steps, 6,570 pairs:

| | before | after |
|---|---:|---:|
| pooled coverage | 85.9% | **95.4%** |
| `sarimax` branch (74% of pairs) | 96.0% | 96.1% |
| `holt_winters` branch (26% of pairs) | 57.2% | 93.0% |
| sd of standardized residual | 1.62 | 0.97 |
| interval width ratio, h=90 vs h=14 (`holt_winters`) | 1.0000 | 1.23 |

The shortfall was not spread across the model at all. SARIMAX's analytic intervals were already honest at 96.0% coverage and remain so. The quarter of steps that selected Holt-Winters ran a *different* interval construction — per-horizon quantiles of five walk-forward residuals — and that branch alone accounted for the entire gap.

**What makes this recalibration rather than arbitrary widening.** The intervals genuinely are wider now; they had to be, because they were too narrow. Three things separate that from inflating them until the number looked right. The multiplier is *measured* from held-out error and never tuned against the coverage target. It lands within a few percent of the value a correctly specified model would need anyway (see the q̂ table in §3), so the correction is small and explicable rather than a large unexplained constant. And coverage is now flat across the horizon — every band within about a point of nominal — where a blanket widening would have over-covered at short horizons to rescue the long ones.

Three defects, each fixed by the thing that addressed it rather than by widening: (1) a per-horizon quantile of five residuals has a coverage ceiling of (n−1)/(n+1) = 66.7% and could never express 95%, so the scores are now pooled across horizons and read at the split-conformal rank ceil((N+1)c); (2) the interval stopped widening past the cross-validation horizon — a width ratio of exactly 1.0000 between day 90 and day 14 — so the scale profile is now defined over the full horizon with its exponent fitted rather than assumed; (3) the walk-forward folds sat in the middle of the history and never validated on recent data, so they are now anchored to the end of the series.

Coverage by forecast horizon band:

| Days ahead | n | Coverage | Mean interval width |
|---:|---:|---:|---:|
| 1–7 | 343 | 95.6% | 188,857 |
| 8–14 | 343 | 94.8% | 190,944 |
| 15–30 | 784 | 95.9% | 192,647 |
| 31–45 | 735 | 94.3% | 194,264 |
| 46–60 | 735 | 95.8% | 195,548 |
| 61–75 | 735 | 96.1% | 196,696 |
| 76–90 | 735 | 96.1% | 197,759 |

The breakdown is by **band** rather than by single horizon, and that change matters. The old table sampled h ∈ {1, 7, 14, 30, 60, 90}; since the replay advances `as_of` by a whole number of weeks, each of those horizons always lands on the same weekday, and coverage varies by weekday because weekend flows are near zero. Those six points averaged 89.5% while true pooled coverage was 85.9% — the checkpoint set was aliased against the calendar and flattered the result. Bands cover every horizon and cannot alias.

What this still does not fix: the interval width is constant across days of the week while realized volatility is not (weekday/weekend residual SD ratio 1.20 against a width ratio of 1.00). The result is visible as over-coverage at weekends — 99.3% on Sundays against 93.3% on Tuesdays. It is second-order next to the defects above and it errs safe, but a day-of-week or GARCH-type variance model is the honest next step.

## 3. Runway-at-Risk over the replay

- RaR(95%) ranged **1-90 days** (median 90)
- CRaR(95%) ranged **1.0-90.0 days** (median 90.0)
- Mean Monte Carlo standard error on the RaR estimate: 0.132 days

CRaR sits at or below RaR at every step, as it must — it is the mean of the tail beyond the RaR threshold. Were it ever above, the tail direction would be inverted and the metric would be reporting comfort where it should report danger.

Model selected by walk-forward validation across steps: `sarimax` 35x, `holt_winters` 14x

Conformal multiplier applied, by selected model. **Read q̂ against z = 1.960, not against 1.0** — q̂ multiplies a standard deviation, so a model whose stated scale is exactly right still needs 1.96 of them to cover 95%. The interpretable column is `q̂/z`, which is how many times too narrow the model's own scale was:

| Model | steps | mean q̂ | q̂/z | min | max |
|---|---:|---:|---:|---:|---:|
| `sarimax` | 35 | 2.045 | **1.043** | 1.457 | 2.789 |
| `holt_winters` | 14 | 2.057 | **1.049** | 1.333 | 2.487 |

Both models land close to the Gaussian reference, and that is the finding — not that they differ. Each model's *scale profile* is roughly the right size once it is measured against held-out error; the previous failure was in the interval **construction** wrapped around that scale, which no longer exists. A correction of a few percent on top of z is a small, explicable adjustment rather than the large unexplained constant that blanket widening would have required.

An earlier draft of this report claimed the multiplier came out near 1.0 for SARIMAX and near 2.3 for Holt-Winters, and read that as evidence the two branches were treated differently. That was a units error — comparing a sigma multiplier against 1.0 instead of against z — and the measured table above does not support it. It is corrected here rather than quietly dropped, because the mistake is exactly the kind this report is supposed to catch.

## 4. What this backtest does not establish

- **Synthetic data cannot validate real-world counterparty behaviour.** It validates that the estimators work against a known generating process, which is a genuine but strictly narrower claim.
- **The generating process is one the models are well suited to.** Delays are Gamma and the fitted candidates include Gamma, so the marginal fit is being graded on a question it was told the answer to. Real delay distributions may lie outside the candidate set entirely.
- **Regret is measured against a benchmark that is unachievable in production.** It bounds efficiency loss under known dynamics; it does not predict live performance.
- **One seed, one world.** These figures describe a single realization. A production evaluation would repeat across many seeds and report the distribution of regret, not a point estimate.
- **The obligation structure is imposed, not observed.** The generator models costs as an aggregate outflow series; the decomposition into payroll/rent/vendor obligations is a fixed chart-of-accounts assumption layered on top, so the optimizer's advantage depends on that structure being roughly right.
