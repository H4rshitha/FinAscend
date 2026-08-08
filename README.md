# FinAscend

A liquidity-risk engine for small businesses: cash-flow forecasting, Monte Carlo
runway-at-risk, payment allocation under uncertainty, credit-risk scoring and
receipt OCR ingestion — with a quant-terminal dashboard over the whole thing.

The organizing principle is that **a method is only worth what its validation is
worth.** Every number in this repository is produced by a run that can be
repeated, carries the configuration that produced it, and is reported as
measured — including where the result is inconvenient.

## Where to start

| Document | What it covers |
|---|---|
| [`finascend/README.md`](finascend/README.md) | **Run instructions**, project layout, quick start |
| [`finascend/QUANT_METHODOLOGY.md`](finascend/QUANT_METHODOLOGY.md) | Every statistical and optimization choice, what was rejected, honest limitations |
| [`finascend/FORECASTING_METHODOLOGY.md`](finascend/FORECASTING_METHODOLOGY.md) | Model selection memo, and the interval-coverage diagnosis in full |
| [`finascend/BACKTEST_REPORT.md`](finascend/BACKTEST_REPORT.md) | Generated backtest: regret, solver comparison, interval calibration |
| [`finascend/OCR_ACCURACY.md`](finascend/OCR_ACCURACY.md) | Per-tier OCR field accuracy, never pooled |
| [`FinAscend_Architecture_Plan.md`](FinAscend_Architecture_Plan.md) | The original full-system design |

## Three results reported against interest

**The optimizer did not beat the naive baseline.** Over 49 replayed decision
points the rules baseline incurred 1.58M in penalty against the LP's 1.60M
(+1.1%). Solving the allocation exactly against an *imperfect cash forecast*
over-commits: the LP planned to spend money that never arrived at 19/49 steps.
The greedy baseline is myopic, but its myopia leaves slack that absorbs forecast
error. The chance-constrained variant eliminated over-commitment entirely (0/49)
and paid 2.2x the penalty to do it. None of the four dominates, and that is a
risk-appetite question rather than a modelling one.

**The prediction intervals were overconfident — 85.1% coverage on a 95% claim.**
The cause was not what it looked like. The shortfall was not spread across the
model: SARIMAX's analytic intervals were already honest at 96.0%, and the entire
gap lived in the 26% of steps selecting Holt-Winters, whose intervals came from a
per-horizon quantile of *five* residuals — a construction with a structural
coverage ceiling of (n−1)/(n+1) = 66.7% that could never have expressed 95%.
Normalized split-conformal recalibration took pooled coverage to **95.6%**, flat
across all seven horizon bands. Residual day-of-week heteroskedasticity is
documented and still unfixed.

**OCR degrades sharply, and the field the ledger depends on fails unsafely.** On
the HARD tier `invoice_number` declined 21 of 24 times and was wrong 0 times — it
fails safe. `total_amount` declined 0 times and was **wrong 12 times**, because a
corrupted number is still a parseable number. The normalizer therefore refuses to
build a record rather than defaulting an amount, and cross-checks the implied tax
rate against the receipt's own internal redundancy.

## Stack

Python · FastAPI · statsmodels · scikit-learn · PuLP/CBC · SHAP · EasyOCR ·
Next.js · TypeScript

## No hosted demo

There is deliberately no live URL. The dashboard's premise is that **every
displayed number comes from a live API call** — there are no fixtures, seeded
constants or fallback values anywhere in the frontend — so it requires the
FastAPI backend running alongside it and cannot be served as a static build. See
[`finascend/README.md`](finascend/README.md) to run it locally.
