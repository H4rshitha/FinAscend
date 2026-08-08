# FinAscend — Quantitative Methodology

Every statistical and optimization choice in the Quant Core, what was
considered instead, and what the honest limitations are.

The organizing principle: **a method is only worth what its validation is
worth.** Numbers that look plausible without a checkable method behind them
are worse than no numbers, because they invite decisions they cannot support.
Where a result is negative or inconvenient, it is reported as measured.

---

## 0. The synthetic data generator

There is no real bank data, so the generator is the foundation everything else
is validated against. Its defining property is that **the true parameters are
known**, which makes the fitting code falsifiable: it can be scored on whether
it recovers them, rather than merely inspected.

### Composition

```
inflow_t   = receipts_t + cash_sales_t
receipts_t = sum of invoice amounts actually arriving on day t
outflow_t  = base * outflow_trend_t * damped_weekly_t * lognormal_noise_t
```

**Multiplicative, not additive.** Business cash flows scale: a 20% bad quarter
costs a large firm more rupees than a small one, and seasonal amplitude grows
with the level. An additive model imposes constant absolute seasonality, which
is wrong for revenue and would make the multiplicative Holt-Winters variant
untestable.

### Four design decisions, each forced by a failure

These were not designed up front. Each one replaced something that broke, and
the failures are recorded because they are the useful part.

**1. Inflow is built FROM the invoices, not alongside them.**
The first version generated the daily inflow series and the per-invoice
payment records as two independent processes. That is incoherent — in a real
business the bank inflows *are* the invoice payments — and it was not merely
cosmetic. A.2 forecasts the daily series and then adds simulated receivable
arrivals on top, so two unrelated streams injected tens of millions of phantom
cash. Runway-at-Risk was pinned at the horizon in every configuration tested.
The fix introduced `net_ex_receipts` (cash sales minus costs) as the
forecastable series, keeping the decomposition explicit rather than leaving it
to caller convention.

**2. Revenue and costs move on separate trends.**
Sharing one trend locks inflow and outflow in proportion, so the margin can
never deteriorate and the business can never get into trouble. Costs are
sticky — rent, payroll and loan EMIs do not shrink when revenue does — and
giving outflow its own trend is what produces a finite forward runway. The
shock multiplier is applied to revenue only, for the same reason: a demand
shock cuts sales, not the rent. That asymmetry is what makes a shock a
liquidity event.

**3. The revenue trend is piecewise, with a structural break.**
A constant negative margin sustained across three years compounds into a
deficit no real business survives, and forcing the balance to stay positive
under one requires an implausible opening cash pile. Real distress is recent:
the firm traded near break-even, then something changed. The break also gives
A.1 an honestly hard problem, since a structural break is what extrapolative
models handle worst.

**4. A 120-day warm-up.**
Without pre-window invoices, receipts ramp from zero over roughly
(payment terms + delay) days while costs run at full rate — burning about 8M
of opening cash on a pure artifact of where the window starts.

### Two couplings that make the world learnable

**Serial persistence (AR(1), rho = 0.75).** A counterparty's latent "slowness"
carries week to week:

```
z_t = rho * z_{t-1} + sqrt(1 - rho^2) * e_t,    e_t ~ MVN(0, R)
```

The `sqrt(1 - rho^2)` scaling keeps the marginal variance at exactly 1, so the
probability-integral transform still yields uniforms and the cross-sectional
correlation `R` is unchanged. Without persistence, payment *history* carries no
information about *current* state, and every credit model in A.4 is capped at
roughly chance — measured at ROC-AUC 0.57, where the only learnable signal was
the counterparty's own baseline rate.

**Default coupled to stress (beta = 0.9).**
`P(default) = sigmoid(logit(p_i) + beta * z)`. A firm paying late is
empirically more likely to eventually not pay at all; both are symptoms of one
cause. With an i.i.d. default flip there is no relationship between observable
behaviour and default, and A.4's whole model-versus-baseline comparison would
be measuring noise.

These two constants set the ceiling on how well any credit model can do, so
they are stated at module top level rather than buried, and A.4's reported AUC
should be read against them.

### Survivorship bias — a real consequence, not a defect

Because default probability rises with stress, and stress produces long
delays, **defaulted invoices are disproportionately the slow ones**. Only paid
invoices yield an observed delay, so the observed distribution is the true
Gamma with its right tail preferentially removed: it is `P(delay | paid)`, not
`P(delay)`.

This is left in place deliberately. For simulating cash *receipts* it is the
correct conditional distribution — a defaulted invoice contributes no cash on
any date, and default is modelled separately by A.4. But it means A.2's fitted
marginals must be read as conditional-on-payment, and the generator's true
parameters are recoverable only up to that censoring. `stress_beta=0` disables
it so tests can isolate the pure delay mechanism.

---

## 1. Forecasting

See **`FORECASTING_METHODOLOGY.md`** for the full memo. Summary:

- Three models behind one interface: seasonal naive, Holt-Winters (damped
  additive), SARIMAX(1,0,1)(1,0,1,7).
- Selection by **walk-forward (rolling-origin) cross-validation** on
  out-of-sample RMSE. Not AIC: the seasonal naive baseline defines no
  likelihood, so ranking all three on AIC compares incommensurable quantities.
- Measured result: SARIMAX won on out-of-sample RMSE (39,417) despite a
  **worse** AIC than Holt-Winters (9,778 vs 8,697). In-sample fit and
  out-of-sample skill disagreed, which is the case walk-forward exists to
  catch.
- MAPE is reported but never used for selection: net cash flow crosses zero
  and MAPE explodes near zero denominators (observed values of 200-390%).
- Prediction intervals are **normalized split-conformal**: the interval is
  `point ± q̂ · scale(h)`, where `scale(h)` is the model's own uncertainty
  profile and `q̂` is the `ceil((N+1)c)`-th smallest of the nonconformity
  scores `|residual| / scale(h)` pooled over every walk-forward cell. This
  replaced a construction measured at **57.2% coverage on a 95% claim**; the
  full diagnosis and the before/after are in `FORECASTING_METHODOLOGY.md`.
  Pooled backtest coverage went **85.1% → 95.6%**.

  **Reading `q̂`: the reference is z, not 1.** `q̂` multiplies a standard
  deviation, so a model whose stated scale is exactly right still needs
  `z = 1.96` of them to cover 95%. The interpretable quantity is `q̂/z` — how
  many times too narrow the model's own scale was. Measured: mean `q̂` = 2.045
  (SARIMAX) and 2.057 (Holt-Winters), i.e. `q̂/z` = 1.04 and 1.05.

  What makes this recalibration rather than widening is therefore *not* that
  the two branches got different treatment — they did not. It is that (a) the
  multiplier is measured from held-out error and never tuned against the
  coverage target, (b) it lands within ~5% of the value a correctly specified
  model needs anyway, so the correction is small and explicable rather than a
  large unexplained constant, and (c) coverage is now flat across the horizon
  (94.3–96.1% across seven bands), where blanket widening would over-cover
  short horizons in order to rescue long ones.

  An earlier version of this document claimed `q̂ ≈ 1.0` for SARIMAX and
  `≈ 2.3` for Holt-Winters. That was a units error — comparing a sigma
  multiplier against 1.0 rather than against z — and the measured table does
  not support it.

---

## 2. Monte Carlo and Runway-at-Risk

### Marginal fitting

Gamma, Weibull and Log-normal fitted by MLE via `scipy.stats.<dist>.fit(data,
floc=0)`.

**Location pinned at zero.** Left free, scipy slides `loc` just below the
sample minimum, which inflates the likelihood, gives zero density below that
point, and makes the KS test meaninglessly good. A delay of zero days is
physically possible, so `loc=0` is the honest constraint.

**Selection by AIC, not KS p-value.** A p-value tests *refutation*, not
relative fit, and is not monotone in fit quality — with ~200 observations
several families are un-refuted simultaneously. All three candidates have the
same parameter count, so AIC reduces to a likelihood comparison on equal
footing. The KS statistic is still stored, because it answers the separate and
necessary question of whether the winner is *adequate* or merely least-bad.

**Two caveats on that KS p-value**, both stated rather than corrected:
1. *Serial correlation.* The KS test assumes i.i.d. observations. Delays are
   persistent, which shrinks the effective sample size and makes the test
   over-reject a correct distribution. Measured directly on data with a
   known-correct Gamma: the median p-value across counterparties rose
   0.031 → 0.190 → 0.257 → 0.630 at thinning factors 1, 4, 8, 12, while the
   moments were recovered accurately throughout. The distribution was right;
   the test was wrong.
2. *Estimated parameters.* Fitting and testing on the same data invalidates
   the standard null (a Lilliefors-type correction would be needed), biasing
   the p-value up and partially offsetting caveat 1.

The p-value is therefore used as a relative adequacy signal, not a formal test.

### Dependence: why not independence

Sampling each counterparty's delay independently is the intuitive default and
it is **badly wrong in the direction that flatters the business.**

Under independence, all 12 counterparties being simultaneously slow has
probability ~0.2^12 ≈ 4e-9 — effectively never. The simulated worst case is
"a few are late", the cash trough is shallow, and the runway looks
comfortable. In reality those events share common causes: a demand downturn, a
credit squeeze, a festival period. A system-wide slowdown is exactly the event
that empties the account, because it removes every inflow at once with no
offsetting early payment.

**Measured:** on the adversarial world, independent sampling reported
**RaR = 28 days** against the copula's **19 days** — independence overstates
runway by 47%.

### Copula construction

Student-t copula by default (Gaussian available), because the Gaussian copula
has **zero asymptotic tail dependence**: however high its correlation, extreme
joint events still decouple in the far tail — the wrong asymptotic behaviour
for precisely the scenario that matters. Low-df t keeps counterparties coupled
in the tail.

Sampling has three steps, and the middle one is where the bug lives:

1. Draw correlated latent variates `z ~ MVN(0, R)` (or multivariate t).
2. **Probability integral transform**: `u = F_latent(z)` → Uniform(0,1).
3. Invert through each counterparty's own fitted marginal:
   `delay_i = F_i^{-1}(u_i)`.

Skipping step 2 and treating latents as delays gives the right correlation and
completely wrong marginals — symmetric, unbounded, able to go negative. It is
invisible to any correlation check, so
`test_copula_marginals_survive_the_transform` exists specifically to catch it
and is kept **independent** of the correlation test.

### Correlation estimation

Spearman rank correlation, converted to the Gaussian-copula linear parameter by
`rho = 2 * sin(pi * rho_s / 6)` (Kruskal 1958). Ranks rather than Pearson
because Pearson measures linear association and is distorted by the heavy right
tail of a Gamma delay — one enormous delay can dominate the estimate — while
ranks are invariant to the marginal transform, exactly the property a copula
parameter needs.

**A bug worth recording.** Correlation must be estimated from a **time-aligned
panel**, not from ragged per-counterparty arrays. Defaulted invoices leave gaps
at different dates for different counterparties, so a single dropped row shifts
every subsequent entry and element-wise correlation ends up comparing week 5
against week 6.

**Recovery is configuration-dependent, and a single quoted number is
misleading.** An earlier version of this document reported "0.178 (easy) and
0.735 (adversarial)" with no configuration attached, and those figures do not
reproduce. Re-measured on seed 42, with the configuration stated:

| regime | days | counterparties | true ρ | recovered (mean off-diagonal) | ragged-array bug |
|---|---:|---:|---:|---:|---:|
| easy | 1095 | 12 | 0.15 | **0.122** | −0.020 |
| easy | 1500 | 10 | 0.15 | **0.144** | — |
| adversarial | 1500 | 10 | 0.72 | **0.739** | — |
| adversarial | 1095 | 12 | 0.72 | **0.527** | 0.169 |
| adversarial | 1460 | 8 | 0.72 | **0.546** | 0.151 |

The last row is the configuration **notebook 02** uses, listed here because that
notebook is where most readers first meet these numbers and an unattributed
0.546/0.151 pair is exactly the stale-figure hazard this table exists to close.
Its world has a 20.0% default rate.

The same true parameter recovers as 0.527 or 0.739 depending only on history
length and counterparty count, because the estimate depends on how many paired
weekly observations survive default censoring — and the adversarial regime
censors hard. The ragged-array column is the bug for contrast: it collapses the
estimate toward zero regardless of the truth.

`test_correlation_recovery_is_bounded_per_configuration` pins each row with its
own band, so a stale figure cannot survive a refactor unnoticed. The bands are
deliberately wide: they protect against the alignment bug, not against ordinary
sampling variation.

Pairwise estimates need not cohere into a positive-definite matrix, so the
result is projected onto the nearest PD matrix by eigenvalue clipping
(Higham 2002, simplified) before Cholesky.

### RaR and CRaR

```
RaR_c  = quantile(days_to_zero, 1 - c)
CRaR_c = E[ days_to_zero | days_to_zero <= RaR_c ]
```

The tail of interest is the **lower** one, because short runway is bad. This
is the mirror image of market-risk VaR, where the bad tail is large losses;
inverting it would produce a reassuring number. CRaR is reported alongside
because two businesses can share a RaR of 11 days while one's bad tail averages
10 days and the other's averages 3. RaR locates the cliff; CRaR measures the
drop.

**Why 10,000 iterations.** Not because it is a round number. The convergence
study fits standard error against N on a log-log scale and checks the slope
against the theoretical **−0.5**, which is what makes the standard error at any
chosen N predictable. Iteration count is then set by the error tolerance the
decision needs, and the achieved SE travels with every result.

**The slope must be quoted per configuration, and with its sampling noise.** An
earlier version of this document reported a single figure of **−0.517** as
confirmation of the rate. That figure does not reproduce in either
configuration, and quoting any single slope to three decimals was false
precision: a slope fitted through six points is itself a noisy estimate.
Re-measured:

| configuration | grid | slope | notes |
|---|---|---:|---|
| toy continuous paths (`_toy_paths`, seed 3) | 500–16k | −0.602 | the unit-test config |
| same, across seeds 1–10 | 500–16k | **−0.434 mean**, range −0.342 to −0.602, sd 0.083 | single-seed spread |
| full simulator, adversarial world (notebook 02, seed 5) | 250–32k | **−0.321** | SE floors out; see below |

The **toy continuous** configuration is the one that legitimately demonstrates
the rate, and it does: mean −0.43 across ten seeds, with a single-seed spread of
0.26 that comfortably brackets −0.5. The honest reading is "consistent with
O(N^{-1/2}) given the noise in the slope estimator", not "confirms −0.5".

The **full-simulator** configuration measures −0.321 and is the more
interesting number, because it is shallow for a structural reason rather than a
sampling one. `days_to_zero` is **integer-valued with a point mass at the
horizon**, so its quantile is heavily tied. The bootstrap SE of a tied quantile
does not decay smoothly — it decays in steps and then floors at the resolution
of the integer grid. Measured on that run, the SE plateaus at ≈0.49 between
N=4,000 and N=8,000 and hits **exactly 0.000** at N=32,000, where every
bootstrap resample returns the same integer (RaR is pinned at 41 days from
N=250 onward). Fitting a power law through a floored series biases the slope
toward zero, which is precisely what −0.321 is.

So the asymptotic rate is demonstrated where it is meaningful (continuous
estimand) and the departure from it is explained where it is not (discrete,
tied estimand). `test_monte_carlo_error_decays_as_one_over_sqrt_n` drops the
degenerate `SE = 0` points before fitting and bands the slope at
(−0.85, −0.25) — wide on purpose, because with sd = 0.083 per seed a tight band
would be testing the seed rather than the estimator.

The SE is computed by **bootstrap** rather than the asymptotic quantile-variance
formula, which requires estimating the density at the quantile — itself noisy,
and badly behaved when the distribution has a point mass at the censoring
horizon. The bootstrap handles that without special-casing.

### Uncertainty propagation

A.1's prediction intervals are converted to a per-step standard deviation,
`sigma = (upper - lower) / (2 * z)`, and simulated jointly with receivable
timing. Treating the forecast as certain and then simulating "uncertainty" on
top of it double-counts confidence and understates risk. The conversion assumes
approximately symmetric Gaussian intervals — exact for SARIMAX's analytic
intervals, an approximation for empirical-quantile ones that slightly
understates asymmetric tails.

---

## 3. Optimization

### The problem

```
minimize    sum_i w_i * (a_i - x_i)          total penalty incurred
subject to  sum_i x_i <= C
            0 <= x_i <= a_i
            x_i in {0, a_i}  if RIGID
```

Defined once in `problem.py` and shared, so that when the two solvers agree
that agreement is between two *algorithms* rather than two transcriptions.

`w_i` is **derived from contract terms**
(`late_fee_rate_per_day * days_overdue * relationship_weight`), not assigned.
A hand-typed penalty severity would silently determine which obligations get
paid, making the "optimization" the analyst's prior expressed as a weight
vector. `relationship_weight` isolates the one genuinely subjective input as a
single named multiplier rather than blending it invisibly into a composite.

### LP/MILP

PuLP + CBC. The rigid linkage is the exact equality `x_i = a_i * y_i` rather
than the more common big-M pair — big-M is a standard source of silent
wrongness, since too small cuts off the optimum and too large loosens the
relaxation until branch-and-bound crawls.

### The DP, and its rewrite

Minimizing incurred penalty equals maximizing avoided penalty (the difference
is a constant), which is exactly a knapsack: cash is capacity, amount is
weight, avoided penalty is value.

The naive divisible-item transition is O(k) per cell, giving **O(n·C·k)** —
genuinely too slow, running for minutes on a 30-obligation instance. It
collapses to O(C) with one substitution. With `v = unit * w_i` and `j = c - t`:

```
dp[i][c] = max_j { dp[i-1][j] + (c - j) * v }
         = c*v + max_{j in [c-k, c]} { dp[i-1][j] - j*v }
                 \_______________________________/
                   sliding-window maximum, width k+1
```

Computed in amortized O(1) per cell with a monotonic deque — the max-plus
convolution of a linear kernel, the same trick that linearizes bounded
knapsack. Total **O(n·C)** for both item types; minutes became milliseconds.

Space stays O(n·C) rather than the rolling O(C), because reconstructing the
chosen allocation requires the parent table, and an allocation the user cannot
see is not an answer.

### Cross-validation, and its asymmetry

The DP optimizes over a **grid subset** of the LP's continuous feasible region,
so its incurred penalty can only ever be **higher**. That makes the tolerance
one-sided:

- DP worse than LP by more than the discretization bound → too coarse a unit.
- **DP better than LP by any amount → a bug**, almost certainly the DP
  spending cash it does not have.

Knowing which direction an approximation can err in is what turns a tolerance
into a test. The bound itself is derived
(`unit * sum(penalty_rates)` plus CBC's own relative optimality gap), not
chosen to make the test pass.

**Measured:** exact agreement (delta = 0.0000) on all-rigid unit-aligned
instances where the DP is exact; and the gap shrinks proportionally with the
unit — 272 → 128 → 41 → 18 → 9 → 5 at units 10000 → 200 — confirming it is
discretization rather than error.

### Chance-constrained allocation

The textbook SAA formulation adds one binary per scenario plus big-M linking.
With hundreds of scenarios on top of the rigid-obligation binaries, that MILP
is slow exactly where the Section C harness re-solves it every replay day.

This implementation exploits the structure instead. The constraint couples to
the decision only through total spend, and spending is **monotone** — spending
more can only make a shortfall more likely. So the chance constraint reduces
exactly to a single budget cap:

```
spend <= Q_eps,  Q_eps = eps-quantile of simulated minimum free cash
```

One deterministic MILP solve. The monotonicity argument is what makes the
reduction legitimate, and it would fail if paying an obligation could itself
generate inflow (a settlement discount, say), which this model does not
include.

**Measured:** achieved shortfall probability lands exactly on epsilon
(0.200, 0.100, 0.050, 0.010 for the four levels tested), and the risk/penalty
frontier is monotone — tighter epsilon spends less and costs more.

**Scenario cap: 200-500, not the full 10,000.** Justified the same way as the
iteration count: subsample stability, measured as the coefficient of variation
of the safe-spend limit across independent resamples, improves as ~1/sqrt(S) —
0.0130 at S=50, 0.0063 at S=300, 0.0022 at S=1000. Picking a cap because it
solves quickly, without that check, would be the magic-number failure this
project rules out.

**Infeasibility is surfaced.** When even zero spend breaches epsilon, the
solver says so explicitly rather than returning a bland "spend 0" — which
reads as a cautious recommendation when it actually means the target is
unreachable and the business needs financing or renegotiation, not a cleverer
payment schedule.

---

## 4. Credit risk

The deck answered its own "black-box trust barrier" by avoiding ML and
declaring the scorer rule-based and therefore auditable. That buys
explainability by giving up predictive power and never establishes that the
rules are any good. The position here is the harder one: fit a real model, then
*earn* the trust with calibration, attribution, and a measured comparison.

### Setup

Features are computed from each counterparty's history **strictly before** the
invoice being predicted, accumulated in issue order. An "historical on-time
rate" computed over the full dataset would include the very invoice being
predicted and produce an ROC-AUC near 1.0 that means nothing.

`delay_trend` (OLS slope of delay against sequence number) is included because
a counterparty whose delays are *lengthening* is a different risk from one with
the same mean that is stable — a model seeing only the mean cannot distinguish
them.

### Measured results

| model | ROC-AUC | Brier | lift vs rules |
|---|---:|---:|---:|
| `rule_baseline` (days overdue × amount) | 0.505 | 0.176 | — |
| `logistic_l2` | 0.583 | 0.128 | +0.078 |
| `gbm` | **0.654** | 0.124 | **+0.149** |

The GBM clearly beats the logistic model, which is explicable rather than
mysterious: the true stress→default link is a sigmoid of a latent variable, so
there is genuine nonlinearity a linear model cannot capture. AUC 0.65 is a
credible credit-model number — real-world scorecards commonly land in
0.65-0.75 — not an inflated one.

**Accuracy is never reported.** With a ~15% default rate, predicting "never
defaults" scores 85% accuracy while being useless. Quoting it would be the most
misleading thing this module could do.

### Calibration over ranking

`class_weight="balanced"` was tried and **rejected**. It improves ranking on an
imbalanced problem, but it reweights the likelihood, shifts the intercept and
destroys the probability scale: measured at 0.38-0.60 predicted where observed
was 0.08-0.22, more than doubling the Brier score (0.244 vs 0.108).

Calibration matters more than ranking here because the output is consumed as an
actual probability — it feeds the Monte Carlo engine and through it the
optimizer. A well-ranked but badly scaled probability silently corrupts RaR.
ROC-AUC still measures ranking, and is invariant to monotone rescaling, so
nothing is lost by reporting both.

Post-fix calibration tracks closely: predicted 0.096-0.227 against observed
0.101-0.190.

### Explainability

Logistic: coefficient × standardized value, the exact additive effect on the
log-odds, with Wald confidence intervals. GBM: SHAP values via
`TreeExplainer` — chosen over impurity-based importance because importance is a
global model-level statistic and cannot answer "why is *this* invoice scored
high", which is the question a user reviewing one decision actually asks.

The Wald intervals are **conservative and stated as such**: they are unpenalized
intervals around L2-penalized coefficients, so they are too wide and not centred
on an unbiased estimate. They indicate which features are precisely estimated,
not exact frequentist coverage. A bootstrap would give honest intervals under
penalization and is the right upgrade if these are ever used for inference
rather than explanation.

`pinv` rather than `inv` for the information matrix: a zero-variance feature
standardizes to an all-zero column, making `X'WX` exactly singular and sending
every interval to NaN. (This actually happened — invoices were issued on a
fixed weekly day, so `days_since_last_invoice` was constant at 7. The generator
now jitters issue dates, which is also more realistic.)

---

## 5. Anomaly detection

Three detectors that fail in **different** ways, because a single unsupervised
detector on unlabelled data cannot be validated — there is no ground truth, so
"it found 40 anomalies" is not a result. Comparing detectors that fail
differently is the closest thing to validation available without labels.

- **Robust z (MAD-based).** Univariate on amount. Uses median/MAD because the
  mean and SD are dragged toward the outliers being hunted — one extreme value
  inflates the SD enough to hide itself. This is the masking effect, and it is
  why the textbook z-score is wrong for the job it is most often used for. The
  MAD has a 50% breakdown point.
- **Isolation Forest.** Multivariate. Flags records unremarkable on every
  single axis but implausible in combination.
- **DBSCAN.** Density clustering for near-duplicate resolution; genuine
  one-offs land in the noise label.

Low agreement between them is the *expected* result and would be more
suspicious if it were high, because they are answering different questions.

---

## 6. Unstructured data

**Receipt classification** by character n-gram TF-IDF + cosine similarity to a
small reference set. Regex was rejected: it fails on exactly the cases that
matter (OCR noise, abbreviations, unseen vendors), and a rule list cannot
generalize to a vendor it has no rule for.

Character n-grams (`char_wb`, 3-5) rather than a neural sentence embedding — a
stated trade. Character models tolerate OCR character corruption ("INV0ICE" vs
"INVOICE"), need no model download, and are reproducible offline. A transformer
would capture that "cab fare" and "taxi journey" mean the same thing, and is the
right upgrade if semantic generalization outweighs OCR robustness. The
vectorizer is swappable without touching callers.

The runner-up score is always returned, because the **margin** indicates
reliability: a top score of 0.4 is confident when the runner-up is 0.1 and a
coin-flip when it is 0.39.

**Seasonality** via `statsmodels` STL rather than the architecture plan's
unfalsifiable "macro signals". STL allows the seasonal component to evolve, and
`robust=True` downweights outliers so a single shock does not distort the whole
estimate — directly relevant given the injected regime shocks. Variance shares
are reported so a seasonality claim can be checked: if the seasonal component
explains 2% of variation, "this business is highly seasonal" is false and the
number says so. Those shares are described as *share of component variation*,
not "variance explained", because the components are correlated and their
variances do not sum to the total.

---

## 6b. Receipt OCR ingestion

The chain is `image → ocr_service → A.6 classifier → normalizer → A.5 dedup`,
each stage a function on the previous stage's declared type, so the engine is
swappable without touching any caller.

### Engine choice

**EasyOCR**, not pytesseract. The constraint was no credentials and no
separately-installed binary, and `pip install easyocr` is the whole
installation — a CRAFT detector plus a CRNN recognizer on PyTorch, CPU-only.
`pip install pytesseract` installs only a *wrapper*; the Tesseract binary is a
separate OS-level install, which is precisely the dependency that was excluded.

The cost is real and worth naming: PyTorch pulls ~2 GB of wheels and the
recognition weights download on first use. Both engines are implemented behind
the same `OcrEngine` protocol, and `FINASCEND_OCR_ENGINE=tesseract` selects the
fallback on a size- or network-constrained host. Having a second engine behind
the interface is what demonstrates the interface is actually engine-agnostic
rather than EasyOCR's shape with a different name.

### Why a generated corpus

The same argument as A.0. Three tiers — CLEAN (flatbed scan), MODERATE (careful
phone photo), HARD (bad light, skew, blur) — with degradations applied in
**physical acquisition order**: geometry, then optics, then illumination, then
sensor noise, then JPEG. Compressing before blurring would produce ringing no
real photograph contains. Every tier renders the *same* receipt content, so any
accuracy difference is attributable to image quality rather than to having
drawn an easier set of vendors.

### Measured, per tier — never pooled

Full tables in `OCR_ACCURACY.md`. Headline:

| Tier | vendor | invoice no. | date | total | all four |
|---|---:|---:|---:|---:|---:|
| clean | 100.0% | 91.7% | 100.0% | 95.8% | 87.5% |
| moderate | 100.0% | 83.3% | 100.0% | 95.8% | 83.3% |
| hard | 75.0% | 12.5% | 33.3% | 50.0% | 4.2% |

A single blended number over these would be a statement about the corpus mix,
not the pipeline: weight it toward clean scans and the identical code reports a
far better figure.

### Two findings worth more than the accuracy numbers

**1. Layout analysis, not just recognition.** A detector splits a text region
wherever it sees a gap — including the gaps a thousands separator and a decimal
point leave once blur has swallowed them. `INR 159,312.98` arrives as the
separate regions `159`, `312`, `98`. Parsed region by region that total is
unrecoverable; parsed as the visual row it actually is, it reads back exactly.
Row grouping is done on the **deskewed** coordinate `y·cosθ − x·sinθ`, with θ
the median of the detector's own per-region baseline angles, because at 7° a
single row drifts further vertically across the page than a row is tall and
grouping on raw `y` shatters it. (Observed symptom: a vendor name coming back
as "Properties Pvt Ltd Sunrise".)

**2. The field the ledger depends on is the one that fails unsafely.** On the
HARD tier `invoice_number` declined 21 of 24 times and was wrong 0 times — it
fails safe. `total_amount` declined 0 times and was **wrong 12 times**. A
corrupted number is still a parseable number, so nothing downstream knows to
doubt it.

Two responses, both structural. `normalize` **refuses to build a record** when a
required field is unreadable, rather than defaulting the amount — a placeholder
zero is a valid `Outflow` that reaches the optimizer looking like data. And
because a receipt is internally redundant (`total = subtotal + tax`), the
implied tax rate is checked against a 1–40% band whenever the tax line was also
read. That catches the lost-decimal-point case, which is wrong by a factor of
100 and otherwise entirely plausible. On this evidence the HARD tier should
route to human review rather than straight into the ledger, and the pipeline
says so rather than reporting an average that hides it.

---

## 7. Honest limitations

- **Synthetic data cannot validate real-world counterparty behaviour.** It
  validates that the estimators work against a known generating process, which
  is genuine but strictly narrower.
- **Parameter recovery proves the estimator inverts its own generating
  process.** The delays are Gamma and Gamma is among the candidates, so the
  marginal fit is graded on a question it was told the answer to. If real
  delays lie outside {Gamma, Weibull, Log-normal}, recovery here says nothing.
- **Copula correlation is assumed, not observed.** The dependence structure is
  imposed by the generator, so the estimator recovers a number that was put
  there by hand. On real data this is the hardest and most consequential
  parameter to estimate.
- **Default labels are generated, not observed.** A.4's ROC-AUC is an upper
  bound on what the same features would achieve against real defaults, and is
  capped by the generator's own `DEFAULT_STRESS_BETA` and `DELAY_PERSISTENCE`.
- **The prediction intervals were measurably overconfident, and now are not —
  but the residual heteroskedasticity is unfixed.** Pooled coverage went
  85.1% → 95.6% after the conformal recalibration. What remains is that
  interval width is constant across days of the week while realized volatility
  is not: the weekday/weekend residual SD ratio is 1.20 against a width ratio
  of 1.00, which shows up as over-coverage at weekends (99.3% Sunday against
  93.3% Tuesday). It errs in the safe direction, and it is second-order next to
  what was fixed, but a day-of-week or GARCH-type variance model is the honest
  next step.
- **Conformal coverage is validated, not guaranteed.** Split conformal
  guarantees coverage under *exchangeability* of the nonconformity scores.
  Scores from the same walk-forward fold share a training set, and neighbouring
  horizons share forecast error, so they are correlated: the effective sample
  size is below N and the textbook guarantee does not strictly apply. This is
  why the claim rests on the measured backtest coverage rather than on the
  theorem, and the theorem is only the reason to have expected it.
- **One seed, one world.** All figures describe a single realization. A
  production evaluation would repeat across seeds and report distributions.
- **The obligation structure is imposed.** The generator models costs as an
  aggregate outflow series; the split into payroll/rent/vendor obligations is a
  fixed chart-of-accounts assumption layered on top, so the optimizer's measured
  advantage depends on that structure being roughly right.
- **The optimizer did not beat the naive baseline in the backtest.** Reported in
  full in `BACKTEST_REPORT.md`. Solving the wrong problem precisely (an exact LP
  against an optimistic cash forecast) proved worse than solving roughly the
  right problem approximately.
