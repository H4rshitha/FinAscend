"""A.0 — Synthetic data generator with known ground truth.

There is no real bank data available, so this module is the foundation that
everything downstream is validated against. Its defining property is that the
*true* parameters are known and returned alongside the data: the delay
distribution for each counterparty is drawn from a distribution whose shape
and scale we chose. That lets A.2's fitting code be checked for whether it
**recovers** those parameters, which is what makes the statistics falsifiable
rather than decorative.

Design of the cash-flow series (composed, not noise with a label on it):

    level_t = trend_t * weekly_t * monthly_t * regime_t * noise_t

A multiplicative composition is used rather than additive because business
cash flows scale: a 20% bad quarter reduces a large business's inflows by more
rupees than a small one's, and seasonal amplitude grows with the level. An
additive model would impose constant absolute seasonality, which is wrong for
revenue series and would make the Holt-Winters `seasonal="mul"` option in A.1
untestable.

Two regimes are emitted:
  - EASY:        moderate dispersion, weak cross-counterparty correlation
  - ADVERSARIAL: heavy-tailed delays, strong correlation, deeper shocks

The adversarial regime exists so every downstream model is tested where it is
likely to fail, not only where it is likely to look good. A model selected
only on EASY data is selected on the easiest possible evidence.

All randomness flows from an explicit `numpy.random.Generator` seeded by the
caller. There are no module-level globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# Strength of the link between a counterparty's current slowness and its
# probability of defaulting on that invoice, on the log-odds scale. This is a
# property of the synthetic world, so it is a legitimate chosen constant — but
# it is the single most consequential one in the generator, because it sets the
# ceiling on how well ANY credit model in A.4 can possibly do. Stated here
# rather than buried so that A.4's reported AUC can be read against it.
DEFAULT_STRESS_BETA = 0.9

# AR(1) persistence of a counterparty's latent "slowness" state, week to week.
# Together with DEFAULT_STRESS_BETA this determines how much of A.4's
# prediction problem is actually learnable: persistence is what makes PAST
# payment behaviour informative about FUTURE default. At 0 the task is
# near-impossible by construction regardless of model quality.
DELAY_PERSISTENCE = 0.75


class Regime(str, Enum):
    """Difficulty setting for the generated world."""

    EASY = "easy"
    ADVERSARIAL = "adversarial"


@dataclass(frozen=True)
class CounterpartyTruth:
    """The *true* generating parameters for one counterparty.

    This is the ground truth that A.2's MLE fits are scored against. Nothing
    downstream is permitted to read these fields except tests and notebooks —
    the production path must rediscover them from the observed delays.
    """

    counterparty_id: str
    name: str
    # Delay distribution: scipy.stats.gamma(a=shape, scale=scale), in days.
    # Gamma is the true family throughout because it is the natural model for
    # a waiting time built from a sum of independent sub-delays (approval,
    # batching, bank transfer), each roughly exponential.
    delay_shape: float
    delay_scale: float
    default_probability: float
    # Relative size, centred on 1.0. Relative rather than absolute so that
    # changing the business's overall scale does not require re-picking every
    # counterparty's invoice size — the receipt stream stays calibrated.
    size_factor: float
    payment_terms_days: int

    @property
    def mean_delay_days(self) -> float:
        """Gamma mean = shape * scale."""
        return self.delay_shape * self.delay_scale


@dataclass(frozen=True)
class GeneratorTruth:
    """Everything the generator knows that the fitting code must rediscover."""

    regime: Regime
    seed: int
    counterparties: list[CounterpartyTruth]
    # True latent correlation between counterparty delays, in the Gaussian
    # copula's latent space. A.2 must recover this from ranks alone.
    delay_correlation: float
    trend_daily_growth: float
    weekly_amplitude: float
    monthly_amplitude: float
    shock_windows: list[tuple[date, date, float]] = field(default_factory=list)


@dataclass(frozen=True)
class SyntheticDataset:
    """Generated world: observable data plus the hidden truth behind it."""

    daily: pd.DataFrame           # date, inflow, outflow, net, balance
    payments: pd.DataFrame        # per-invoice payment records with realized delays
    truth: GeneratorTruth

    def observable(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return only what a real system could see. Guards against leakage."""
        return self.daily.copy(), self.payments.copy()


# Regime presets. These are the only "chosen" numbers in the Quant Core, and
# they are legitimate: they define the synthetic world rather than describing
# a real one. Everything downstream must be *derived* from data this produces.
_REGIME_PRESETS: dict[Regime, dict[str, float]] = {
    Regime.EASY: {
        "delay_shape_lo": 2.0,
        "delay_shape_hi": 6.0,
        "delay_scale_lo": 1.5,
        "delay_scale_hi": 3.5,
        "delay_correlation": 0.15,
        "default_prob_lo": 0.01,
        "default_prob_hi": 0.06,
        "noise_cv": 0.08,
        "shock_depth": 0.75,
        "n_shocks": 1.0,
        # Profitable and growing, with costs growing slightly slower.
        # Runway is long, and RaR should honestly say so.
        "net_margin": 0.06,
        "inflow_trend_pre": 0.00035,
        "inflow_trend_post": 0.00035,   # no regime change
        "decline_start_frac": 1.0,      # break never occurs
        "outflow_trend": 0.00030,
        "opening_months": 3.0,
    },
    Regime.ADVERSARIAL: {
        # Low shape => high skew and a fat right tail: a Gamma with shape 1.1
        # is nearly exponential, so extreme delays are far more likely.
        "delay_shape_lo": 1.1,
        "delay_shape_hi": 2.2,
        "delay_scale_lo": 4.0,
        "delay_scale_hi": 11.0,
        # Strong correlation is the whole point: it is what makes independent
        # sampling in A.2 dangerously optimistic.
        "delay_correlation": 0.72,
        "default_prob_lo": 0.05,
        "default_prob_hi": 0.20,
        "noise_cv": 0.22,
        "shock_depth": 0.55,
        "n_shocks": 2.0,
        # Starts modestly profitable, so the history is plausible — a business
        # cannot burn cash for three straight years and still be trading.
        # Revenue then declines while costs stay sticky, which is the classic
        # small-business failure mode and the reason runway becomes finite by
        # the end of the window without needing an absurd opening balance.
        "net_margin": 0.02,
        "inflow_trend_pre": 0.00010,    # mildly growing, plausible history
        # Sharp revenue decline after the break. Tuned so the balance
        # approaches zero near the end of the window without crossing it: a
        # negative cash balance is unphysical without an overdraft facility,
        # and a history containing one would be describing a business that had
        # already failed rather than one this product could still help.
        "inflow_trend_post": -0.00115,
        "decline_start_frac": 0.70,     # deterioration is RECENT, not chronic
        "outflow_trend": 0.00005,       # costs stay sticky throughout
        # A healthy 5-month buffer at the START. The business is not supposed
        # to begin in distress — it begins solvent and near break-even, and the
        # structural break burns the buffer down over the window. Anything
        # smaller and the balance crosses zero inside the history, which would
        # describe a firm that had already failed. Calibrated empirically
        # against the realized (not base) default rate of ~19%.
        "opening_months": 5.0,
    },
}


def make_counterparties(
    n: int,
    regime: Regime,
    rng: np.random.Generator,
) -> list[CounterpartyTruth]:
    """Draw `n` counterparties with known Gamma delay parameters.

    Parameters are drawn uniformly from the regime's range so that the test
    suite sees a spread of shapes rather than one repeated value — a fitter
    that only works at shape=3 would otherwise pass.

    Args:
        n: number of counterparties.
        regime: difficulty preset controlling dispersion and default rates.
        rng: seeded generator; all randomness flows from here.

    Returns:
        List of `CounterpartyTruth`, the ground truth for parameter recovery.
    """
    p = _REGIME_PRESETS[regime]
    out: list[CounterpartyTruth] = []
    for i in range(n):
        out.append(
            CounterpartyTruth(
                counterparty_id=f"CP{i:03d}",
                name=f"Counterparty {i:03d}",
                delay_shape=float(rng.uniform(p["delay_shape_lo"], p["delay_shape_hi"])),
                delay_scale=float(rng.uniform(p["delay_scale_lo"], p["delay_scale_hi"])),
                default_probability=float(rng.uniform(p["default_prob_lo"], p["default_prob_hi"])),
                # Centred on 1.0 (uniform 0.4-1.6) so the average across
                # counterparties leaves the receipt-stream calibration intact.
                size_factor=float(rng.uniform(0.4, 1.6)),
                payment_terms_days=int(rng.choice([15, 30, 45, 60])),
            )
        )
    return out


def _expected_default_probability(base_p: float, stress_beta: float, n_nodes: int = 40) -> float:
    """E_z[ sigmoid(logit(base_p) + beta*z) ] for z ~ N(0,1), by quadrature.

    Method: Gauss-Hermite quadrature. The integral is against a Gaussian
    weight, which is exactly what Gauss-Hermite is built for, so 40 nodes give
    effectively exact accuracy at negligible cost. Substitution: the physicists'
    convention integrates against exp(-x^2), so z = sqrt(2)*x and the weights
    are normalized by sqrt(pi).

    Needed because the realized default rate is NOT the base rate once default
    is coupled to stress — see the gross-up comment in `generate_dataset`.
    """
    if stress_beta <= 0:
        return base_p
    p = float(np.clip(base_p, 1e-6, 1 - 1e-6))
    logit_p = np.log(p / (1 - p))
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    z = np.sqrt(2.0) * nodes
    probs = 1.0 / (1.0 + np.exp(-(logit_p + stress_beta * z)))
    return float(np.sum(weights * probs) / np.sqrt(np.pi))


def _gaussian_copula_uniforms(
    n_samples: int,
    n_dims: int,
    correlation: float,
    rng: np.random.Generator,
    persistence: float = 0.0,
) -> np.ndarray:
    """Draw correlated uniforms on (0,1) via a Gaussian copula.

    Method: sample from a multivariate normal with equicorrelation matrix,
    then apply the probability integral transform through the standard normal
    CDF. The PIT step is what turns correlated normals into correlated
    *uniforms*, which can then be pushed through any marginal's inverse CDF.

    Omitting the PIT and using the normals directly is the classic copula bug:
    the dependence would be right but every marginal would be wrong.

    PERSISTENCE (serial correlation over time)
    ------------------------------------------
    With `persistence > 0` the latent state follows an AR(1) across rows:

        z_t = rho * z_{t-1} + sqrt(1 - rho^2) * e_t,    e_t ~ MVN(0, R)

    The sqrt(1 - rho^2) scaling is what keeps the marginal variance at exactly
    1, so the PIT still produces uniforms and the cross-sectional correlation
    R is unchanged. Getting that scaling wrong would inflate the marginals and
    silently break every downstream distribution fit.

    Why it is needed: without persistence, a counterparty's stress this week is
    independent of last week's, so its payment HISTORY carries no information
    about its CURRENT state. That caps every credit model in A.4 at roughly
    chance — measured at ROC-AUC 0.57, where the only learnable signal was the
    counterparty's own baseline default rate. Financial distress is persistent
    in reality: a firm paying late this month is likely to be late next month.
    Modelling that makes the prediction task genuine rather than decorative.

    Args:
        n_samples: rows to draw (time steps).
        n_dims: number of correlated dimensions (counterparties).
        correlation: cross-sectional pairwise latent correlation.
        rng: seeded generator.
        persistence: AR(1) coefficient in [0, 1). 0 recovers i.i.d. rows.

    Returns:
        Array of shape (n_samples, n_dims) with Uniform(0,1) marginals.
    """
    cov = np.full((n_dims, n_dims), correlation, dtype=float)
    np.fill_diagonal(cov, 1.0)
    innovations = rng.multivariate_normal(
        mean=np.zeros(n_dims), cov=cov, size=n_samples
    )

    if persistence <= 0.0:
        return stats.norm.cdf(innovations)

    rho = float(np.clip(persistence, 0.0, 0.99))
    scale = np.sqrt(1.0 - rho**2)
    z = np.empty_like(innovations)
    z[0] = innovations[0]
    for t in range(1, n_samples):
        z[t] = rho * z[t - 1] + scale * innovations[t]
    return stats.norm.cdf(z)


def generate_dataset(
    *,
    seed: int,
    regime: Regime = Regime.EASY,
    start: date = date(2022, 1, 1),
    n_days: int = 1095,
    n_counterparties: int = 12,
    base_daily_inflow: float = 180_000.0,
    net_margin: Optional[float] = None,
    opening_balance_months: float = 3.0,
    opening_balance: Optional[float] = None,
    stress_beta: float = DEFAULT_STRESS_BETA,
) -> SyntheticDataset:
    """Generate a multi-year daily cash-flow history with known ground truth.

    Composition (multiplicative — see module docstring for why):
        inflow_t = base * trend_t * weekly_t * monthly_t * regime_t * noise_t

    Per-counterparty payment delays are drawn through a Gaussian copula so
    that slowdowns are *correlated* across counterparties. This is the
    property that makes A.2's independence assumption testably wrong.

    SOLVENCY IS A PARAMETER, NOT AN ACCIDENT
    ----------------------------------------
    Outflow is derived from inflow via `net_margin` rather than set as an
    independent constant. An earlier version fixed both levels separately,
    which made solvency an uncontrolled by-product of the seasonal and shock
    multipliers: the EASY regime compounded into a business so cash-rich it
    could never run out (RaR pinned at the horizon), while ADVERSARIAL drove
    the balance negative *within the historical window* — a business that had
    already failed. Both are degenerate for a liquidity model, which is only
    interesting when runway is finite and positive.

    `net_margin` is a genuine economic quantity (operating margin), and the
    resulting runway is an emergent consequence of it, the shocks, and the
    payment delays — not a number chosen to look reasonable.

    Args:
        seed: master seed. The same seed always produces the same world.
        regime: EASY or ADVERSARIAL.
        start: first calendar day.
        n_days: length of the history (default ~3 years, enough for SARIMAX
            to see three full annual cycles).
        n_counterparties: number of distinct payers.
        base_daily_inflow: pre-seasonality mean daily inflow.
        net_margin: (inflow - outflow) / inflow. Positive = profitable.
            Defaults to the regime preset: mildly profitable for EASY, mildly
            loss-making for ADVERSARIAL.
        opening_balance_months: starting cash expressed in months of baseline
            outflow. Scale-free, so changing `base_daily_inflow` does not
            silently change how much runway the business starts with.
        opening_balance: absolute override; wins over `opening_balance_months`.
        stress_beta: strength of the link between a counterparty's current
            slowness and its probability of defaulting on that invoice.
            Exposed as a parameter (not just the module constant) so tests can
            set it to 0 and isolate the pure delay distribution — with the
            coupling ON, observed delays are censored (see below) and no
            longer follow the unconditional Gamma.

    SURVIVORSHIP BIAS IN OBSERVED DELAYS
    ------------------------------------
    Because default probability rises with stress, and stress is what produces
    long delays, defaulted invoices are disproportionately the SLOW ones. Only
    paid invoices yield an observed delay, so the observed delay distribution
    is the true Gamma with its right tail preferentially removed — it is
    P(delay | eventually paid), not P(delay).

    This is not a defect to correct. For simulating cash RECEIPTS it is the
    correct conditional distribution, since a defaulted invoice contributes no
    cash on any date; default is modelled separately by A.4. But it does mean
    A.2's fitted marginals should be read as conditional-on-payment, and the
    generator's `delay_shape`/`delay_scale` are recoverable only up to that
    censoring. `stress_beta=0` removes it for testing.

    Returns:
        `SyntheticDataset` with `daily`, `payments`, and `truth`.
    """
    rng = np.random.default_rng(seed)
    preset = _REGIME_PRESETS[regime]
    counterparties = make_counterparties(n_counterparties, regime, rng)

    if net_margin is None:
        net_margin = float(preset["net_margin"])
    base_daily_outflow = base_daily_inflow * (1.0 - net_margin)
    if opening_balance is None:
        # The regime preset wins unless the caller explicitly overrides, since
        # the two regimes need very different buffers to stay non-degenerate.
        months = (
            opening_balance_months
            if opening_balance_months != 3.0
            else float(preset["opening_months"])
        )
        opening_balance = float(months) * 30.0 * base_daily_outflow

    dates = pd.date_range(start=start, periods=n_days, freq="D")
    t = np.arange(n_days, dtype=float)

    # --- trend: revenue and costs move on SEPARATE trends ---
    # Costs are sticky: rent, payroll and loan EMIs do not shrink when revenue
    # does. Giving outflow its own trend is what allows the margin to
    # deteriorate, which is the mechanism that produces a finite forward
    # runway. A shared trend keeps inflow and outflow locked in proportion, so
    # the business could never actually get into trouble.
    #
    # Revenue trend is PIECEWISE, with a structural break. A constant negative
    # margin sustained across a three-year history compounds into a deficit no
    # real business could survive, and forcing the balance to stay positive
    # under one would require an implausible opening cash pile. Real distress
    # is recent: the firm traded near break-even, then something changed. The
    # break also gives A.1 an honestly hard forecasting problem, since a
    # structural break is exactly what extrapolative models handle worst.
    pre_growth = float(preset["inflow_trend_pre"])
    post_growth = float(preset["inflow_trend_post"])
    t_break = float(preset["decline_start_frac"]) * n_days

    trend = np.where(
        t < t_break,
        np.exp(pre_growth * t),
        np.exp(pre_growth * t_break) * np.exp(post_growth * (t - t_break)),
    )
    trend_daily_growth = pre_growth
    outflow_trend_growth = float(preset["outflow_trend"])
    outflow_trend = np.exp(outflow_trend_growth * t)

    # --- weekly seasonality: business days carry more volume than weekends ---
    weekly_amplitude = 0.35
    dow = dates.dayofweek.to_numpy()
    weekly = 1.0 + weekly_amplitude * np.where(dow >= 5, -1.0, 0.4)

    # --- monthly seasonality: month-end settlement spike ---
    monthly_amplitude = 0.28
    dom = dates.day.to_numpy()
    days_in_month = dates.days_in_month.to_numpy()
    monthly = 1.0 + monthly_amplitude * np.cos(
        2.0 * np.pi * (dom - 1) / np.maximum(days_in_month - 1, 1)
    )

    # --- regime shocks: bad quarters that scale the level down ---
    regime_mult = np.ones(n_days)
    shock_windows: list[tuple[date, date, float]] = []
    n_shocks = int(preset["n_shocks"])
    for _ in range(n_shocks):
        if n_days < 140:
            break
        s = int(rng.integers(60, n_days - 100))
        length = int(rng.integers(45, 100))
        depth = float(preset["shock_depth"])
        regime_mult[s : s + length] *= depth
        shock_windows.append(
            (dates[s].date(), dates[min(s + length, n_days - 1)].date(), depth)
        )

    # --- multiplicative noise, lognormal so the level stays positive ---
    cv = preset["noise_cv"]
    sigma = np.sqrt(np.log(1.0 + cv**2))
    noise_in = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=n_days)
    noise_out = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=n_days)

    # Outflows are far less seasonal — rent and payroll do not care what day
    # of the week it is. Damping the seasonal factors reflects that. Note also
    # that `regime_mult` is deliberately absent: a demand shock cuts revenue,
    # it does not cut the rent. That asymmetry is what makes a shock a
    # liquidity event rather than a neutral rescaling.
    outflow = base_daily_outflow * outflow_trend * (1.0 + 0.25 * (weekly - 1.0)) * noise_out

    # ------------------------------------------------------------------
    # INFLOW IS BUILT FROM THE INVOICES, NOT ALONGSIDE THEM
    # ------------------------------------------------------------------
    # An earlier version generated the daily inflow series and the per-invoice
    # payment records as two independent processes. That is incoherent: in a
    # real business the bank inflows ARE the invoice payments arriving. The
    # incoherence was not cosmetic — A.2 forecasts the daily series and then
    # adds simulated receivable arrivals on top, so two unrelated streams
    # meant tens of millions of phantom cash and a business that could never
    # run out. RaR was pinned at the horizon in every configuration.
    #
    # Now: invoices are issued on a seasonal schedule, each is paid on
    # due_date + a copula-drawn delay, and `receipts_t` is the sum of whatever
    # actually lands on day t. A small `cash_sales` component covers
    # over-the-counter revenue that never becomes a receivable, so the
    # forecastable and the delay-driven parts of inflow are cleanly separated.
    invoice_share = 0.75
    target_invoice_daily = base_daily_inflow * invoice_share
    # Weekly issuance per counterparty => n_counterparties invoices per 7 days.
    # Mean amount is DERIVED so the resulting receipt stream matches the target
    # level, rather than being an arbitrary rupee figure.
    #
    # Grossed up for defaults: a fraction of invoices never pay, so billing
    # exactly the target would leave receipts systematically short of it. The
    # business must bill 1/(1-p_default) to collect the target.
    # The gross-up must use the REALIZED default rate, not the base rate.
    # Default probability is sigmoid(logit(p_i) + beta*z) with z ~ N(0,1), and
    # the sigmoid is convex in the small-p region, so by Jensen's inequality
    # E[sigmoid(...)] > p_i. Grossing up by the base rate therefore
    # under-collects, and the shortfall compounds: it drove the adversarial
    # balance 11M negative inside the history. The expectation is computed by
    # Gauss-Hermite quadrature rather than simulated, so it is exact and adds
    # no sampling noise to the calibration.
    mean_default = float(
        np.mean([
            _expected_default_probability(c.default_probability, stress_beta)
            for c in counterparties
        ])
    )
    mean_invoice = (
        target_invoice_daily * 7.0 / max(n_counterparties, 1) / (1.0 - mean_default)
    )

    # WARM-UP: invoices must exist before day 0.
    # Without this, no invoice has had time to be issued and paid when the
    # series starts, so receipts ramp up from zero over roughly
    # (payment_terms + delay) days while costs run at full rate. That is a pure
    # edge artifact of where we chose to start the window, and it burned ~8M of
    # opening cash before the simulation had said anything. Invoices are
    # therefore issued from `warmup_days` before `start`, and the reported
    # series is sliced to begin once the receipt stream is in steady state.
    warmup_days = 120
    warm_dates = pd.date_range(start=start - timedelta(days=warmup_days),
                               periods=warmup_days, freq="D")
    all_dates = warm_dates.append(dates)
    # Level during warm-up is held at its day-0 value: extrapolating the trend
    # backwards would invent history the series does not claim to have.
    warm_level = np.concatenate([np.full(warmup_days, (trend * monthly * regime_mult)[0]),
                                 trend * monthly * regime_mult])

    payments = _generate_payments(
        dates=all_dates,
        counterparties=counterparties,
        correlation=float(preset["delay_correlation"]),
        rng=rng,
        mean_invoice=mean_invoice,
        level=warm_level,
        stress_beta=stress_beta,
    )

    receipts = np.zeros(n_days)
    date_to_idx = {d: i for i, d in enumerate(dates)}
    for paid_date, amt in zip(payments["paid_date"], payments["amount"]):
        if paid_date is None or pd.isna(paid_date):
            continue
        idx = date_to_idx.get(pd.Timestamp(paid_date).normalize())
        if idx is not None:
            receipts[idx] += float(amt)

    cash_sales = (
        base_daily_inflow * (1.0 - invoice_share)
        * trend * weekly * monthly * regime_mult * noise_in
    )
    inflow = receipts + cash_sales

    net = inflow - outflow
    balance = opening_balance + np.cumsum(net)

    # The series A.2 must forecast. `net` already contains receipts, so
    # forecasting `net` and THEN adding simulated receivable arrivals would
    # double-count every rupee of receivable. `net_ex_receipts` is the
    # operating residual (cash sales minus costs); the receivable stream is
    # supplied separately by the copula simulation. Keeping both columns makes
    # the decomposition explicit rather than implicit in caller conventions.
    net_ex_receipts = cash_sales - outflow

    daily = pd.DataFrame(
        {
            "date": dates,
            "inflow": inflow,
            "receipts": receipts,
            "cash_sales": cash_sales,
            "outflow": outflow,
            "net": net,
            "net_ex_receipts": net_ex_receipts,
            "balance": balance,
        }
    )

    truth = GeneratorTruth(
        regime=regime,
        seed=seed,
        counterparties=counterparties,
        delay_correlation=float(preset["delay_correlation"]),
        trend_daily_growth=trend_daily_growth,
        weekly_amplitude=weekly_amplitude,
        monthly_amplitude=monthly_amplitude,
        shock_windows=shock_windows,
    )
    return SyntheticDataset(daily=daily, payments=payments, truth=truth)


def _generate_payments(
    *,
    dates: pd.DatetimeIndex,
    counterparties: list[CounterpartyTruth],
    correlation: float,
    rng: np.random.Generator,
    mean_invoice: float,
    level: np.ndarray,
    stress_beta: float = DEFAULT_STRESS_BETA,
) -> pd.DataFrame:
    """Generate per-invoice payment records with correlated realized delays.

    Each invoice's delay is produced by pushing a copula uniform through that
    counterparty's Gamma inverse CDF (`stats.gamma.ppf`). Because the uniforms
    are correlated across counterparties within an issue-week, a bad week is
    bad for everyone at once — the joint behaviour that independent sampling
    cannot reproduce.

    Returns:
        DataFrame with issue_date, due_date, counterparty_id, amount,
        delay_days, paid (bool), paid_date, days_overdue.
    """
    n_cp = len(counterparties)
    # Invoices are issued weekly per counterparty; one copula draw per week
    # couples that week's delays across all counterparties.
    issue_idx = np.arange(0, len(dates), 7)
    n_weeks = len(issue_idx)

    uniforms = _gaussian_copula_uniforms(
        n_weeks, n_cp, correlation, rng, persistence=DELAY_PERSISTENCE
    )

    rows = []
    for w, di in enumerate(issue_idx):
        for j, cp in enumerate(counterparties):
            # Jitter the issue date within the week, per counterparty. Weekly
            # issuance on a fixed day made `days_since_last_invoice` exactly 7
            # for every row — a constant column, which becomes all-zero after
            # standardization and makes the logistic information matrix
            # singular, so every Wald confidence interval came back NaN.
            # Irregular billing is also simply more realistic.
            offset = int(rng.integers(0, 7))
            di_j = min(di + offset, len(dates) - 1)
            issue_date = dates[di_j]
            u = float(np.clip(uniforms[w, j], 1e-9, 1 - 1e-9))
            # PIT inversion: uniform -> Gamma delay with this counterparty's
            # true parameters.
            delay = float(stats.gamma.ppf(u, a=cp.delay_shape, scale=cp.delay_scale))
            # Invoice size carries the trend/seasonal/shock level at issue
            # time, so a demand shock shrinks what is billed and therefore what
            # is later received — the mechanism by which a shock becomes a
            # liquidity event weeks after it happens.
            #
            # The lognormal is parameterized so its MEAN is the target, not its
            # median: E[lognormal] = exp(mu + sigma^2/2), so mu must be
            # log(target) - sigma^2/2. Using log(target) directly would inflate
            # average invoice size by exp(sigma^2/2) ~ 11% at sigma=0.45 and
            # silently break the calibration of the receipt stream.
            sigma_amt = 0.45
            target = mean_invoice * cp.size_factor * float(level[di_j])
            amount = float(
                rng.lognormal(mean=np.log(target) - 0.5 * sigma_amt**2, sigma=sigma_amt)
            )
            # Default depends on the counterparty's CURRENT stress, not on an
            # i.i.d. coin flip. `u` is the copula draw that produced this
            # invoice's delay, so a high u means this counterparty is having a
            # slow period — and a firm that is paying late is empirically more
            # likely to eventually not pay at all. Both symptoms share a cause.
            #
            # This matters for A.4: with an i.i.d. default flip there is no
            # relationship between observable payment behaviour and default,
            # so the best achievable ROC-AUC is near 0.5 no matter how good the
            # model is. Measured directly — the first version of this generator
            # capped every model at ~0.57 AUC and the "model beats baseline"
            # comparison was measuring noise. Linking the two makes the
            # learning task real rather than decorative.
            stress = float(stats.norm.ppf(u))          # latent z for this draw
            logit_p = np.log(cp.default_probability / (1 - cp.default_probability))
            p_default = 1.0 / (1.0 + np.exp(-(logit_p + stress_beta * stress)))
            defaulted = bool(rng.random() < p_default)
            due_date = issue_date + timedelta(days=cp.payment_terms_days)
            paid_date = None if defaulted else due_date + timedelta(days=float(delay))
            rows.append(
                {
                    "invoice_id": f"INV{w:04d}-{cp.counterparty_id}",
                    "counterparty_id": cp.counterparty_id,
                    "issue_date": issue_date,
                    "due_date": due_date,
                    "amount": amount,
                    "delay_days": delay,
                    "paid": not defaulted,
                    "paid_date": paid_date,
                }
            )

    df = pd.DataFrame(rows)
    df["days_overdue"] = np.where(df["paid"], df["delay_days"], np.nan)
    return df


def observed_delays_by_counterparty(payments: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract per-counterparty realized delays — the input A.2 fits from.

    Only paid invoices contribute a delay. Defaults are censored observations,
    not delays of infinite length, and silently treating them as either would
    bias the fit. They are handled separately by A.4's default model.

    Args:
        payments: the `payments` frame from `generate_dataset`.

    Returns:
        Mapping counterparty_id -> array of realized delay days.
    """
    out: dict[str, np.ndarray] = {}
    paid = payments[payments["paid"]]
    for cp_id, grp in paid.groupby("counterparty_id"):
        out[str(cp_id)] = grp["delay_days"].to_numpy(dtype=float)
    return out


def delay_panel(payments: pd.DataFrame) -> pd.DataFrame:
    """Pivot payments into a time-aligned delay panel (issue_date x counterparty).

    Correlation between counterparties MUST be estimated from this, not from
    the ragged arrays that `observed_delays_by_counterparty` returns.

    Why: defaulted invoices produce no delay observation, and different
    counterparties default in different weeks. Dropping them leaves each
    counterparty with a different-length array whose k-th element refers to a
    different week than another counterparty's k-th element. A single dropped
    row shifts every subsequent entry by one, so an element-wise correlation
    ends up comparing week 5 against week 6 and measures approximately
    nothing. Measured empirically: ragged arrays gave a correlation estimate
    of 0.02 against a true latent value of 0.15.

    The panel keeps the shared `issue_date` index and inserts NaN where a
    payment never arrived, so pairwise-complete correlation compares the same
    week to the same week.

    Returns:
        DataFrame indexed by issue_date, one column per counterparty, values
        are realized delay days with NaN for defaults.
    """
    paid = payments[payments["paid"]]
    return paid.pivot_table(
        index="issue_date",
        columns="counterparty_id",
        values="delay_days",
        aggfunc="mean",
    )
