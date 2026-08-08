"""A.2 — Monte Carlo liquidity risk engine, producing Runway-at-Risk.

This module is the headline artifact of the project. It answers: *given what
we know about how our counterparties actually pay, how likely are we to run
out of cash, and how soon?*

Three pieces, in order:

1. **Fit a real delay distribution per counterparty** (`fit_delay_distribution`).
   Candidates Gamma, Weibull and Log-normal are fitted by maximum likelihood
   via `scipy.stats`, scored with a Kolmogorov-Smirnov test, and ranked by
   AIC. Every candidate is retained with its statistics, so "we chose Weibull"
   is always accompanied by what Gamma and Log-normal scored.

2. **Couple them with a copula** (`sample_correlated_delays`). Counterparty
   delays are *not* independent, and assuming they are is the single most
   dangerous simplification available here — see the extended note in that
   function's docstring.

3. **Simulate forward and read off RaR / CRaR** (`run_simulation`). RaR is the
   liquidity analogue of Value-at-Risk; CRaR is the expected-shortfall
   analogue. Both are reported, because RaR alone tells you where the cliff
   is but nothing about how far the drop goes.

Every entry point takes an explicit seed. There is no module-level RNG.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from app.schemas.quant import (
    ConvergencePoint,
    CopulaFamily,
    CopulaSpec,
    CounterpartyDelayFit,
    DistributionFamily,
    GoodnessOfFit,
    RunwayAtRisk,
    UncertaintyModelSpec,
)

# scipy frozen-distribution constructors for each candidate family. All three
# are supported on [0, inf) and all three are two-parameter shape/scale
# families once location is pinned at 0 (see `_fit_one` for why we pin it).
_FAMILY_DISTS = {
    DistributionFamily.GAMMA: stats.gamma,
    DistributionFamily.WEIBULL: stats.weibull_min,
    DistributionFamily.LOGNORMAL: stats.lognorm,
}


# ---------------------------------------------------------------------------
# 1. Marginal fitting
# ---------------------------------------------------------------------------

def _fit_one(
    family: DistributionFamily, delays: np.ndarray
) -> tuple[dict[str, float], float, float, float]:
    """MLE-fit one family to observed delays and score the fit.

    Method: `scipy.stats.<dist>.fit(data, floc=0)` — maximum likelihood with
    the location parameter **pinned at zero**. Pinning matters: left free,
    scipy will happily slide `loc` to just below the sample minimum, which
    inflates the likelihood, produces a distribution with zero density below
    that point, and makes the KS test meaninglessly good. A payment delay of
    zero days is physically possible, so loc=0 is the honest constraint.

    Returns:
        (params, ks_statistic, ks_pvalue, log_likelihood)
    """
    dist = _FAMILY_DISTS[family]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = dist.fit(delays, floc=0.0)

    frozen = dist(*fitted)
    ks_stat, ks_p = stats.kstest(delays, frozen.cdf)
    loglik = float(np.sum(frozen.logpdf(delays)))

    if family is DistributionFamily.GAMMA:
        params = {"shape": float(fitted[0]), "loc": float(fitted[1]), "scale": float(fitted[2])}
    elif family is DistributionFamily.WEIBULL:
        params = {"c": float(fitted[0]), "loc": float(fitted[1]), "scale": float(fitted[2])}
    else:  # LOGNORMAL
        params = {"s": float(fitted[0]), "loc": float(fitted[1]), "scale": float(fitted[2])}

    return params, float(ks_stat), float(ks_p), loglik


def fit_delay_distribution(
    counterparty_id: str,
    delays: np.ndarray,
    *,
    expected_delay_threshold: float = 0.0,
) -> CounterpartyDelayFit:
    """Fit Gamma / Weibull / Log-normal to one counterparty's delays and select.

    Selection is by **AIC**, not by KS p-value. Two reasons: (a) a KS p-value
    is a test of "is this distribution refuted", not "is it best" — with 150
    observations several families are un-refuted at once; (b) p-values are not
    comparable as a ranking because they are not monotone in fit quality. AIC
    penalizes by parameter count, and all three families here have the same
    count, so AIC reduces to a likelihood comparison on equal footing.

    The KS statistic is still computed and stored, because it answers a
    different and necessary question: is the *winner* actually adequate, or
    merely the least-bad of three bad options?

    TWO CAVEATS ON THE KS P-VALUE, both of which make it CONSERVATIVE here:

    1. **Serial correlation.** The KS test assumes i.i.d. observations.
       Payment delays are persistent — a counterparty slow this month tends to
       be slow next month — which shrinks the effective sample size and makes
       the test over-reject a distribution that is actually correct. Measured
       on synthetic data with a known-correct Gamma, the median p-value across
       counterparties rose from 0.03 to 0.63 simply by thinning the sample to
       remove the autocorrelation. A low p-value here is therefore weaker
       evidence against the fit than it appears.

    2. **Estimated parameters.** The distribution is fitted to the same data
       the test is run on, which makes the standard KS null distribution
       invalid (a Lilliefors-type correction would be needed). This biases the
       p-value UP, partially offsetting caveat 1.

    Both are stated rather than corrected: the p-value is used as a relative
    adequacy signal, not as a formal hypothesis test, and the selection itself
    rests on AIC.

    Args:
        counterparty_id: identifier carried into the output.
        delays: observed realized delays in days, non-negative.
        expected_delay_threshold: delay at or below which a payment counts as
            "on time"; feeds `prob_on_time`, which replaces the struck
            `Inflow.certainty = 1.0` default.

    Returns:
        `CounterpartyDelayFit` with every candidate scored, winner marked.
    """
    delays = np.asarray(delays, dtype=float)
    delays = delays[np.isfinite(delays) & (delays >= 0)]
    if len(delays) < 10:
        raise ValueError(
            f"{counterparty_id}: {len(delays)} usable delays; need >= 10 to fit "
            "a two-parameter distribution with any credibility"
        )

    candidates: list[GoodnessOfFit] = []
    for family in _FAMILY_DISTS:
        try:
            params, ks_stat, ks_p, loglik = _fit_one(family, delays)
        except Exception:
            continue
        # AIC = 2k - 2 log L. k=2 free parameters (shape, scale); loc is pinned.
        aic = 2 * 2 - 2 * loglik
        candidates.append(
            GoodnessOfFit(
                family=family,
                params=params,
                ks_statistic=ks_stat,
                ks_pvalue=ks_p,
                log_likelihood=loglik,
                aic=aic,
            )
        )

    if not candidates:
        raise RuntimeError(f"{counterparty_id}: no candidate family could be fitted")

    best = min(candidates, key=lambda c: c.aic)
    scored = []
    for c in candidates:
        scored.append(c.model_copy(update={"selected": c.family == best.family}))

    dist = _FAMILY_DISTS[DistributionFamily(best.family)]
    frozen = dist(**_scipy_kwargs(DistributionFamily(best.family), best.params))
    prob_on_time = float(frozen.cdf(expected_delay_threshold)) if expected_delay_threshold > 0 else float(frozen.cdf(0.0))

    ranking = ", ".join(
        f"{c.family}: AIC={c.aic:,.1f}, KS D={c.ks_statistic:.4f} (p={c.ks_pvalue:.3f})"
        for c in sorted(candidates, key=lambda c: c.aic)
    )
    adequacy = (
        "winner is not refuted by KS at the 5% level"
        if best.ks_pvalue > 0.05
        else f"WARNING: winner is REFUTED by KS (p={best.ks_pvalue:.4f}); "
        "the true delay law is probably outside this candidate set"
    )
    rationale = (
        f"Selected {best.family} by lowest AIC on n={len(delays)} observed delays. "
        f"Ranking — {ranking}. Adequacy check: {adequacy}. "
        "AIC (not KS p-value) drives selection because p-values test refutation, "
        "not relative fit, and are not monotone in fit quality."
    )

    return CounterpartyDelayFit(
        counterparty_id=counterparty_id,
        n_observations=len(delays),
        candidates=scored,
        selected_family=DistributionFamily(best.family),
        selected_params=best.params,
        selection_rationale=rationale,
        prob_on_time=prob_on_time,
    )


def _scipy_kwargs(family: DistributionFamily, params: dict[str, float]) -> dict[str, float]:
    """Map stored parameter names back onto scipy's argument names."""
    if family is DistributionFamily.GAMMA:
        return {"a": params["shape"], "loc": params["loc"], "scale": params["scale"]}
    if family is DistributionFamily.WEIBULL:
        return {"c": params["c"], "loc": params["loc"], "scale": params["scale"]}
    return {"s": params["s"], "loc": params["loc"], "scale": params["scale"]}


def frozen_from_fit(fit: CounterpartyDelayFit):
    """Rebuild the selected frozen scipy distribution from a stored fit."""
    family = DistributionFamily(fit.selected_family)
    return _FAMILY_DISTS[family](**_scipy_kwargs(family, fit.selected_params))


# ---------------------------------------------------------------------------
# 2. Dependence
# ---------------------------------------------------------------------------

def estimate_delay_correlation(
    panel: pd.DataFrame,
    order: Sequence[str],
) -> np.ndarray:
    """Estimate the copula correlation matrix from **rank** correlation.

    Method: Spearman rank correlation, converted to the Gaussian-copula linear
    parameter by the standard identity

        rho_pearson = 2 * sin(pi * rho_spearman / 6)

    (Kruskal 1958; see also McNeil, Frey & Embrechts, *Quantitative Risk
    Management*, §5.3.) Rank correlation is used rather than raw Pearson
    correlation of the delays because Pearson measures *linear* association
    and is distorted by the heavy right tail of a Gamma delay: one enormous
    delay can dominate the estimate. Ranks are invariant to the marginal
    transform, which is exactly the property a copula parameter needs.

    ALIGNMENT
    ---------
    Input is a **time-aligned panel** (see `synthetic_data.delay_panel`), not
    a dict of ragged arrays. This is not a stylistic preference: defaulted
    invoices leave gaps at different dates for different counterparties, so
    ragged arrays silently misalign and the estimate collapses toward zero.
    That failure was measured at 0.02 recovered against 0.15 true. Pairwise
    complete observations are used, so a gap costs one week rather than
    corrupting the rest of the series.

    Args:
        panel: DataFrame indexed by date, one column per counterparty, NaN
            where no payment was observed.
        order: column order for the returned matrix.

    Returns:
        Positive-definite correlation matrix in `order`, ready for the copula.
    """
    n = len(order)
    rho = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if order[i] not in panel.columns or order[j] not in panel.columns:
                continue
            pair = panel[[order[i], order[j]]].dropna()
            if len(pair) < 5:
                continue
            rs, _ = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            if not np.isfinite(rs):
                continue
            pearson = 2.0 * np.sin(np.pi * float(rs) / 6.0)
            rho[i, j] = rho[j, i] = pearson

    return _nearest_positive_definite(rho)


def _nearest_positive_definite(mat: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the nearest positive-definite one.

    Pairwise-estimated correlation matrices are frequently not positive
    semi-definite — each entry is estimated from a different subsample, so
    they need not cohere. Cholesky then fails. Clipping the eigenvalues at a
    small positive floor and renormalizing to unit diagonal is the standard
    repair (Higham 2002, simplified).
    """
    sym = (mat + mat.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, 1e-8, None)
    rebuilt = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(rebuilt))
    return rebuilt / np.outer(d, d)


def sample_correlated_delays(
    fits: Sequence[CounterpartyDelayFit],
    correlation: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    *,
    family: CopulaFamily = CopulaFamily.STUDENT_T,
    df: float = 4.0,
) -> np.ndarray:
    """Draw correlated payment delays via a copula.

    WHY NOT INDEPENDENT SAMPLING
    ----------------------------
    Sampling each counterparty's delay independently is the intuitive default
    and it is badly wrong for this problem, in a direction that flatters the
    business.

    Under independence, the probability that all 12 counterparties are
    simultaneously slow is the product of 12 individual probabilities. If each
    is slow 20% of the time, that joint event has probability 0.2^12 ~ 4e-9 —
    effectively never. So the simulated worst case is "a few counterparties
    are late", the cash trough is shallow, and the reported runway looks
    comfortable.

    In reality those events are driven by common causes: a demand downturn, a
    credit squeeze, a sector-wide slowdown, a festival period. When one payer
    slows, others usually are too. The empirical joint probability of a
    system-wide slowdown is orders of magnitude above 4e-9 — and a system-wide
    slowdown is precisely the event that empties the account, because it
    removes every inflow at once with no offsetting early payment.

    Independent sampling therefore produces a comfortable tail that does not
    exist, and RaR computed from it is not conservative — it is fiction. The
    copula restores the dependence.

    METHOD
    ------
    Student-t copula by default (Gaussian available). The t copula is
    preferred because it exhibits **tail dependence**: the Gaussian copula has
    zero asymptotic tail dependence, meaning that however high you set its
    correlation, extreme joint events still decouple in the far tail. That is
    the wrong asymptotic behaviour for exactly the scenario we care about. The
    t copula with low `df` keeps counterparties coupled in the tail.

    Sampling proceeds in three steps, and the middle one is the one that is
    easy to get wrong:

        1. Draw correlated latent variates  z ~ MVN(0, R)   [or t_df]
        2. **Probability integral transform**: u = F_latent(z)  ->  Uniform(0,1)
        3. Invert through each counterparty's own fitted marginal:
           delay_i = F_i^{-1}(u_i)

    Skipping step 2 and treating the latent draws as delays would give the
    right correlation and completely wrong marginals — the delays would be
    normally distributed, symmetric, and able to go negative, rather than
    following the fitted Gamma/Weibull/Log-normal. `test_copula_marginals`
    exists specifically to catch that, independently of any correlation test,
    because a correlation test passes even when the transform is missing.

    Args:
        fits: per-counterparty fitted marginals, defining column order.
        correlation: latent correlation matrix, same order as `fits`.
        n_samples: rows to draw.
        rng: seeded generator.
        family: GAUSSIAN or STUDENT_T.
        df: t degrees of freedom; ignored for Gaussian. Lower = fatter joint tails.

    Returns:
        Array (n_samples, len(fits)) of delays in days, each column following
        its counterparty's fitted marginal.
    """
    n_dim = len(fits)
    chol = np.linalg.cholesky(correlation)
    z = rng.standard_normal(size=(n_samples, n_dim)) @ chol.T

    if family is CopulaFamily.STUDENT_T:
        # t = z / sqrt(chi2_df / df), giving multivariate-t latents.
        chi2 = rng.chisquare(df, size=(n_samples, 1))
        t_latent = z / np.sqrt(chi2 / df)
        # STEP 2 — PIT through the t CDF (must match the latent's law).
        u = stats.t.cdf(t_latent, df=df)
    else:
        # STEP 2 — PIT through the standard normal CDF.
        u = stats.norm.cdf(z)

    u = np.clip(u, 1e-10, 1.0 - 1e-10)

    # STEP 3 — invert through each counterparty's own fitted marginal.
    out = np.empty_like(u)
    for j, fit in enumerate(fits):
        out[:, j] = frozen_from_fit(fit).ppf(u[:, j])
    return out


# ---------------------------------------------------------------------------
# 3. Simulation and Runway-at-Risk
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationPaths:
    """Raw simulated output, kept so downstream code can reuse the draws."""

    balances: np.ndarray        # (n_iterations, horizon_days)
    days_to_zero: np.ndarray    # (n_iterations,) censored at horizon
    hit_zero: np.ndarray        # (n_iterations,) bool
    horizon_days: int
    seed: int


def simulate_cash_paths(
    *,
    opening_balance: float,
    forecast_point: np.ndarray,
    forecast_sigma: np.ndarray,
    receivables: pd.DataFrame,
    fits: Sequence[CounterpartyDelayFit],
    correlation: np.ndarray,
    n_iterations: int,
    horizon_days: int,
    seed: int,
    copula_family: CopulaFamily = CopulaFamily.STUDENT_T,
    copula_df: float = 4.0,
) -> SimulationPaths:
    """Simulate forward cash balances under joint forecast + delay uncertainty.

    Two independent sources of randomness are combined, which is the point:

      * **Forecast uncertainty** — the operating net-flow path is drawn around
        A.1's point forecast using its prediction-interval width. Treating the
        forecast as a known constant and only simulating receivable timing
        would understate risk by ignoring the larger of the two uncertainties.
      * **Receivable timing** — each outstanding receivable arrives on its due
        date plus a copula-drawn delay, so slowdowns are correlated.

    Args:
        opening_balance: cash at t=0.
        forecast_point: (horizon_days,) point forecast of daily net operating flow.
        forecast_sigma: (horizon_days,) implied std dev per step, derived from
            the prediction interval by `sigma = (upper - lower) / (2 * z)`.
        receivables: frame with `counterparty_id`, `amount`, `days_until_due`.
        fits: fitted marginals; column order defines the copula dimension order.
        correlation: latent correlation matrix.
        n_iterations: Monte Carlo draws. Justified by `convergence_study`.
        horizon_days: simulation length.
        seed: master seed.

    Returns:
        `SimulationPaths` with balances, censored days-to-zero, and hit flags.
    """
    rng = np.random.default_rng(seed)
    cp_index = {f.counterparty_id: j for j, f in enumerate(fits)}

    # --- operating flow: point path + correlated-in-time noise ---
    # Noise is i.i.d. across days here. Serial correlation in the *residual*
    # would matter, but SARIMAX has already absorbed the autocorrelation it
    # could find; what remains is closer to white noise by construction.
    shocks = rng.standard_normal(size=(n_iterations, horizon_days))
    operating = forecast_point[None, :] + shocks * forecast_sigma[None, :]

    # --- receivable arrivals: correlated delays ---
    delays = sample_correlated_delays(
        fits, correlation, n_iterations, rng, family=copula_family, df=copula_df
    )

    receipts = np.zeros((n_iterations, horizon_days))
    for _, row in receivables.iterrows():
        cp = str(row["counterparty_id"])
        if cp not in cp_index:
            continue
        j = cp_index[cp]
        arrival = np.asarray(row["days_until_due"], dtype=float) + delays[:, j]
        idx = np.floor(arrival).astype(int)
        inside = (idx >= 0) & (idx < horizon_days)
        # np.add.at handles repeated indices correctly; plain fancy-index
        # assignment would silently drop all but the last collision.
        np.add.at(receipts, (np.where(inside)[0], idx[inside]), float(row["amount"]))

    balances = opening_balance + np.cumsum(operating + receipts, axis=1)

    hit = balances <= 0.0
    hit_any = hit.any(axis=1)
    # argmax on a boolean row returns the first True; +1 converts to a 1-based
    # day count. Rows that never hit zero are censored at horizon_days.
    first = np.argmax(hit, axis=1) + 1
    days_to_zero = np.where(hit_any, first, horizon_days).astype(float)

    return SimulationPaths(
        balances=balances,
        days_to_zero=days_to_zero,
        hit_zero=hit_any,
        horizon_days=horizon_days,
        seed=seed,
    )


def compute_runway_at_risk(
    paths: SimulationPaths,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> RunwayAtRisk:
    """Compute RaR and CRaR from simulated days-to-zero.

    DEFINITIONS
    -----------
    Let D be the simulated days-to-zero distribution and alpha = 1 - confidence.

        RaR_c  = quantile(D, alpha)
                 "there is an alpha chance of hitting zero within RaR days"
        CRaR_c = E[ D | D <= RaR_c ]
                 the mean runway across the bad tail

    The tail of interest is the **lower** tail, because short runway is bad.
    This is the mirror image of market-risk VaR, where the bad tail is large
    losses; getting the direction wrong would produce a reassuring number.

    CRaR is reported alongside because two businesses can share a RaR of 11
    days while one's bad tail averages 10 days and the other's averages 3.
    RaR locates the cliff; CRaR measures the drop.

    STANDARD ERROR
    --------------
    The SE of the RaR estimate is obtained by **bootstrap resampling** the
    simulated draws, rather than the asymptotic quantile-variance formula. The
    asymptotic formula requires estimating the density at the quantile, which
    is itself noisy and badly behaved when the distribution is censored at the
    horizon (a point mass sits at horizon_days). The bootstrap handles the
    point mass without special-casing.

    Args:
        paths: output of `simulate_cash_paths`.
        confidence: e.g. 0.95 for 95% RaR.
        n_bootstrap: resamples for the standard error.
        rng: seeded generator; derived from the paths' seed if omitted.

    Returns:
        `RunwayAtRisk` including the SE that justifies the iteration count.
    """
    if rng is None:
        rng = np.random.default_rng(paths.seed + 1)

    d = paths.days_to_zero
    alpha = 1.0 - confidence
    rar = float(np.quantile(d, alpha))

    tail = d[d <= rar]
    crar = float(tail.mean()) if tail.size else float(rar)

    n = len(d)
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        boot[b] = np.quantile(rng.choice(d, size=n, replace=True), alpha)
    se = float(np.std(boot, ddof=1))

    return RunwayAtRisk(
        confidence_level=confidence,
        runway_at_risk_days=int(np.floor(rar)),
        conditional_runway_at_risk_days=crar,
        probability_of_shortfall=float(paths.hit_zero.mean()),
        n_iterations=n,
        mc_standard_error=se,
        random_seed=paths.seed,
    )


def convergence_study(
    simulate_fn,
    n_grid: Sequence[int],
    *,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[ConvergencePoint]:
    """Estimate RaR standard error as a function of N.

    This is what turns "10,000 iterations" from a number copied off a slide
    into a defensible choice. Monte Carlo error decays as O(1/sqrt(N)), so the
    curve should be roughly linear on a log-log plot with slope -1/2; the test
    suite checks that slope explicitly.

    Args:
        simulate_fn: callable(n_iterations, seed) -> SimulationPaths.
        n_grid: iteration counts to evaluate.
        confidence: RaR level.
        seed: base seed; each grid point uses a distinct derived seed.

    Returns:
        One `ConvergencePoint` per grid value.
    """
    out: list[ConvergencePoint] = []
    for i, n in enumerate(n_grid):
        paths = simulate_fn(int(n), seed + i)
        rar = compute_runway_at_risk(paths, confidence=confidence, n_bootstrap=200)
        out.append(
            ConvergencePoint(
                n_iterations=int(n),
                estimate=float(rar.runway_at_risk_days),
                standard_error=rar.mc_standard_error,
            )
        )
    return out


def sigma_from_interval(
    lower: np.ndarray, upper: np.ndarray, confidence: float
) -> np.ndarray:
    """Convert a prediction interval into an implied per-step standard deviation.

        sigma = (upper - lower) / (2 * z_{(1+c)/2})

    This is how A.1's uncertainty is propagated into the simulation instead of
    being discarded. It assumes the interval is symmetric and approximately
    Gaussian — true for SARIMAX's analytic intervals, an approximation for the
    empirical-quantile intervals. The approximation is documented rather than
    hidden because it slightly understates asymmetric tails.
    """
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    return np.abs(np.asarray(upper) - np.asarray(lower)) / (2.0 * z)


def run_simulation(
    *,
    opening_balance: float,
    forecast,
    receivables: pd.DataFrame,
    delays_by_cp: dict[str, np.ndarray],
    panel: pd.DataFrame,
    n_iterations: int,
    seed: int,
    horizon_days: Optional[int] = None,
    confidence: float = 0.95,
    copula_family: CopulaFamily = CopulaFamily.STUDENT_T,
    copula_df: float = 4.0,
) -> tuple[SimulationPaths, RunwayAtRisk, UncertaintyModelSpec]:
    """End-to-end: fit uncertainty, simulate, and read off RaR / CRaR.

    This is the single entry point Section B's `simulation/` package calls and
    the one that backs `GET /simulation/runway-at-risk`.

    Args:
        opening_balance: current cash.
        forecast: a `ForecastResult` from A.1; its intervals become the
            operating-flow uncertainty.
        receivables: outstanding receivables with `counterparty_id`, `amount`,
            `days_until_due`.
        delays_by_cp: historical delays for marginal fitting.
        panel: time-aligned delays for correlation estimation.
        n_iterations: Monte Carlo draws.
        seed: master seed.
        horizon_days: defaults to the forecast horizon.

    Returns:
        (paths, runway_at_risk, uncertainty_model_spec)
    """
    spec, corr = build_uncertainty_model(
        delays_by_cp, panel, copula_family=copula_family, copula_df=copula_df
    )

    point = np.array([p.point for p in forecast.path], dtype=float)
    lower = np.array([p.lower for p in forecast.path], dtype=float)
    upper = np.array([p.upper for p in forecast.path], dtype=float)
    sigma = sigma_from_interval(lower, upper, forecast.interval_confidence)

    h = horizon_days or len(point)
    point, sigma = point[:h], sigma[:h]

    paths = simulate_cash_paths(
        opening_balance=opening_balance,
        forecast_point=point,
        forecast_sigma=sigma,
        receivables=receivables,
        fits=spec.fits,
        correlation=corr,
        n_iterations=n_iterations,
        horizon_days=h,
        seed=seed,
        copula_family=copula_family,
        copula_df=copula_df,
    )
    rar = compute_runway_at_risk(paths, confidence=confidence)
    return paths, rar, spec


def build_uncertainty_model(
    delays_by_cp: dict[str, np.ndarray],
    panel: pd.DataFrame,
    *,
    copula_family: CopulaFamily = CopulaFamily.STUDENT_T,
    copula_df: float = 4.0,
) -> tuple[UncertaintyModelSpec, np.ndarray]:
    """Fit all marginals and the dependence structure in one call.

    Takes both representations deliberately: marginal fitting wants every
    observation it can get (order irrelevant), while correlation estimation
    requires time alignment. Passing only one would force the other to be
    reconstructed wrongly.

    Args:
        delays_by_cp: ragged per-counterparty delays, for marginal MLE fits.
        panel: time-aligned delay panel, for rank correlation.

    Returns:
        (spec, correlation_matrix) — the spec is the auditable record that
        travels with `SimulationResult`; the matrix is passed to the sampler.
    """
    order = sorted(delays_by_cp.keys())
    fits = [fit_delay_distribution(cp, delays_by_cp[cp]) for cp in order]
    corr = estimate_delay_correlation(panel, order)

    spec = UncertaintyModelSpec(
        fits=fits,
        copula=CopulaSpec(
            family=copula_family,
            df=copula_df if copula_family is CopulaFamily.STUDENT_T else None,
            correlation_matrix=corr.tolist(),
            counterparty_order=order,
            correlation_source=(
                "Spearman rank correlation of historical delays, converted to the "
                "Gaussian-copula linear parameter via rho = 2*sin(pi*rho_s/6), then "
                "projected to the nearest positive-definite matrix."
            ),
        ),
        fitted_at=datetime.now(timezone.utc),
    )
    return spec, corr
