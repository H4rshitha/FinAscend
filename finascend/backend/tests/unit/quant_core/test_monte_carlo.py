"""Statistical-correctness tests for A.2 — fitting, copula, RaR, convergence.

These are the tests that make the headline metric checkable. In particular
`test_copula_marginals_survive_the_transform` exists specifically to catch a
missing probability-integral transform, and is deliberately independent of any
correlation test — a correlation check passes even when the PIT is missing, so
one test cannot cover both failure modes.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from app.services.quant_core.monte_carlo_engine import (
    CopulaFamily,
    build_uncertainty_model,
    compute_runway_at_risk,
    convergence_study,
    estimate_delay_correlation,
    fit_delay_distribution,
    frozen_from_fit,
    sample_correlated_delays,
    simulate_cash_paths,
)
from app.services.quant_core.pipeline import build_as_of_view
from app.services.quant_core.synthetic_data import (
    Regime,
    delay_panel,
    generate_dataset,
    observed_delays_by_counterparty,
)
import pandas as pd


@pytest.fixture(scope="module")
def world():
    return generate_dataset(seed=42, regime=Regime.EASY, n_days=1460, n_counterparties=6)


@pytest.fixture(scope="module")
def fitted(world):
    delays = observed_delays_by_counterparty(world.payments)
    panel = delay_panel(world.payments)
    return build_uncertainty_model(delays, panel)


# ---------------------------------------------------------------------------
# Parameter recovery — the central ground-truth check
# ---------------------------------------------------------------------------

def test_mle_recovers_known_gamma_parameters(world):
    """The fitter must recover the parameters the generator actually used.

    This is the validation trick the architecture plan calls for: because the
    true shape and scale are known, the fit can be *scored* rather than merely
    inspected. Tolerance is on the distribution mean (shape x scale) rather
    than on shape and scale individually, because those two are strongly
    negatively correlated in the Gamma likelihood — a fit can trade shape
    against scale and still describe the same distribution. The mean is the
    identifiable quantity.
    """
    delays = observed_delays_by_counterparty(world.payments)
    errors = []
    for cp in world.truth.counterparties:
        fit = fit_delay_distribution(cp.counterparty_id, delays[cp.counterparty_id])
        gamma_fit = next(c for c in fit.candidates if c.family == "gamma")
        fitted_mean = gamma_fit.params["shape"] * gamma_fit.params["scale"]
        errors.append(abs(fitted_mean - cp.mean_delay_days) / cp.mean_delay_days)

    assert max(errors) < 0.15, f"worst mean-delay recovery error {max(errors):.1%}"
    assert float(np.mean(errors)) < 0.08


def test_recovery_improves_with_more_data(world):
    """A consistent estimator must get better as n grows.

    Averaged ACROSS counterparties, not measured on one. A single
    counterparty's error is noisy enough to move either direction by chance —
    and the delays are serially correlated, so an early slice is not even an
    independent sample. Averaging over counterparties is what makes this a
    test of consistency rather than a coin flip.
    """
    delays = observed_delays_by_counterparty(world.payments)

    def mean_relative_error(n: int | None) -> float:
        errs = []
        for cp in world.truth.counterparties:
            obs = delays[cp.counterparty_id]
            sample = obs if n is None else obs[:n]
            if len(sample) < 15:
                continue
            fit = fit_delay_distribution(cp.counterparty_id, sample)
            g = next(c for c in fit.candidates if c.family == "gamma")
            fitted_mean = g.params["shape"] * g.params["scale"]
            errs.append(abs(fitted_mean - cp.mean_delay_days) / cp.mean_delay_days)
        return float(np.mean(errs))

    small = mean_relative_error(25)
    large = mean_relative_error(None)
    assert large < small, f"error did not shrink with n ({large:.4f} vs {small:.4f})"


def test_selected_fit_is_not_refuted_by_ks(fitted):
    """The winning family must actually describe the data, not just win a race."""
    spec, _ = fitted
    for f in spec.fits:
        winner = next(c for c in f.candidates if c.selected)
        assert winner.ks_pvalue > 0.01, (
            f"{f.counterparty_id}: selected {winner.family} is refuted by KS "
            f"(p={winner.ks_pvalue:.4f})"
        )


def test_all_candidates_are_retained(fitted):
    """Losers must be kept so the selection is auditable."""
    spec, _ = fitted
    for f in spec.fits:
        assert len(f.candidates) == 3
        assert sum(1 for c in f.candidates if c.selected) == 1


def test_fit_rejects_insufficient_data():
    """Fitting two parameters to a handful of points should refuse, not guess."""
    with pytest.raises(ValueError, match="need >= 10"):
        fit_delay_distribution("CP", np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Copula — the PIT test, deliberately separate from the correlation test
# ---------------------------------------------------------------------------

def test_copula_marginals_survive_the_transform(fitted):
    """Copula-sampled delays must still follow each counterparty's fitted marginal.

    THIS IS THE PIT REGRESSION TEST. The classic copula bug is to sample
    correlated latent normals and use them directly as delays, skipping the
    probability-integral transform and the inverse-CDF step. That produces the
    right correlation and completely wrong marginals — symmetric, unbounded,
    able to go negative — and it is invisible in any correlation check.

    It is kept independent of `test_copula_reproduces_target_correlation` for
    exactly that reason: a correlation test passes with the transform missing.
    """
    spec, corr = fitted
    rng = np.random.default_rng(123)
    sample = sample_correlated_delays(spec.fits, corr, 20_000, rng)

    assert (sample >= 0).all(), "negative delays: the inverse-CDF step is missing"

    for j, f in enumerate(spec.fits):
        _, p = stats.kstest(sample[:, j], frozen_from_fit(f).cdf)
        assert p > 0.01, (
            f"{f.counterparty_id}: copula output does not follow its fitted "
            f"marginal (KS p={p:.4f}) — check the PIT and ppf inversion"
        )


@pytest.mark.parametrize("family", ["gamma", "weibull", "lognormal"])
def test_copula_marginals_survive_the_transform_for_every_family(family):
    """The PIT must hold for Gamma, Weibull AND Log-normal, not just the winner.

    `test_copula_marginals_survive_the_transform` runs on whichever families
    AIC happened to select on the synthetic world, which in practice is almost
    always Gamma — that is the family the generator draws from. A transform
    that is correct for Gamma and wrong for Log-normal would therefore never be
    exercised. This test pins each family explicitly by fitting to data drawn
    from it, so all three inverse-CDF paths in `frozen_from_fit` are covered.
    """
    rng = np.random.default_rng(20)
    truth = {
        "gamma": stats.gamma(a=2.5, scale=4.0),
        "weibull": stats.weibull_min(c=1.8, scale=9.0),
        "lognormal": stats.lognorm(s=0.6, scale=8.0),
    }[family]

    fits = [
        fit_delay_distribution(f"CP{i}", truth.rvs(size=1200, random_state=rng))
        for i in range(3)
    ]
    for f in fits:
        assert f.selected_family == family, (
            f"expected AIC to select {family} on data drawn from it, got "
            f"{f.selected_family}; the test would not be exercising that path"
        )

    corr = np.full((3, 3), 0.5)
    np.fill_diagonal(corr, 1.0)
    sample = sample_correlated_delays(fits, corr, 20_000, np.random.default_rng(3))

    assert (sample >= 0).all(), "negative delays: the inverse-CDF step is missing"
    for j, f in enumerate(fits):
        _, p = stats.kstest(sample[:, j], frozen_from_fit(f).cdf)
        assert p > 0.01, (
            f"{family}: copula output does not follow its fitted marginal "
            f"(KS p={p:.4f}) — the PIT or the {family} ppf inversion is wrong"
        )


def test_the_pit_test_actually_catches_a_missing_transform(fitted):
    """The marginal test must FAIL on the bug, and the correlation test must not.

    A regression test that cannot fail is decoration. This reproduces the
    classic copula bug directly — correlated latent normals used as delays,
    with no probability-integral transform and no inverse-CDF step — and
    asserts two things about it:

      1. the marginal (KS) check rejects it, so the real test has teeth;
      2. the rank-correlation check still PASSES on it, which is the whole
         reason the two tests must stay separate. A single combined test would
         be satisfied by the buggy sampler.
    """
    spec, corr = fitted
    rng = np.random.default_rng(77)
    chol = np.linalg.cholesky(corr)
    buggy = rng.standard_normal(size=(20_000, len(spec.fits))) @ chol.T

    rejected = [
        stats.kstest(buggy[:, j], frozen_from_fit(f).cdf).pvalue <= 0.01
        for j, f in enumerate(spec.fits)
    ]
    assert all(rejected), (
        "the KS marginal check failed to reject latent normals used as delays "
        "— it would not catch a missing PIT"
    )
    assert (buggy < 0).any(), "latent normals should be able to go negative"

    # ...and yet the correlation is right, which is exactly the trap.
    observed = stats.spearmanr(buggy).statistic
    iu = np.triu_indices_from(corr, 1)
    target = (6.0 / np.pi) * np.arcsin(np.asarray(corr)[iu] / 2.0)
    assert np.allclose(observed[iu], target, atol=0.06), (
        "the buggy sampler should still reproduce the target correlation — if "
        "it does not, this test is no longer demonstrating why the marginal "
        "check has to exist separately"
    )


def test_copula_reproduces_target_correlation(fitted):
    """Sampled rank correlation must match the correlation that was fitted."""
    spec, corr = fitted
    rng = np.random.default_rng(5)
    sample = sample_correlated_delays(spec.fits, corr, 20_000, rng)

    observed = stats.spearmanr(sample).statistic
    iu = np.triu_indices_from(corr, 1)
    # Compare on the Spearman scale by converting the fitted linear parameter
    # back: rho_s = (6/pi) * arcsin(rho/2).
    target_spearman = (6.0 / np.pi) * np.arcsin(np.asarray(corr)[iu] / 2.0)
    assert np.allclose(observed[iu], target_spearman, atol=0.06)


def test_correlation_recovered_from_time_aligned_panel(world):
    """Correlation must be recovered from the panel, and ragged input must not be used.

    Documents the alignment bug: estimating from ragged per-counterparty
    arrays gave 0.02 against a true latent 0.15, because defaults shift the
    arrays relative to each other.
    """
    panel = delay_panel(world.payments)
    order = sorted(panel.columns)
    corr = estimate_delay_correlation(panel, order)
    off = corr[np.triu_indices_from(corr, 1)]
    assert off.mean() > 0.5 * world.truth.delay_correlation, (
        f"recovered {off.mean():.3f} vs true {world.truth.delay_correlation}"
    )


@pytest.mark.parametrize(
    "regime, n_days, n_cp, true_rho, lo, hi",
    [
        (Regime.EASY, 1095, 12, 0.15, 0.05, 0.30),
        (Regime.ADVERSARIAL, 1500, 10, 0.72, 0.55, 0.85),
        (Regime.ADVERSARIAL, 1095, 12, 0.72, 0.40, 0.85),
    ],
)
def test_correlation_recovery_is_bounded_per_configuration(
    regime, n_days, n_cp, true_rho, lo, hi
):
    """Recovery must be bracketed, and the bracket is CONFIGURATION-SPECIFIC.

    Quoting a single recovery figure without saying which world produced it is
    how a stale number survives a refactor. Measured on seed 42 at the time of
    writing:

        easy,        1095d, 12cp, true 0.15 -> 0.122
        adversarial, 1500d, 10cp, true 0.72 -> 0.739
        adversarial, 1095d, 12cp, true 0.72 -> 0.527

    Those differ by more than 0.2 across configurations of the *same* true
    parameter, because the estimate depends on how many paired weekly
    observations survive default censoring. The band is therefore per
    configuration and deliberately wide: this test protects against the
    alignment bug (which collapsed the estimate toward zero), not against
    ordinary sampling variation.
    """
    ds = generate_dataset(seed=42, regime=regime, n_days=n_days, n_counterparties=n_cp)
    panel = delay_panel(ds.payments)
    order = sorted(panel.columns)
    corr = estimate_delay_correlation(panel, order)
    recovered = float(corr[np.triu_indices_from(corr, 1)].mean())

    assert ds.truth.delay_correlation == pytest.approx(true_rho)
    assert lo <= recovered <= hi, (
        f"{regime.value} {n_days}d/{n_cp}cp: recovered {recovered:.3f} outside "
        f"[{lo}, {hi}] against true {true_rho}"
    )


def test_correlation_matrix_is_positive_definite(fitted):
    """Cholesky in the sampler requires positive-definiteness."""
    _, corr = fitted
    np.linalg.cholesky(corr)   # raises if not PD
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-9)


def test_independence_understates_tail_risk(world):
    """Independent sampling must produce a LONGER apparent runway than the copula.

    This is the module's central claim made testable: independence makes
    simultaneous slowdowns astronomically unlikely, so the simulated tail is
    comfortable and RaR is optimistic. Measured on the adversarial world,
    independence reported 28 days against the copula's 19.
    """
    adv = generate_dataset(seed=42, regime=Regime.ADVERSARIAL, n_days=900, n_counterparties=8)
    view = build_as_of_view(adv, adv.daily["date"].iloc[780].date())
    spec, corr = build_uncertainty_model(view.delays_by_cp, view.delay_panel)

    horizon = 180
    point = np.full(horizon, float(view.daily["net_ex_receipts"].tail(60).mean()))
    sigma = np.full(horizon, float(view.daily["net_ex_receipts"].tail(60).std()))

    def rar_with(correlation):
        paths = simulate_cash_paths(
            opening_balance=view.opening_balance,
            forecast_point=point,
            forecast_sigma=sigma,
            receivables=view.receivables,
            fits=spec.fits,
            correlation=correlation,
            n_iterations=8000,
            horizon_days=horizon,
            seed=99,
        )
        return compute_runway_at_risk(paths, confidence=0.95).runway_at_risk_days

    correlated = rar_with(corr)
    independent = rar_with(np.eye(len(spec.fits)))
    assert independent >= correlated, (
        f"independent RaR ({independent}d) should be no shorter than the "
        f"correlated one ({correlated}d) — independence must look safer"
    )


# ---------------------------------------------------------------------------
# RaR / CRaR and convergence
# ---------------------------------------------------------------------------

_TOY_CACHE: dict = {}


def _toy_model():
    """Fitted marginals + correlation for the toy scenario, built once."""
    if "spec" not in _TOY_CACHE:
        world = generate_dataset(seed=1, n_days=800, n_counterparties=4)
        spec, corr = build_uncertainty_model(
            observed_delays_by_counterparty(world.payments),
            delay_panel(world.payments),
        )
        _TOY_CACHE["spec"], _TOY_CACHE["corr"] = spec, corr
    return _TOY_CACHE["spec"], _TOY_CACHE["corr"]


def _toy_paths(n_iterations: int, seed: int, horizon: int = 120):
    """A small, controlled simulation whose days-to-zero is genuinely spread.

    The parameters matter: an earlier version put the business either safely
    solvent or instantly bankrupt, so days_to_zero was a point mass, every
    bootstrap resample returned the same quantile, and the standard error was
    identically zero — which made the convergence test vacuous rather than
    passing. The drift and volatility below are set so the zero-crossing lands
    inside the horizon with real dispersion around it.
    """
    spec, corr = _toy_model()
    rng = np.random.default_rng(0)
    receivables = pd.DataFrame(
        {
            "counterparty_id": [spec.fits[0].counterparty_id] * 6,
            "amount": rng.uniform(40_000, 90_000, 6),
            "days_until_due": rng.integers(1, 50, 6).astype(float),
        }
    )
    return simulate_cash_paths(
        opening_balance=1_200_000.0,
        forecast_point=np.full(horizon, -18_000.0),
        forecast_sigma=np.full(horizon, 40_000.0),
        receivables=receivables,
        fits=spec.fits,
        correlation=corr,
        n_iterations=n_iterations,
        horizon_days=horizon,
        seed=seed,
    )


def test_rar_is_the_lower_quantile_and_crar_is_worse():
    """RaR must read the SHORT-runway tail, and CRaR must be at least as severe.

    Getting the tail direction wrong would produce a reassuring number: in
    market-risk VaR the bad tail is large losses, but here the bad tail is
    SHORT runway, so it is the lower quantile.
    """
    paths = _toy_paths(6000, seed=11)
    rar = compute_runway_at_risk(paths, confidence=0.95)

    expected = float(np.quantile(paths.days_to_zero, 0.05))
    assert abs(rar.runway_at_risk_days - np.floor(expected)) <= 1
    assert rar.conditional_runway_at_risk_days <= rar.runway_at_risk_days + 1, (
        "CRaR must be no better than RaR — it is the mean of the tail beyond it"
    )
    assert rar.runway_at_risk_days <= np.median(paths.days_to_zero)


def test_probability_of_shortfall_matches_the_paths():
    paths = _toy_paths(4000, seed=12)
    rar = compute_runway_at_risk(paths, confidence=0.95)
    assert abs(rar.probability_of_shortfall - paths.hit_zero.mean()) < 1e-9


def test_simulation_is_reproducible():
    a = _toy_paths(2000, seed=99)
    b = _toy_paths(2000, seed=99)
    np.testing.assert_array_equal(a.days_to_zero, b.days_to_zero)


def test_monte_carlo_error_decays_as_one_over_sqrt_n():
    """Standard error must fall like N^(-1/2), the defining Monte Carlo rate.

    This is what makes the iteration count defensible rather than a number
    copied off a slide: it demonstrates the estimator converges at the
    expected rate, so the SE at a chosen N is predictable.
    """
    grid = [500, 1000, 2000, 4000, 8000, 16000]
    points = convergence_study(
        lambda n, s: _toy_paths(n, s), grid, confidence=0.95, seed=3
    )
    usable = [p for p in points if p.standard_error > 0]
    assert len(usable) >= 4, "too few non-degenerate points to fit a rate"

    slope = np.polyfit(
        np.log([p.n_iterations for p in usable]),
        np.log([p.standard_error for p in usable]),
        1,
    )[0]
    assert -0.85 < slope < -0.25, f"log-log slope {slope:.3f} is not close to -0.5"
