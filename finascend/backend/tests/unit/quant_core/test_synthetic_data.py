"""Statistical-correctness tests for A.0, the ground-truth generator.

Everything downstream is validated against this generator, so if it is wrong,
every other test in this suite is validating against a lie. These tests check
the generator's *claimed properties* hold, not merely that it returns a frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.services.quant_core.synthetic_data import (
    DELAY_PERSISTENCE,
    Regime,
    delay_panel,
    generate_dataset,
    observed_delays_by_counterparty,
)


@pytest.fixture(scope="module")
def easy():
    return generate_dataset(seed=42, regime=Regime.EASY, n_days=900, n_counterparties=8)


@pytest.fixture(scope="module")
def adversarial():
    return generate_dataset(
        seed=42, regime=Regime.ADVERSARIAL, n_days=900, n_counterparties=8
    )


def test_reproducible_from_seed():
    """Identical seeds must produce byte-identical worlds.

    Reproducibility is a hard requirement of the code-quality bar, and it is
    the property most easily broken by an accidental module-level RNG.
    """
    a = generate_dataset(seed=7, n_days=300, n_counterparties=5)
    b = generate_dataset(seed=7, n_days=300, n_counterparties=5)
    pd.testing.assert_frame_equal(a.daily, b.daily)
    pd.testing.assert_frame_equal(a.payments, b.payments)


def test_different_seeds_differ():
    """A seed that does not change the output would make the seed decorative."""
    a = generate_dataset(seed=1, n_days=300, n_counterparties=5)
    b = generate_dataset(seed=2, n_days=300, n_counterparties=5)
    assert not np.allclose(a.daily["inflow"], b.daily["inflow"])


def test_delays_follow_the_declared_gamma_without_default_censoring():
    """Realized delays must match the declared Gamma when nothing censors them.

    This is the ground-truth contract every other test depends on.

    `stress_beta=0` is essential here. With the default coupling ON, default
    probability rises with stress and stress is what produces long delays, so
    defaulted invoices are disproportionately the slow ones. Since only PAID
    invoices yield an observed delay, the observed distribution is the true
    Gamma with its right tail preferentially removed — P(delay | paid), not
    P(delay). Testing the uncoupled world isolates the delay mechanism; the
    censoring itself is asserted separately below.

    THINNING
    --------
    The sample is thinned before the KS test because the delays are serially
    correlated by the AR(1) persistence, and **the KS test assumes i.i.d.
    observations**. Dependence shrinks the effective sample size, so the
    empirical CDF converges more slowly than the standard critical values
    assume and the test over-rejects a distribution that is in fact correct.
    Measured directly on this data: median p-value across counterparties rose
    0.031 -> 0.190 -> 0.257 -> 0.630 at thinning factors 1, 4, 8, 12, while the
    moments were recovered accurately at every thinning. The distribution was
    right the whole time; the test was wrong.
    """
    ds = generate_dataset(
        seed=42, regime=Regime.EASY, n_days=2200, n_counterparties=6, stress_beta=0.0
    )
    delays = observed_delays_by_counterparty(ds.payments)
    thin = 10   # ~0.75^10 = 0.06 residual autocorrelation
    for cp in ds.truth.counterparties:
        obs = delays[cp.counterparty_id][::thin]
        _, p = stats.kstest(
            obs, stats.gamma(a=cp.delay_shape, scale=cp.delay_scale).cdf
        )
        assert p > 0.01, (
            f"{cp.counterparty_id}: realized delays reject the declared Gamma (p={p:.4f})"
        )


def test_delay_moments_match_the_declared_gamma():
    """Sample moments must match the declared Gamma's moments.

    A companion to the KS test that is valid under serial correlation: the
    sample mean and variance remain unbiased when observations are dependent
    (only their standard errors inflate), so this checks the distribution is
    right without the i.i.d. assumption the KS test needs.
    """
    ds = generate_dataset(
        seed=42, regime=Regime.EASY, n_days=2200, n_counterparties=6, stress_beta=0.0
    )
    delays = observed_delays_by_counterparty(ds.payments)
    for cp in ds.truth.counterparties:
        obs = delays[cp.counterparty_id]
        true_mean = cp.delay_shape * cp.delay_scale
        true_var = cp.delay_shape * cp.delay_scale**2
        assert abs(obs.mean() - true_mean) / true_mean < 0.20
        assert abs(obs.var(ddof=1) - true_var) / true_var < 0.40


def test_default_coupling_censors_the_slow_tail(easy):
    """With coupling on, PAID invoices must be faster than DEFAULTED ones.

    Documents the survivorship bias directly: it is a real property of the
    world, not a bug, and A.2's fitted marginals must be read as
    conditional-on-payment because of it.
    """
    p = easy.payments
    paid_delay = p.loc[p["paid"], "delay_days"].mean()
    defaulted_delay = p.loc[~p["paid"], "delay_days"].mean()
    assert defaulted_delay > paid_delay, (
        f"defaulted invoices ({defaulted_delay:.1f}d) should be slower than "
        f"paid ones ({paid_delay:.1f}d) — the stress coupling is not working"
    )


def test_inflow_is_receipts_plus_cash_sales(easy):
    """Inflow must be exactly the two components it claims to decompose into.

    Guards the coherence fix: an earlier version generated `inflow` and the
    invoice payments as unrelated streams, which injected phantom cash and
    made runway meaningless.
    """
    d = easy.daily
    np.testing.assert_allclose(
        d["inflow"].to_numpy(),
        (d["receipts"] + d["cash_sales"]).to_numpy(),
        rtol=1e-9,
    )


def test_net_ex_receipts_excludes_receipts(easy):
    """The forecastable series must not contain the receivable stream.

    Forecasting `net` and then adding simulated receivable arrivals would
    double-count every rupee of receivable.
    """
    d = easy.daily
    np.testing.assert_allclose(
        d["net_ex_receipts"].to_numpy(),
        (d["cash_sales"] - d["outflow"]).to_numpy(),
        rtol=1e-9,
    )


def test_balance_is_cumulative_net(easy):
    d = easy.daily
    expected = d["balance"].iloc[0] + d["net"].iloc[1:].cumsum()
    np.testing.assert_allclose(
        d["balance"].iloc[1:].to_numpy(), expected.to_numpy(), rtol=1e-9
    )


def test_no_receipt_ramp_at_series_start(easy):
    """Warm-up must leave the receipt stream in steady state from day 0.

    Without pre-window invoices, receipts ramp from zero over roughly
    (terms + delay) days while costs run at full rate, burning opening cash on
    a pure edge artifact. The first month's mean receipts should therefore be
    within a reasonable factor of the following months'.
    """
    r = easy.daily["receipts"]
    first_month = r.iloc[:30].mean()
    later = r.iloc[60:180].mean()
    assert first_month > 0.5 * later, (
        f"receipts ramp detected: first 30d mean {first_month:,.0f} vs "
        f"later {later:,.0f} — warm-up is not working"
    )


def test_adversarial_stays_solvent_but_deteriorates(adversarial):
    """The adversarial world must be hard, not already dead.

    A negative balance inside the history describes a business that already
    failed, which is degenerate for a liquidity model; a business that never
    approaches zero is degenerate the other way.
    """
    bal = adversarial.daily["balance"].to_numpy()
    assert bal.min() > 0, "adversarial balance went negative inside the history"
    # Revenue must actually decline after the structural break.
    inflow = adversarial.daily["inflow"]
    assert inflow.iloc[-90:].mean() < 0.85 * inflow.iloc[:90].mean()


def test_adversarial_has_fatter_delay_tails_than_easy(easy, adversarial):
    """The adversarial regime must be genuinely harder, not just relabelled."""
    e = np.concatenate(list(observed_delays_by_counterparty(easy.payments).values()))
    a = np.concatenate(
        list(observed_delays_by_counterparty(adversarial.payments).values())
    )
    assert stats.skew(a) > stats.skew(e), "adversarial delays are not more skewed"
    assert np.quantile(a, 0.99) > np.quantile(e, 0.99)


def test_delay_panel_is_time_aligned(easy):
    """The panel must align counterparties on a shared date index.

    Correlation estimated from ragged per-counterparty arrays collapses toward
    zero because defaults remove different rows from different counterparties.
    """
    panel = delay_panel(easy.payments)
    assert isinstance(panel.index, pd.DatetimeIndex) or panel.index.dtype.kind == "M"
    assert panel.shape[1] == len(easy.truth.counterparties)


def test_delays_are_serially_persistent(easy):
    """Latent stress must persist week to week (AR(1) with rho > 0).

    Without persistence, a counterparty's payment history carries no
    information about its current state, which caps every A.4 credit model at
    roughly chance no matter how good the model is.
    """
    panel = delay_panel(easy.payments)
    lag1 = []
    for col in panel.columns:
        s = panel[col].dropna()
        if len(s) > 30:
            lag1.append(s.autocorr(lag=1))
    mean_ac = float(np.nanmean(lag1))
    assert mean_ac > 0.15, (
        f"mean lag-1 autocorrelation {mean_ac:.3f} is too low; "
        f"DELAY_PERSISTENCE={DELAY_PERSISTENCE} appears not to be applied"
    )


def test_defaults_are_linked_to_slowness_within_counterparty(easy):
    """Default must be predictable from slowness AT THE INVOICE LEVEL.

    The link is within a counterparty over time, not across counterparties:
    each counterparty's base default probability is drawn independently of its
    delay parameters, so a cross-counterparty correlation is zero BY
    CONSTRUCTION and testing for one measures nothing. (It came out at -0.08,
    which is exactly the noise you would expect.)

    Comparing defaulted vs paid invoices *within* each counterparty is the
    test that actually probes the mechanism, and it is the mechanism A.4
    depends on: without it, payment history carries no signal about default.
    """
    p = easy.payments
    wins = 0
    total = 0
    for _, grp in p.groupby("counterparty_id"):
        d = grp.loc[~grp["paid"], "delay_days"]
        q = grp.loc[grp["paid"], "delay_days"]
        if len(d) >= 5 and len(q) >= 5:
            total += 1
            wins += int(d.mean() > q.mean())
    assert total >= 4, "not enough counterparties with both outcomes to test"
    assert wins / total > 0.7, (
        f"only {wins}/{total} counterparties show slower defaults — the "
        "stress->default coupling is too weak to be learnable"
    )
