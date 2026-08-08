"""Statistical-correctness tests for A.7 — ruin probability and Z''-score.

The load-bearing test is `test_monte_carlo_ruin_matches_the_analytic_formula`:
a driftful random walk's first-passage probability has a closed form, so the
simulator can be checked against an answer derived independently of it rather
than against itself. It also pins the *direction* of the discrete-monitoring
bias, which is a real property of the estimator and not a tolerance fudge.

`test_first_passage_is_not_a_terminal_value_check` exists because the two are
easy to conflate in code and the difference is the entire meaning of the
metric — a terminal check would report a business that missed payroll in week
three as healthy because a receivable landed in week six.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from scipy import stats

from app.schemas.solvency import ZScoreZone
from app.services.quant_core.bankruptcy_risk import (
    BalanceSheet,
    InsufficientBalanceSheet,
    Z2_COEFFICIENTS,
    altman_z_double_prime,
    assess_bankruptcy_risk,
    expected_days_to_ruin,
    hazard_table,
    ruin_curve,
    ruin_indicator_matrix,
    validate_ruin_calibration,
)

AS_OF = date(2026, 8, 9)

# Broadie-Glasserman-Kou discrete-barrier correction constant,
# beta = -zeta(1/2)/sqrt(2*pi). Used to compare a daily-step simulation against
# a continuous-time closed form on equal terms.
BGK_BETA = 0.5826


def _walk(*, n_iter: int, horizon: int, opening: float, mu: float,
          sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return opening + np.cumsum(rng.normal(mu, sigma, size=(n_iter, horizon)), axis=1)


def _analytic_first_passage(*, opening: float, mu: float, sigma: float,
                            horizon: float) -> float:
    """P(a Brownian motion with drift hits zero before T), in closed form.

        P = Phi((-x0 - mu*T) / (sigma*sqrt(T)))
            + exp(-2*mu*x0 / sigma^2) * Phi((-x0 + mu*T) / (sigma*sqrt(T)))

    This is the reflection-principle result for the first passage of
    `x0 + mu*t + sigma*W(t)` to zero. It is derived independently of the
    simulator, which is what makes it a real check rather than the simulator
    grading its own homework.
    """
    root_t = sigma * np.sqrt(horizon)
    term1 = stats.norm.cdf((-opening - mu * horizon) / root_t)
    term2 = np.exp(-2.0 * mu * opening / sigma ** 2) * stats.norm.cdf(
        (-opening + mu * horizon) / root_t
    )
    return float(term1 + term2)


# ---------------------------------------------------------------------------
# First-passage semantics
# ---------------------------------------------------------------------------

def test_first_passage_is_not_a_terminal_value_check():
    """A path that dips below zero and recovers has still ruined.

    Row 0 goes negative on day 2 and is well above zero by day 5. A terminal
    check (`balances[:, -1] <= 0`) would call it healthy. Ruin is absorbing.
    """
    balances = np.array([
        [100.0, 10.0, -5.0, 40.0, 500.0],     # dips, then recovers
        [100.0, 120.0, 130.0, 140.0, 150.0],  # never troubled
    ])
    ruined = ruin_indicator_matrix(balances)

    assert ruined[0].tolist() == [False, False, True, True, True]
    assert not ruined[1].any()
    assert ruin_curve(balances, horizons=[5])[0].ruin_probability == 0.5


def test_ruin_curve_is_non_decreasing_in_horizon():
    """More time cannot reduce the probability of an absorbing event."""
    balances = _walk(n_iter=4000, horizon=120, opening=40_000,
                     mu=-250.0, sigma=3_000.0, seed=7)
    probs = [p.ruin_probability for p in ruin_curve(balances)]
    assert all(a <= b + 1e-12 for a, b in zip(probs, probs[1:]))
    assert probs[0] < probs[-1], "the test case must actually accumulate risk"


@pytest.mark.parametrize(
    "opening,mu,sigma,horizon,seed",
    [
        (30_000.0, -200.0, 2_500.0, 90, 11),
        (50_000.0, -100.0, 3_000.0, 120, 3),
        (20_000.0, -300.0, 1_800.0, 60, 5),
    ],
)
def test_monte_carlo_ruin_matches_the_analytic_formula(
    opening, mu, sigma, horizon, seed
):
    """Check the simulator against a closed form, correcting for discreteness.

    Daily-step simulation observes the path only at day boundaries, so it
    cannot see an excursion below zero that begins and ends inside one day.
    The raw continuous formula therefore sits ABOVE the simulated value, always
    — measured here at roughly 2-3 percentage points.

    That gap is not slack to be absorbed by a loose tolerance; it is a known
    quantity. Broadie, Glasserman and Kou (1997) showed that discrete
    monitoring is equivalent to continuous monitoring of a barrier shifted by
    `beta * sigma * sqrt(dt)`, with `beta = -zeta(1/2)/sqrt(2*pi) ~ 0.5826`.
    Applying that shift closes the gap to under half a percentage point across
    every configuration below, which is a far stronger statement than a wide
    tolerance would be: it says the simulator is right *and* that its residual
    error is the one theory predicts, rather than error of unknown origin.
    """
    balances = _walk(n_iter=60_000, horizon=horizon, opening=opening,
                     mu=mu, sigma=sigma, seed=seed)
    simulated = ruin_curve(balances, horizons=[horizon])[0].ruin_probability

    raw = _analytic_first_passage(
        opening=opening, mu=mu, sigma=sigma, horizon=horizon
    )
    corrected = _analytic_first_passage(
        opening=opening + BGK_BETA * sigma, mu=mu, sigma=sigma, horizon=horizon
    )

    assert simulated < raw, "discrete monitoring must under-count crossings"
    assert simulated == pytest.approx(corrected, abs=0.006)


def test_monte_carlo_standard_error_shrinks_as_sqrt_n():
    """The binomial SE must fall like 1/sqrt(N); a 16x N should ~quarter it."""
    kwargs = dict(horizon=90, opening=30_000, mu=-200.0, sigma=2_500.0)
    se_small = ruin_curve(
        _walk(n_iter=2_000, seed=3, **kwargs), horizons=[90]
    )[0].standard_error
    se_large = ruin_curve(
        _walk(n_iter=32_000, seed=3, **kwargs), horizons=[90]
    )[0].standard_error
    assert se_large == pytest.approx(se_small / 4.0, rel=0.15)


def test_confidence_interval_never_leaves_the_unit_interval():
    """A Wald interval on a proportion near 0 can go negative if unclipped."""
    balances = _walk(n_iter=2_000, horizon=30, opening=5_000_000,
                     mu=1_000.0, sigma=500.0, seed=5)
    for point in ruin_curve(balances):
        assert 0.0 <= point.ci_lower <= point.ci_upper <= 1.0


# ---------------------------------------------------------------------------
# Hazard
# ---------------------------------------------------------------------------

def test_hazard_and_cumulative_curve_are_consistent():
    """Survival from the hazards must reproduce the cumulative ruin fraction.

    S(T) = prod(1 - h_i) telescopes to survivors/total, so this is an exact
    identity given exact counts — not an approximation. Any off-by-one in the
    at-risk bookkeeping breaks it.
    """
    balances = _walk(n_iter=5_000, horizon=90, opening=30_000,
                     mu=-250.0, sigma=2_800.0, seed=13)
    hazards = hazard_table(balances, bin_days=15)

    survival = 1.0
    for h in hazards:
        survival *= (1.0 - h.hazard)

    cumulative = ruin_curve(balances, horizons=[90])[0].ruin_probability
    assert survival == pytest.approx(1.0 - cumulative, abs=1e-9)


def test_at_risk_population_shrinks_monotonically():
    balances = _walk(n_iter=3_000, horizon=90, opening=25_000,
                     mu=-300.0, sigma=2_500.0, seed=17)
    at_risk = [h.n_at_risk for h in hazard_table(balances, bin_days=15)]
    assert all(a >= b for a, b in zip(at_risk, at_risk[1:]))
    assert at_risk[0] == 3_000


def test_expected_days_to_ruin_is_conditional_and_none_when_nothing_ruins():
    safe = _walk(n_iter=500, horizon=60, opening=5_000_000,
                 mu=5_000.0, sigma=100.0, seed=19)
    assert expected_days_to_ruin(safe) is None

    doomed = _walk(n_iter=2_000, horizon=90, opening=20_000,
                   mu=-400.0, sigma=2_000.0, seed=23)
    days = expected_days_to_ruin(doomed)
    assert days is not None and 0 < days <= 90


# ---------------------------------------------------------------------------
# Calibration — the check that makes the probability a probability
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def calibration_plugin():
    return validate_ruin_calibration(
        n_businesses=500, n_iterations=1_500, seed=20260809,
        parameter_uncertainty=False,
    )


@pytest.fixture(scope="module")
def calibration_predictive():
    return validate_ruin_calibration(
        n_businesses=500, n_iterations=1_500, seed=20260809,
        parameter_uncertainty=True,
    )


def test_predicted_probabilities_beat_the_base_rate_and_rank_correctly(
    calibration_predictive
):
    """Brier skill and AUC both, because neither substitutes for the other.

    A model predicting the base rate for everyone has skill 0 and AUC 0.5;
    a model with good skill and AUC 0.5 cannot tell a doomed business from a
    healthy one, which is what the product is for.
    """
    c = calibration_predictive
    assert c.brier_skill_score > 0.25, "must beat knowing only the base rate"
    assert c.roc_auc > 0.85, "must rank a failing business above a surviving one"
    assert 0.02 < c.realized_ruin_rate < 0.40, (
        "a degenerate base rate would make both metrics uninformative"
    )


def test_mean_prediction_tracks_the_realized_rate(calibration_predictive):
    c = calibration_predictive
    assert c.mean_predicted_probability == pytest.approx(
        c.realized_ruin_rate, abs=0.04
    )


def test_calibration_curve_is_increasing_where_it_is_populated(
    calibration_predictive
):
    """Buckets with real support must show rising observed frequency."""
    dense = [b for b in calibration_predictive.buckets if b.n >= 8]
    assert len(dense) >= 3
    lo, hi = dense[0], dense[-1]
    assert hi.observed_frequency > lo.observed_frequency


def test_propagating_parameter_uncertainty_improves_calibration(
    calibration_plugin, calibration_predictive
):
    """A measured result, and the reason the predictive mode is the default.

    Both runs are scored against the SAME businesses and the same realized
    outcomes — the split RNG streams in `validate_ruin_calibration` guarantee
    it — so the only difference is whether estimation noise was propagated.
    The plug-in estimator treats point estimates as truth and is therefore
    overconfident; the predictive construction is not.
    """
    assert calibration_plugin.realized_ruin_rate == \
        calibration_predictive.realized_ruin_rate, (
            "the two modes must be scored against an identical world, or the "
            "comparison confounds the estimator with the dataset"
        )

    def ece(c) -> float:
        total = sum(b.n for b in c.buckets)
        return sum(
            b.n * abs(b.observed_frequency - b.mean_predicted) for b in c.buckets
        ) / total

    assert ece(calibration_predictive) < ece(calibration_plugin)
    assert calibration_predictive.brier_score < calibration_plugin.brier_score
    # The plug-in run under-predicts because it discards estimation noise.
    assert (
        calibration_plugin.mean_predicted_probability
        < calibration_predictive.mean_predicted_probability
    )


# ---------------------------------------------------------------------------
# Altman Z''
# ---------------------------------------------------------------------------

def test_z_score_matches_a_hand_computed_value():
    bs = BalanceSheet(
        total_assets=1_000_000.0, total_liabilities=400_000.0,
        current_assets=500_000.0, current_liabilities=300_000.0,
        retained_earnings=150_000.0, ebit=90_000.0,
    )
    z = altman_z_double_prime(bs)

    x1, x2, x3, x4 = 0.2, 0.15, 0.09, 600_000.0 / 400_000.0
    a, b, c, d = Z2_COEFFICIENTS
    assert z.x1_working_capital_to_assets == pytest.approx(x1)
    assert z.x2_retained_earnings_to_assets == pytest.approx(x2)
    assert z.x3_ebit_to_assets == pytest.approx(x3)
    assert z.x4_equity_to_liabilities == pytest.approx(x4)
    assert z.z_score == pytest.approx(a * x1 + b * x2 + c * x3 + d * x4)
    assert z.zone == ZScoreZone.SAFE.value


def test_zones_split_at_the_published_thresholds():
    def z_for(ebit: float) -> str:
        return altman_z_double_prime(
            BalanceSheet(
                total_assets=1_000_000.0, total_liabilities=900_000.0,
                current_assets=300_000.0, current_liabilities=400_000.0,
                retained_earnings=-50_000.0, ebit=ebit,
            )
        ).zone

    # A weak balance sheet (liabilities near assets, negative retained
    # earnings, negative working capital) held fixed, with only EBIT varying —
    # so the zone transition is attributable to one input.
    assert z_for(-120_000.0) == ZScoreZone.DISTRESS.value
    assert z_for(300_000.0) == ZScoreZone.GREY.value
    assert z_for(600_000.0) == ZScoreZone.SAFE.value


def test_equity_is_derived_from_the_accounting_identity_and_tagged_as_such():
    bs = BalanceSheet(
        total_assets=800_000.0, total_liabilities=500_000.0,
        current_assets=400_000.0, current_liabilities=200_000.0,
        retained_earnings=100_000.0, ebit=60_000.0,
    )
    z = altman_z_double_prime(bs)
    equity = next(i for i in z.inputs if i.name == "equity")
    assert equity.value == pytest.approx(300_000.0)
    assert equity.provenance == "derived_from_ledger"
    assert z.approximated_input_names == []


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(total_assets=0.0), "total_assets"),
        (dict(total_liabilities=0.0), "total_liabilities"),
        (dict(retained_earnings=None), "retained_earnings"),
        (dict(ebit=None), "ebit"),
    ],
)
def test_missing_inputs_are_refused_not_zero_filled(kwargs, match):
    """Zero retained earnings is a different claim from unknown."""
    base = dict(
        total_assets=1_000_000.0, total_liabilities=400_000.0,
        current_assets=500_000.0, current_liabilities=300_000.0,
        retained_earnings=150_000.0, ebit=90_000.0,
    )
    with pytest.raises(InsufficientBalanceSheet, match=match):
        altman_z_double_prime(BalanceSheet(**{**base, **kwargs}))


# ---------------------------------------------------------------------------
# The composite assessment
# ---------------------------------------------------------------------------

def test_assessment_without_a_balance_sheet_omits_the_z_score():
    balances = _walk(n_iter=2_000, horizon=90, opening=30_000,
                     mu=-250.0, sigma=2_500.0, seed=29)
    risk = assess_bankruptcy_risk(balances, as_of=AS_OF, random_seed=29)

    assert risk.altman is None, "absent, not defaulted"
    assert risk.agreement is None
    assert risk.horizon_days == 90
    assert risk.ruin_curve[-1].ruin_probability == risk.headline_ruin_probability
    assert any("calibration" in c.lower() for c in risk.caveats)


def test_solvent_but_illiquid_disagreement_is_surfaced_not_averaged():
    """The case the composite exists to catch, and the reason it is not a mean.

    Strong balance sheet, failing cash flow. Averaging the two into one score
    would report a comfortable middle and hide precisely the situation that
    causes a missed payroll.
    """
    balances = _walk(n_iter=3_000, horizon=90, opening=8_000,
                     mu=-600.0, sigma=1_500.0, seed=31)
    healthy = BalanceSheet(
        total_assets=5_000_000.0, total_liabilities=500_000.0,
        current_assets=900_000.0, current_liabilities=200_000.0,
        retained_earnings=1_200_000.0, ebit=700_000.0,
    )
    risk = assess_bankruptcy_risk(
        balances, as_of=AS_OF, random_seed=31, balance_sheet=healthy
    )

    assert risk.headline_ruin_probability > 0.20
    assert risk.altman.zone == ZScoreZone.SAFE.value
    assert "SOLVENT BUT ILLIQUID" in risk.agreement
    assert any("never averaged" in c for c in risk.caveats)


def test_a_bad_balance_sheet_is_reported_rather_than_raising():
    balances = _walk(n_iter=500, horizon=60, opening=50_000,
                     mu=100.0, sigma=800.0, seed=37)
    risk = assess_bankruptcy_risk(
        balances, as_of=AS_OF, random_seed=37,
        balance_sheet=BalanceSheet(
            total_assets=1_000.0, total_liabilities=500.0,
            current_assets=100.0, current_liabilities=50.0,
            retained_earnings=None, ebit=None,
        ),
    )
    assert risk.altman is None
    assert any("no Z''-score" in c for c in risk.caveats)


def test_seed_and_iteration_count_travel_with_the_result():
    balances = _walk(n_iter=1_234, horizon=45, opening=30_000,
                     mu=-100.0, sigma=2_000.0, seed=41)
    risk = assess_bankruptcy_risk(balances, as_of=AS_OF, random_seed=41)
    assert risk.n_iterations == 1_234
    assert risk.random_seed == 41
    assert risk.as_of == AS_OF
