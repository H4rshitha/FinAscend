"""Tests for A.1 (forecasting), A.4 (credit risk), A.5/A.6 (anomaly, text).

The forecasting tests check the two things that are easy to get wrong and
invisible when wrong: that walk-forward validation never trains on the future,
and that the reported metrics are computed correctly against a hand-checkable
ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.quant_core.anomaly_detection import detect_anomalies, robust_z_scores
from app.services.quant_core.forecasting import (
    ForecastModelName,
    SeasonalNaiveForecaster,
    HoltWintersForecaster,
    mape,
    rmse,
    select_and_forecast,
    walk_forward_validate,
    conformal_multiplier,
    conformal_interval,
    fit_scale_profile,
)
from app.services.quant_core.risk_scoring import (
    RiskModelName,
    build_features,
    calibration_curve,
    compare_to_baseline,
    explain_prediction,
    rationale_from_contributions,
    train_models,
)
from app.services.quant_core.synthetic_data import Regime, generate_dataset
from app.services.quant_core.unstructured import (
    ReceiptClassifier,
    compare_periods,
    decompose_seasonality,
)


# ---------------------------------------------------------------------------
# A.1 — metrics and no-look-ahead
# ---------------------------------------------------------------------------

def test_rmse_and_mape_against_hand_computed_values():
    """Metrics verified against arithmetic done by hand, not against themselves."""
    actual = np.array([100.0, 200.0, 300.0])
    pred = np.array([110.0, 180.0, 330.0])
    # errors: -10, +20, -30 -> squares 100, 400, 900 -> mean 466.667 -> sqrt
    assert rmse(actual, pred) == pytest.approx(np.sqrt(1400.0 / 3.0))
    # |%| errors: 10%, 10%, 10% -> mean 10%
    assert mape(actual, pred) == pytest.approx(10.0)


def test_seasonal_naive_repeats_the_last_cycle():
    """The baseline must do exactly what it claims: repeat the last period."""
    y = np.array([1.0, 2, 3, 4, 5, 6, 7, 10, 20, 30, 40, 50, 60, 70])
    f = SeasonalNaiveForecaster(period=7).fit(y)
    out = f.predict(7)
    np.testing.assert_allclose(out.point, [10, 20, 30, 40, 50, 60, 70])


def test_walk_forward_never_trains_on_the_future():
    """Each fold must be fitted only on data strictly before its test window.

    Verified with a level shift placed so that one fold straddles it. The fold
    trained entirely before the shift must be badly wrong on it; a fold that
    had leaked future data would predict the new level and score well.

    The shift sits near the END of the series rather than the middle, because
    the folds are anchored to the end (see `test_walk_forward_folds_are_
    anchored_to_the_end_of_the_series`). A shift at the midpoint would now fall
    before every evaluation window and the test would pass vacuously — which is
    exactly how it failed when the anchoring changed, and the reason the
    construction is spelled out here.
    """
    rng = np.random.default_rng(0)
    # n=300, min_train=150, horizon=14, n_folds=4 puts the fold boundaries at
    # 244/258/272/286/300, so a shift at 260 lands inside the second fold.
    y = np.concatenate([rng.normal(100, 5, 260), rng.normal(10_000, 5, 40)])
    res = walk_forward_validate(
        y, lambda: SeasonalNaiveForecaster(period=7), horizon=14, n_folds=4
    )
    assert len(res.fold_rmses) == 4

    assert res.fold_rmses[0] < 100.0, (
        "the fold entirely before the shift should be accurate; if it is not, "
        "the construction no longer isolates the leak it is testing for"
    )
    assert res.fold_rmses[1] > 1000.0, (
        "the fold straddling the shift must be badly wrong — a model that had "
        "seen the future would have predicted the new level"
    )


def test_walk_forward_folds_are_anchored_to_the_end_of_the_series():
    """The last fold must evaluate on the FINAL `horizon` observations.

    The previous implementation started at `min_train = n // 2` and walked
    forward, so with n=1000, horizon=14 and 5 folds it evaluated days 500-570
    and never touched days 570-1000 at all. Both model selection and the
    interval calibration were therefore computed from the middle of the history
    rather than the recent past — a measured contributor to the interval
    under-coverage documented in FORECASTING_METHODOLOGY.md.

    Detected by making only the tail predictable: an end-anchored fold set sees
    it, a midpoint-anchored one cannot.
    """
    n, horizon, n_folds = 1000, 14, 5
    y = np.full(n, 50.0)
    # A distinctive tail covering exactly the region the end-anchored folds
    # must evaluate: the last n_folds*horizon points.
    y[n - n_folds * horizon:] = 900.0

    res = walk_forward_validate(
        y, lambda: SeasonalNaiveForecaster(period=7), horizon=horizon,
        n_folds=n_folds,
    )
    assert len(res.fold_rmses) == n_folds

    # The first fold trains on the flat 50s and is tested on the 900s, so it is
    # badly wrong; later folds have seen the new level and recover. Neither
    # would happen if the folds sat at the midpoint, where every value is 50.
    assert res.fold_rmses[0] > 100.0, (
        "the first end-anchored fold should be evaluated on the distinctive "
        "tail — folds are not reaching the end of the series"
    )
    assert res.fold_rmses[-1] == pytest.approx(0.0, abs=1e-9), (
        "the final fold should train and test entirely inside the tail"
    )
    assert res.residuals_by_horizon.shape == (n_folds, horizon)
    assert res.scales_by_horizon.shape == (n_folds, horizon)


def test_conformal_multiplier_refuses_a_level_it_cannot_express():
    """With too few scores, the level is unreachable and must be declared so.

    This is the defect that caused the original 57% coverage. A per-horizon
    quantile of n residuals cannot exceed (n-1)/(n+1) coverage — 66.7% at five
    folds — yet the old code returned `np.quantile(col, 0.975)` and labelled it
    a 95% interval. The split-conformal rank ceil((N+1)*c) makes the
    impossibility explicit instead of silently returning the sample maximum.
    """
    rng = np.random.default_rng(0)
    resid = rng.normal(0, 1, size=(5, 1))
    scale = np.ones((5, 1))

    cal = conformal_multiplier(resid, scale, 0.95)
    assert not cal.achieved, "5 scores cannot express a 95% level"
    assert cal.rank > cal.n_scores
    assert "does not exist" in cal.note
    assert cal.q_hat == pytest.approx(float(np.abs(resid).max()))

    # Pooling across horizons is what makes the level reachable — the same
    # five folds at a 14-step horizon give 70 scores.
    wide = conformal_multiplier(rng.normal(0, 1, (5, 14)), np.ones((5, 14)), 0.95)
    assert wide.achieved
    assert wide.n_scores == 70
    assert wide.rank == int(np.ceil(71 * 0.95)) <= 70


def test_conformal_multiplier_recovers_a_known_scale_error():
    """If the model's stated scale is k times too narrow, `scale_ratio` must find k.

    THIS TEST PINS THE UNITS. `q_hat` multiplies a standard deviation, so a
    correctly scaled Gaussian model produces `q_hat = z = 1.96`, NOT 1.0. A
    draft of this project read `q_hat` against 1.0 and therefore reported a
    perfectly calibrated model as "2x too narrow" in four documents. The
    interpretable quantity is `scale_ratio = q_hat / z`, and it is asserted
    here explicitly so the confusion cannot come back.
    """
    rng = np.random.default_rng(5)
    truth = rng.normal(0, 1, size=(40, 20))

    for factor in (1.0, 2.5):
        cal = conformal_multiplier(truth, np.full_like(truth, 1.0 / factor), 0.95)
        assert cal.achieved
        assert cal.z_reference == pytest.approx(1.96, abs=0.01)
        # q_hat lives on the sigma scale...
        assert cal.q_hat == pytest.approx(factor * cal.z_reference, rel=0.15), (
            f"stated scale {factor}x too narrow but q_hat={cal.q_hat:.3f}"
        )
        # ...and scale_ratio is the number a human should read.
        assert cal.scale_ratio == pytest.approx(factor, rel=0.15), (
            f"scale_ratio {cal.scale_ratio:.3f} should recover the {factor}x error"
        )

    # A correctly scaled model must NOT be described as needing a correction.
    exact = conformal_multiplier(truth, np.ones_like(truth), 0.95)
    assert exact.scale_ratio == pytest.approx(1.0, rel=0.15)
    assert "about right" in exact.note, (
        f"a correctly scaled model was described as {exact.note!r}"
    )


def test_conformal_interval_is_calibrated_on_held_out_data():
    """End-to-end: the conformal interval must actually cover ~95%.

    The point of the whole construction, checked on a synthetic series with a
    known noise law rather than only on the backtest.
    """
    rng = np.random.default_rng(12)
    n = 600
    y = 100.0 + 10.0 * np.sin(np.arange(n) * 2 * np.pi / 7) + rng.normal(0, 8, n)
    train, test = y[:500], y[500:530]

    wf = walk_forward_validate(
        train, lambda: HoltWintersForecaster(period=7), horizon=10, n_folds=8
    )
    model = HoltWintersForecaster(period=7).fit(train)
    out = model.predict(30)

    profile, gamma = fit_scale_profile(wf.residuals_by_horizon, 30)
    cal = conformal_multiplier(
        wf.residuals_by_horizon,
        np.tile(profile[: wf.residuals_by_horizon.shape[1]],
                (wf.residuals_by_horizon.shape[0], 1)),
        0.95,
    )
    assert cal.achieved
    lower, upper = conformal_interval(out.point, profile, cal.q_hat)

    coverage = float(np.mean((test >= lower) & (test <= upper)))
    assert coverage >= 0.80, f"conformal interval covered only {coverage:.1%}"
    # The scale profile must be defined and non-decreasing over the FULL
    # horizon, not clamped at the CV horizon — the old bug gave a width ratio
    # of exactly 1.0000 between h=90 and h=14.
    assert len(profile) == 30
    assert profile[-1] >= profile[9]


def test_fitted_scale_exponent_beats_an_assumed_sqrt_on_a_stationary_series():
    """gamma must be FITTED, not assumed to be 0.5.

    Forecast error variance on a stationary series converges to the
    unconditional variance instead of accumulating, so the textbook sqrt(h)
    growth over-widens far horizons. Measured on the real series the exponent
    comes out near 0.1, not 0.5.
    """
    rng = np.random.default_rng(1)
    # Residuals with genuinely constant spread across the horizon.
    resid = rng.normal(0, 25.0, size=(30, 20))
    _profile, gamma = fit_scale_profile(resid, 90)
    assert gamma < 0.2, f"fitted gamma {gamma:.3f} should be near 0 on flat residuals"

    # ...and it must still detect real growth when it is there.
    growing = resid * np.sqrt(np.arange(1, 21))[None, :]
    _p2, gamma2 = fit_scale_profile(growing, 90)
    assert gamma2 == pytest.approx(0.5, abs=0.12), (
        f"fitted gamma {gamma2:.3f} should recover the sqrt(h) law when it holds"
    )


def test_walk_forward_rejects_too_short_series():
    with pytest.raises(ValueError, match="too short"):
        walk_forward_validate(
            np.arange(10.0), lambda: SeasonalNaiveForecaster(period=7), horizon=14
        )


def test_selection_reports_every_candidate_including_losers():
    """Rejected models must be reported; the rejections are the interesting part."""
    ds = generate_dataset(seed=3, n_days=500, n_counterparties=5)
    r = select_and_forecast(ds.daily, horizon_days=30, value_column="net_ex_receipts")
    names = {s.model_name for s in r.scores}
    assert names == {
        ForecastModelName.SEASONAL_NAIVE.value,
        ForecastModelName.HOLT_WINTERS.value,
        ForecastModelName.SARIMAX.value,
    }
    assert r.selected_model in names
    assert "RMSE" in r.selection_rationale


def test_prediction_intervals_are_ordered_and_widen():
    """Intervals must bracket the point estimate and grow with horizon."""
    ds = generate_dataset(seed=4, n_days=500, n_counterparties=5)
    r = select_and_forecast(ds.daily, horizon_days=60, value_column="net_ex_receipts")
    for p in r.path:
        assert p.lower <= p.point <= p.upper
    first = r.path[0].upper - r.path[0].lower
    last = r.path[-1].upper - r.path[-1].lower
    assert last > first, "uncertainty must accumulate with horizon"


# ---------------------------------------------------------------------------
# A.4 — credit risk
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def risk_models():
    ds = generate_dataset(
        seed=42, regime=Regime.ADVERSARIAL, n_days=1460, n_counterparties=25
    )
    data = build_features(ds.payments)
    return data, train_models(data, seed=42)


def test_features_use_no_future_information():
    """A row's features must derive only from strictly-earlier invoices.

    Checked structurally: the first `min_history` invoices of each counterparty
    must produce no rows at all, since there is not yet enough prior history.
    """
    ds = generate_dataset(seed=8, n_days=400, n_counterparties=4)
    data = build_features(ds.payments, min_history=3)
    per_cp = pd.Series(data.counterparty_ids).value_counts()
    for cp, total in ds.payments.groupby("counterparty_id").size().items():
        assert per_cp.get(cp, 0) <= total - 3


def test_prior_default_rate_is_a_proper_fraction(risk_models):
    data, _ = risk_models
    assert (data.X["prior_default_rate"] >= 0).all()
    assert (data.X["prior_default_rate"] <= 1).all()


def test_models_beat_chance_and_are_reported_honestly(risk_models):
    """AUC must exceed chance; the comparison must be reported either way."""
    _, models = risk_models
    for name in (RiskModelName.LOGISTIC_L2, RiskModelName.GBM):
        assert models[name].performance.roc_auc > 0.55
        lift = compare_to_baseline(models, name)
        assert lift.auc_lift == pytest.approx(
            models[name].performance.roc_auc - models[RiskModelName.RULE_BASELINE].performance.roc_auc
        )
        assert lift.verdict


def test_probabilities_are_calibrated(risk_models):
    """Predicted probabilities must track observed rates, not merely rank well.

    Calibration matters because the output is consumed as an actual
    probability downstream. Ranking-only quality would silently corrupt RaR.
    """
    _, models = risk_models
    for name in (RiskModelName.LOGISTIC_L2, RiskModelName.GBM):
        buckets = models[name].calibration
        assert len(buckets) >= 5
        errors = [abs(b.mean_predicted - b.observed_rate) for b in buckets]
        assert float(np.mean(errors)) < 0.08, (
            f"{name}: mean calibration error {np.mean(errors):.3f}"
        )


def test_accuracy_is_never_reported(risk_models):
    """Accuracy on imbalanced default data is misleading and must be absent."""
    _, models = risk_models
    perf = models[RiskModelName.GBM].performance
    assert not hasattr(perf, "accuracy")


def test_logistic_coefficients_have_finite_confidence_intervals(risk_models):
    """A singular information matrix silently produced NaN intervals before."""
    _, models = risk_models
    ci = models[RiskModelName.LOGISTIC_L2].coef_ci
    assert ci is not None
    finite = [v for v in ci.values() if np.isfinite(v[1]) and np.isfinite(v[2])]
    assert len(finite) >= 8, "most coefficients should have finite Wald intervals"
    for coef, lo, hi in finite:
        assert lo <= coef <= hi


def test_explanations_are_produced_and_ordered(risk_models):
    data, models = risk_models
    logit = models[RiskModelName.LOGISTIC_L2]
    row = data.X.iloc[[0]]
    contribs = explain_prediction(logit, row)
    assert contribs
    mags = [abs(c.contribution) for c in contribs]
    assert mags == sorted(mags, reverse=True), "contributions must be ranked by magnitude"
    text = rationale_from_contributions(0.42, contribs)
    assert "42" in text and contribs[0].feature.replace("_", " ") in text


def test_calibration_curve_buckets_are_well_formed():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1] * 10)
    p = np.linspace(0.01, 0.99, len(y))
    buckets = calibration_curve(y, p, n_buckets=5)
    assert all(b.n >= 2 for b in buckets)
    assert all(0.0 <= b.observed_rate <= 1.0 for b in buckets)


# ---------------------------------------------------------------------------
# A.5 — anomaly detection
# ---------------------------------------------------------------------------

def test_robust_z_is_not_masked_by_the_outlier_it_hunts():
    """The MAD-based score must flag an outlier that a plain z-score misses.

    This is the masking effect: one extreme value inflates the standard
    deviation enough to hide itself, which is precisely why the textbook
    z-score is the wrong tool here.
    """
    # Ordinary data with genuine spread, plus one extreme value. (Using
    # identical baseline values instead would make MAD exactly zero and hit
    # the documented degenerate branch rather than testing the masking effect.)
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(10.0, 1.0, 30), [10_000.0]])
    rz = robust_z_scores(x)
    plain_z = (x - x.mean()) / x.std(ddof=1)
    assert abs(rz[-1]) > 100, "robust z failed to flag an obvious outlier"
    assert abs(plain_z[-1]) < 6, "plain z should be masked here (that is the point)"


def test_robust_z_handles_degenerate_input():
    """More than half identical values makes MAD zero; must not return inf."""
    out = robust_z_scores(np.array([5.0] * 20))
    assert np.all(np.isfinite(out)) and np.all(out == 0.0)


def test_detectors_disagree_and_the_disagreement_is_reported():
    ds = generate_dataset(seed=15, n_days=500, n_counterparties=6)
    p = ds.payments.copy()
    report = detect_anomalies(
        p, amount_column="amount",
        feature_columns=["amount", "delay_days"], id_column="invoice_id",
    )
    assert len(report.flags) == len(p)
    c = report.comparison
    assert c.n_records == len(p)
    assert 0.0 <= c.jaccard_agreement <= 1.0
    assert "univariate" in c.commentary


# ---------------------------------------------------------------------------
# A.6 — unstructured
# ---------------------------------------------------------------------------

def test_receipt_classifier_matches_obvious_categories():
    clf = ReceiptClassifier()
    preds = clf.classify([
        "monthly office rent payment",
        "electricity bill for march",
        "salary disbursement to staff",
    ])
    assert preds[0].category == "rent"
    assert preds[1].category == "utilities"
    assert preds[2].category == "payroll"


def test_receipt_classifier_survives_ocr_corruption():
    """Character n-grams must tolerate character-level OCR noise.

    This is the specific reason char_wb was chosen over word n-grams.
    """
    clf = ReceiptClassifier()
    clean = clf.classify(["electricity bill"])[0]
    noisy = clf.classify(["e1ectricity bi11"])[0]
    assert noisy.category == clean.category


def test_receipt_classifier_flags_uncertainty():
    """A text matching nothing well must report low confidence, not guess loudly."""
    clf = ReceiptClassifier()
    pred = clf.classify(["zzzz qqqq wxyz"])[0]
    assert pred.confidence < 0.2


def test_stl_recovers_a_known_seasonal_signal():
    """A series built with a strong weekly cycle must decompose as strongly seasonal."""
    t = np.arange(400)
    seasonal = 50.0 * np.sin(2 * np.pi * t / 7)
    series = pd.Series(1000 + 0.5 * t + seasonal + np.random.default_rng(0).normal(0, 3, 400))
    d = decompose_seasonality(series, period=7)
    assert d.seasonal_variance_share > 0.10
    assert "STL" in d.interpretation
    np.testing.assert_allclose(
        d.seasonal_variance_share + d.trend_variance_share + d.residual_variance_share,
        1.0, atol=1e-9,
    )


def test_stl_reports_weak_seasonality_as_weak():
    """A series with no seasonal structure must not be described as seasonal."""
    series = pd.Series(np.random.default_rng(1).normal(100, 5, 400))
    d = decompose_seasonality(series, period=7)
    assert d.seasonal_variance_share < 0.5
    if d.seasonal_variance_share <= 0.10:
        assert "overstated" in d.interpretation


def test_t_test_detects_a_real_shift_and_reports_effect_size():
    rng = np.random.default_rng(2)
    series = pd.Series(np.concatenate([rng.normal(100, 10, 200), rng.normal(70, 10, 200)]))
    res = compare_periods(series, range(200), range(200, 400))
    assert res["p_value"] < 0.001
    assert res["difference"] < 0
    assert abs(res["cohens_d"]) > 1.0


def test_t_test_does_not_invent_a_shift():
    rng = np.random.default_rng(3)
    series = pd.Series(rng.normal(100, 10, 400))
    res = compare_periods(series, range(200), range(200, 400))
    assert res["p_value"] > 0.01
    assert abs(res["cohens_d"]) < 0.3
