"""A.7 — business-level insolvency risk: ruin probability and balance-sheet distress.

WHY THIS IS NOT ALREADY COVERED BY RunwayAtRisk
-----------------------------------------------
RaR answers *"how long until the cash runs out?"* and reports a quantile of the
first-passage time. That is a different question from *"how likely is this
business to fail in the next quarter?"*, and the difference is not cosmetic:

  * RaR is **censored at the horizon**. A business that never runs out inside
    90 days contributes `days_to_zero = 90`, indistinguishable from one that
    hits zero on day 90 exactly. The quantile is well defined; the failure
    probability it implies is not readable off it.
  * A quantile **cannot be scored against an outcome**. "95% RaR = 42 days"
    is not a prediction that can be right or wrong about any single business.
    "P(ruin within 90 days) = 0.18" is, and that is what makes it checkable.
  * `probability_of_shortfall` on `RunwayAtRisk` is the closest existing
    number, but it is reported at exactly one horizon and never validated. The
    curve, its hazard, and — critically — its **calibration against realized
    outcomes** are what this module adds.

The credit-risk model in A.4 is also a different thing: it scores the
probability that a **counterparty** defaults on the business. This scores the
probability that the **business itself** fails. Confusing the two would be a
serious error and they are kept in separate modules for that reason.

TWO INDEPENDENT VIEWS, DELIBERATELY NOT BLENDED
-----------------------------------------------
1. **Cash-flow ruin probability** — first-passage probability from A.2's Monte
   Carlo paths. Validated here against realized ruin on data with a known
   generating process.
2. **Altman Z''-score** — a published balance-sheet model applied to supplied
   inputs. It is *not* fitted or validated by this project and says so.

They are reported side by side and never averaged. Averaging a validated
estimate into an unvalidated one produces a composite with no validity claim,
and hides which half moved it. Where they disagree, the disagreement is the
finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

import numpy as np
from scipy import stats

from app.schemas.solvency import (
    AltmanZScore,
    BalanceSheetInput,
    BankruptcyRisk,
    HazardPoint,
    InputProvenance,
    RuinCalibration,
    RuinCalibrationBucket,
    RuinCurvePoint,
    ZScoreZone,
)

# Altman (1995) Z''-score for non-manufacturers / emerging markets. Published
# coefficients, NOT refitted here — there is no default-labelled panel of small
# business balance sheets in this project to refit them on, and refitting them
# on synthetic data would produce numbers that look like Altman's and mean
# something else.
Z2_COEFFICIENTS = (6.56, 3.26, 6.72, 1.05)
Z2_SAFE_THRESHOLD = 2.60
Z2_DISTRESS_THRESHOLD = 1.10


# ---------------------------------------------------------------------------
# 1. Ruin probability from simulated paths
# ---------------------------------------------------------------------------

def ruin_indicator_matrix(balances: np.ndarray) -> np.ndarray:
    """Cumulative first-passage indicator: has this path ruined by day t?

    `np.minimum.accumulate` along the time axis gives each path's running
    minimum, so comparing it to zero yields a matrix that, once True, stays
    True. That is the definition of ruin as an **absorbing** event, and it is
    the whole reason this is not a terminal-value check: a business that misses
    payroll in week three has failed, whatever a receivable does for the
    balance in week six. Testing `balances[:, t] <= 0` instead would let a
    later inflow un-fail it.

    A KNOWN AND MEASURED BIAS
    -------------------------
    Daily-step paths are monitored only at day boundaries, so an excursion that
    dips below zero and recovers within a single day is invisible. The estimate
    is therefore biased LOW relative to continuous monitoring — measured at 2-3
    percentage points for the configurations in the test suite. This is not a
    defect to be apologised for; it is the Broadie-Glasserman-Kou discrete-
    barrier effect, equivalent to monitoring a barrier shifted by
    `0.5826 * sigma * sqrt(dt)`, and
    `test_monte_carlo_ruin_matches_the_analytic_formula` checks the simulator
    against the corrected closed form to within half a percentage point. Daily
    resolution is also the honest one for the underlying claim: a business is
    not insolvent because its balance dipped for six hours between a debit and
    a credit that both cleared the same day.
    """
    return np.minimum.accumulate(np.asarray(balances, dtype=float), axis=1) <= 0.0


def ruin_curve(
    balances: np.ndarray,
    *,
    horizons: Optional[Sequence[int]] = None,
    z: float = 1.96,
) -> list[RuinCurvePoint]:
    """P(ruin at or before day t) for each requested horizon, with its MC error.

    The standard error is binomial — `sqrt(p(1-p)/N)` — because the estimate is
    a mean of independent Bernoulli draws across Monte Carlo iterations. This is
    a different estimator from the bootstrap used for RaR's SE, and correctly
    so: RaR is a quantile, whose sampling distribution has no closed form worth
    trusting under censoring, while a proportion's does.

    The interval is clipped to [0, 1] rather than reported raw. A Wald interval
    on a proportion near the boundary can extend past it, and a reported
    probability of -0.02 is not a conservative statement, it is a broken one.

    Args:
        balances: (n_iterations, horizon_days) simulated balance paths.
        horizons: days at which to report. Defaults to a weekly-ish grid.
        z: normal quantile for the interval; 1.96 gives ~95%.

    Returns:
        One `RuinCurvePoint` per horizon, ascending.
    """
    arr = np.asarray(balances, dtype=float)
    n_iter, n_days = arr.shape
    ruined = ruin_indicator_matrix(arr)

    if horizons is None:
        horizons = [h for h in (7, 14, 30, 45, 60, 90, 120, 180) if h <= n_days]
        if n_days not in horizons:
            horizons.append(n_days)

    out: list[RuinCurvePoint] = []
    for h in sorted({int(min(max(h, 1), n_days)) for h in horizons}):
        p = float(ruined[:, h - 1].mean())
        se = float(np.sqrt(max(p * (1.0 - p), 0.0) / n_iter))
        out.append(
            RuinCurvePoint(
                horizon_days=h,
                ruin_probability=p,
                standard_error=se,
                ci_lower=float(np.clip(p - z * se, 0.0, 1.0)),
                ci_upper=float(np.clip(p + z * se, 0.0, 1.0)),
            )
        )
    return out


def hazard_table(balances: np.ndarray, *, bin_days: int = 15) -> list[HazardPoint]:
    """Conditional ruin rate per interval, given survival to its start.

    Reported beside the cumulative curve because they answer different
    questions. A cumulative curve rising smoothly to 30% can be produced either
    by steady attrition or by a single cliff at a payroll date, and only the
    hazard distinguishes them — which is the difference between "this business
    is gradually weakening" and "this business fails on the 30th unless
    something changes before then."

    `n_at_risk` shrinks as paths are absorbed, so late intervals are estimated
    from fewer paths and their hazard is noisier. The count is reported rather
    than smoothed away so that thinness is visible.
    """
    arr = np.asarray(balances, dtype=float)
    n_iter, n_days = arr.shape
    ruined = ruin_indicator_matrix(arr)

    out: list[HazardPoint] = []
    for start in range(0, n_days, bin_days):
        end = min(start + bin_days, n_days)
        # At risk = not yet ruined at the instant the interval opens.
        at_risk = n_iter if start == 0 else int((~ruined[:, start - 1]).sum())
        failed = int((~ruined[:, start - 1] & ruined[:, end - 1]).sum()) if start > 0 \
            else int(ruined[:, end - 1].sum())
        out.append(
            HazardPoint(
                interval_start_day=start + 1,
                interval_end_day=end,
                n_at_risk=at_risk,
                n_failed=failed,
                hazard=float(failed / at_risk) if at_risk > 0 else 0.0,
            )
        )
    return out


def expected_days_to_ruin(balances: np.ndarray) -> Optional[float]:
    """E[time to ruin | ruin occurs], or None when no path ruined.

    Conditional, not unconditional. The unconditional mean is dominated by
    surviving paths censored at the horizon, which makes it a function of how
    long the simulation was run rather than of how fast the business fails —
    lengthening the horizon would "improve" it.
    """
    arr = np.asarray(balances, dtype=float)
    ruined = ruin_indicator_matrix(arr)
    ever = ruined[:, -1]
    if not ever.any():
        return None
    first = np.argmax(ruined[ever], axis=1) + 1
    return float(first.mean())


# ---------------------------------------------------------------------------
# 2. Altman Z''-score
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BalanceSheet:
    """Inputs to the Z''-score, with what is optional made explicit.

    `total_assets` and `total_liabilities` have no defaults because there is no
    honest default for them; a Z-score computed from an assumed asset base is
    fiction with a decimal point. `retained_earnings` and `ebit` may be None,
    and the score is refused rather than zero-filled — zero retained earnings
    is a real and meaningfully different claim from unknown retained earnings.
    """

    total_assets: float
    total_liabilities: float
    current_assets: float
    current_liabilities: float
    retained_earnings: Optional[float] = None
    ebit: Optional[float] = None
    equity: Optional[float] = None


class InsufficientBalanceSheet(ValueError):
    """Not enough was supplied to compute a Z''-score honestly."""


def altman_z_double_prime(bs: BalanceSheet) -> AltmanZScore:
    """Compute the Z''-score and its zone from a supplied balance sheet.

        Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

    `equity` is derived as `total_assets - total_liabilities` when not supplied,
    and tagged `DERIVED_FROM_LEDGER` — that is an accounting identity, not an
    estimate, so it is the one input that can be filled in without weakening
    the result.

    Raises:
        InsufficientBalanceSheet: a required input is missing or degenerate.
            Refusing is the point. Every alternative — defaulting retained
            earnings to zero, assuming EBIT from cash flow — produces a number
            that reads like a measurement and is not one.
    """
    if bs.total_assets is None or bs.total_assets <= 0:
        raise InsufficientBalanceSheet(
            "total_assets must be positive; every Altman ratio is scaled by it"
        )
    if bs.total_liabilities is None or bs.total_liabilities <= 0:
        raise InsufficientBalanceSheet(
            "total_liabilities must be positive; X4 divides by it"
        )
    missing = [n for n, v in (("retained_earnings", bs.retained_earnings),
                              ("ebit", bs.ebit)) if v is None]
    if missing:
        raise InsufficientBalanceSheet(
            f"missing {', '.join(missing)}; refusing to substitute zero, which is a "
            "different and much stronger claim than 'unknown'"
        )

    inputs: list[BalanceSheetInput] = []
    approximated: list[str] = []

    def record(name: str, value: float, prov: InputProvenance, note: str | None = None):
        inputs.append(BalanceSheetInput(name=name, value=float(value),
                                        provenance=prov, note=note))
        if prov is InputProvenance.APPROXIMATED:
            approximated.append(name)

    equity = bs.equity
    if equity is None:
        equity = bs.total_assets - bs.total_liabilities
        record("equity", equity, InputProvenance.DERIVED_FROM_LEDGER,
               "accounting identity: assets - liabilities, not an estimate")
    else:
        record("equity", equity, InputProvenance.SUPPLIED_BY_BUSINESS)

    for name, value in (
        ("total_assets", bs.total_assets),
        ("total_liabilities", bs.total_liabilities),
        ("current_assets", bs.current_assets),
        ("current_liabilities", bs.current_liabilities),
        ("retained_earnings", bs.retained_earnings),
        ("ebit", bs.ebit),
    ):
        record(name, value, InputProvenance.SUPPLIED_BY_BUSINESS)

    x1 = (bs.current_assets - bs.current_liabilities) / bs.total_assets
    x2 = bs.retained_earnings / bs.total_assets
    x3 = bs.ebit / bs.total_assets
    x4 = equity / bs.total_liabilities

    a, b, c, d = Z2_COEFFICIENTS
    z = a * x1 + b * x2 + c * x3 + d * x4

    zone = (
        ZScoreZone.SAFE if z > Z2_SAFE_THRESHOLD
        else ZScoreZone.DISTRESS if z < Z2_DISTRESS_THRESHOLD
        else ZScoreZone.GREY
    )

    return AltmanZScore(
        z_score=float(z),
        zone=zone,
        x1_working_capital_to_assets=float(x1),
        x2_retained_earnings_to_assets=float(x2),
        x3_ebit_to_assets=float(x3),
        x4_equity_to_liabilities=float(x4),
        inputs=inputs,
        approximated_input_names=approximated,
    )


# ---------------------------------------------------------------------------
# 3. Calibration — the check that makes the probability worth reporting
# ---------------------------------------------------------------------------

def _first_passage_probability(
    *, opening: float, mu: float, sigma: float, horizon: int,
    n_iterations: int, rng: np.random.Generator,
    parameter_uncertainty: bool, n_history: int,
) -> float:
    """Monte Carlo P(ruin within `horizon`) for a Gaussian net-flow process.

    PLUG-IN VS PREDICTIVE
    ---------------------
    With `parameter_uncertainty = False` the simulation uses the point
    estimates (mu, sigma) as if they were the truth. That is the ordinary
    plug-in approach and it is **systematically overconfident**: it propagates
    the process noise but discards the estimation noise, so the simulated
    spread is narrower than the real predictive spread and tail probabilities
    come out too small.

    With it True, each iteration draws its own parameters from their sampling
    distributions before drawing the path —

        sigma^2 ~ sigma_hat^2 * (n-1) / chi2_{n-1}
        mu      ~ Normal(mu_hat, sigma / sqrt(n))

    — which is the standard Bayesian predictive construction under a reference
    prior. Both modes are offered because the difference between them is
    measurable, and `validate_ruin_calibration` measures it rather than
    asserting it.
    """
    if parameter_uncertainty and n_history > 2:
        df = n_history - 1
        sig = sigma * np.sqrt(df / rng.chisquare(df, size=n_iterations))
        mus = rng.normal(mu, sig / np.sqrt(n_history))
        steps = rng.normal(
            mus[:, None], sig[:, None], size=(n_iterations, horizon)
        )
    else:
        steps = rng.normal(mu, sigma, size=(n_iterations, horizon))

    balances = opening + np.cumsum(steps, axis=1)
    return float((np.minimum.accumulate(balances, axis=1) <= 0.0)[:, -1].mean())


def validate_ruin_calibration(
    *,
    n_businesses: int = 400,
    horizon_days: int = 90,
    history_days: int = 180,
    n_iterations: int = 2_000,
    seed: int = 20260809,
    parameter_uncertainty: bool = True,
    n_buckets: int = 10,
) -> RuinCalibration:
    """Are the predicted ruin probabilities honest? Measured, not asserted.

    THE EXPERIMENT
    --------------
    For each of `n_businesses` independently generated firms:

    1. Draw its **true** daily net-flow drift and volatility, and an opening
       balance. The spread across firms is deliberately wide — a corpus of
       near-identical businesses would produce near-identical predictions, and
       ROC-AUC would be undefined in practice because there is nothing to rank.
    2. Show the estimator only a **history** of `history_days` draws, and let
       it estimate the parameters. This is the step that makes the exercise
       non-trivial: the model never sees the true parameters, so what is being
       validated is the estimator plus the simulator, which is what actually
       ships.
    3. Predict P(ruin within `horizon_days`) by Monte Carlo from the estimates.
    4. Play the future out **once** from the TRUE parameters and record whether
       ruin actually occurred.

    Then compare predictions to outcomes. One realized path per business, not
    an average over many, because that is the situation a real business is in
    — it lives its future once — and averaging would measure something easier
    than the thing being claimed.

    WHY THREE METRICS
    -----------------
    * **Brier score** — mean squared error of the probability. Catches a model
      whose probabilities are systematically off.
    * **Brier skill score** — Brier against a baseline that always predicts the
      base rate. This is the one that matters: a raw Brier of 0.08 sounds good
      and is *worse than useless* if the base rate alone scores 0.07. Positive
      skill is the claim; the raw score cannot make it.
    * **ROC-AUC** — can the model rank a doomed business above a healthy one?
      A model can be perfectly calibrated and useless at ranking (predict the
      base rate for everyone: Brier skill 0, AUC 0.5), so neither metric
      substitutes for the other.

    Returns:
        `RuinCalibration` with the curve and all three metrics.
    """
    # TWO INDEPENDENT STREAMS, WHICH MATTERS MORE THAN IT LOOKS.
    #
    # `world_rng` draws the firms, their histories and their realized futures;
    # `sim_rng` drives only the predictor's Monte Carlo. Sharing one stream
    # would make the world itself depend on how many draws the predictor
    # happened to consume — and the two `parameter_uncertainty` modes consume
    # different numbers. The plug-in and predictive runs would then be scored
    # against DIFFERENT sets of businesses with different base rates, so any
    # difference between them would confound the estimator change with a change
    # of dataset. Split streams make the comparison a controlled one.
    world_rng = np.random.default_rng(seed)
    sim_rng = np.random.default_rng(seed + 977)

    predicted = np.empty(n_businesses)
    realized = np.empty(n_businesses, dtype=bool)

    for b in range(n_businesses):
        # Heterogeneous firms. Drift is centred slightly negative so a
        # meaningful fraction genuinely fails inside the horizon; a corpus with
        # a 1% base rate makes every metric noise-dominated.
        true_mu = float(world_rng.normal(-40.0, 260.0))
        true_sigma = float(np.exp(world_rng.normal(7.4, 0.45)))
        opening = float(np.exp(world_rng.normal(11.0, 0.7)))

        history = world_rng.normal(true_mu, true_sigma, size=history_days)
        mu_hat = float(history.mean())
        sigma_hat = float(history.std(ddof=1))

        future = world_rng.normal(true_mu, true_sigma, size=horizon_days)
        path = opening + np.cumsum(future)
        realized[b] = bool((np.minimum.accumulate(path) <= 0.0)[-1])

        predicted[b] = _first_passage_probability(
            opening=opening, mu=mu_hat, sigma=sigma_hat, horizon=horizon_days,
            n_iterations=n_iterations, rng=sim_rng,
            parameter_uncertainty=parameter_uncertainty, n_history=history_days,
        )

    y = realized.astype(float)
    base_rate = float(y.mean())
    brier = float(np.mean((predicted - y) ** 2))
    brier_base = float(np.mean((base_rate - y) ** 2))
    skill = float(1.0 - brier / brier_base) if brier_base > 0 else 0.0

    auc = _roc_auc(y, predicted)

    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    buckets: list[RuinCalibrationBucket] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Right-closed on the final bucket so p == 1.0 is not dropped.
        sel = (predicted >= lo) & ((predicted < hi) | (hi >= 1.0) & (predicted <= hi))
        if not sel.any():
            continue
        buckets.append(
            RuinCalibrationBucket(
                lower=float(lo), upper=float(hi), n=int(sel.sum()),
                mean_predicted=float(predicted[sel].mean()),
                observed_frequency=float(y[sel].mean()),
            )
        )

    return RuinCalibration(
        n_businesses=n_businesses,
        horizon_days=horizon_days,
        realized_ruin_rate=base_rate,
        mean_predicted_probability=float(predicted.mean()),
        brier_score=brier,
        brier_skill_score=skill,
        roc_auc=auc,
        buckets=buckets,
        seed=seed,
    )


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U identity, with ties at half credit.

    Computed from ranks rather than by calling scikit-learn so the definition
    is visible: AUC is exactly P(score of a positive > score of a negative),
    with ties counted as half. Returns 0.5 when one class is absent, since
    ranking is undefined with nothing to rank against.
    """
    y = np.asarray(y_true, dtype=float)
    n_pos, n_neg = float((y == 1).sum()), float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = stats.rankdata(scores)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# 4. The composite assessment
# ---------------------------------------------------------------------------

def assess_bankruptcy_risk(
    balances: np.ndarray,
    *,
    as_of: date,
    random_seed: int,
    balance_sheet: Optional[BalanceSheet] = None,
    horizons: Optional[Sequence[int]] = None,
    hazard_bin_days: int = 15,
    calibration: Optional[RuinCalibration] = None,
) -> BankruptcyRisk:
    """Assemble the cash-flow and balance-sheet views into one report.

    Args:
        balances: (n_iterations, horizon_days) from `simulate_cash_paths`.
        as_of: the vantage date the simulation was run from.
        random_seed: the simulation's seed, carried for reproducibility.
        balance_sheet: optional. Omitted -> no Z''-score, rather than a
            defaulted one.
        horizons / hazard_bin_days: reporting grids.
        calibration: an existing calibration result to attach. Not computed
            here by default because it is an expensive property of the *method*
            rather than of this business, so recomputing it per request would
            be both slow and misleading about what it measures.

    Returns:
        A `BankruptcyRisk` whose two components are reported separately.
    """
    arr = np.asarray(balances, dtype=float)
    n_iter, n_days = arr.shape

    curve = ruin_curve(arr, horizons=horizons)
    hazards = hazard_table(arr, bin_days=hazard_bin_days)
    headline = curve[-1].ruin_probability if curve else 0.0

    caveats = [
        "Ruin is defined as cash reaching zero at any point in the horizon, not "
        "at the end of it. A business rescued by a late receivable has still "
        "missed the payment it could not make.",
        "This is the probability that THIS BUSINESS fails. It is unrelated to "
        "the counterparty default probabilities in A.4's credit model.",
    ]

    altman = None
    if balance_sheet is not None:
        try:
            altman = altman_z_double_prime(balance_sheet)
        except InsufficientBalanceSheet as exc:
            caveats.append(f"no Z''-score: {exc}")

    agreement = None
    if altman is not None:
        # Compare on direction only. Any attempt to express the two on one
        # scale would be inventing a mapping between a validated probability
        # and an external score whose calibration on this population is
        # unknown.
        cash_flow_worried = headline >= 0.20
        balance_sheet_worried = altman.zone in (ZScoreZone.DISTRESS, ZScoreZone.GREY)
        if cash_flow_worried and balance_sheet_worried:
            agreement = (
                "Both views agree the business is under stress: the cash-flow "
                f"simulation puts P(ruin) at {headline:.1%} over {n_days} days and "
                f"the balance sheet sits in the {altman.zone} zone."
            )
        elif cash_flow_worried and not balance_sheet_worried:
            agreement = (
                f"They disagree. P(ruin) is {headline:.1%} while the balance sheet "
                "is in the safe zone — the signature of a SOLVENT BUT ILLIQUID "
                "business: assets exceed liabilities, but not in a form that "
                "settles this month's payroll. Solvency is not liquidity, and it "
                "is the liquidity view that predicts the missed payment."
            )
        elif balance_sheet_worried and not cash_flow_worried:
            agreement = (
                f"They disagree. Near-term cash looks adequate (P(ruin) "
                f"{headline:.1%}) while the balance sheet is in the "
                f"{altman.zone} zone — near-term comfort financed by a "
                "structural position that is deteriorating. The cash-flow view "
                "has a 90-day horizon and cannot see past it."
            )
        else:
            agreement = (
                f"Both views are benign: P(ruin) {headline:.1%} and a balance "
                f"sheet in the {altman.zone} zone."
            )
        caveats.append(
            "The Z''-score uses Altman's published coefficients applied to supplied "
            "inputs. It is NOT fitted or validated by this project, unlike the ruin "
            "probability, which is why the two are reported separately and never "
            "averaged into one score."
        )

    if calibration is None:
        caveats.append(
            "No calibration result attached. The ruin probability is only as good "
            "as its calibration — run validate_ruin_calibration() and read it "
            "before treating this number as a probability rather than an index."
        )

    return BankruptcyRisk(
        as_of=as_of,
        horizon_days=n_days,
        ruin_curve=curve,
        hazard=hazards,
        headline_ruin_probability=headline,
        expected_days_to_ruin_given_ruin=expected_days_to_ruin(arr),
        altman=altman,
        agreement=agreement,
        calibration=calibration,
        n_iterations=n_iter,
        random_seed=random_seed,
        caveats=caveats,
    )
