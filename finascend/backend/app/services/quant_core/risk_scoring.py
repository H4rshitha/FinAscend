"""A.4 — Credit / default risk model, validated and explainable.

The deck's original answer to its own "black-box trust barrier" was to avoid
ML entirely and declare the scorer rule-based, therefore auditable. That buys
explainability by giving up predictive power, and it never establishes that
the rules are any good. This module takes the harder position: fit a real
model, then *earn* the trust with a calibration curve, per-feature
attribution, and an explicit measured comparison against the rules baseline.

If the model does not beat the rules, that is reported. A negative lift is a
finding, not a failure to suppress.

THE PREDICTION TASK
-------------------
For each invoice, predict P(default) using only the counterparty's payment
history **strictly before that invoice was issued**. The temporal cut is
enforced in `build_features`, and it is the whole game: computing an
"historical on-time rate" over the full dataset would include the very invoice
being predicted, and would produce an ROC-AUC near 1.0 that means nothing.

MODELS
------
  * `rule_baseline`  — the deck's approach, made concrete: a score
    proportional to (days overdue x amount). Kept as the thing to beat.
  * `logistic_l2`    — regularized logistic regression. Primary model, chosen
    for interpretable coefficients that carry confidence intervals.
  * `gbm`            — HistGradientBoostingClassifier. Comparison model, able
    to capture interactions the logistic model cannot.

METRICS
-------
ROC-AUC (ranking quality), Brier score and a calibration curve (are the
probabilities *right*, not just correctly ordered), and log-loss. Accuracy is
deliberately never reported: with a ~8% default rate, predicting "never
defaults" scores 92% accuracy while being useless, and quoting that number
would be the single most misleading thing this module could do.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.schemas.quant import (
    BaselineLift,
    CalibrationBucket,
    FeatureContribution,
    ModelPerformance,
    RiskModelName,
)

FEATURE_NAMES = [
    "n_prior_invoices",
    "prior_default_rate",
    "mean_prior_delay",
    "std_prior_delay",
    "max_prior_delay",
    "delay_trend",
    "days_since_last_invoice",
    "log_amount",
    "amount_vs_counterparty_mean",
    "payment_terms_days",
]


@dataclass(frozen=True)
class RiskDataset:
    """Feature matrix, labels, and the metadata needed to explain a score."""

    X: pd.DataFrame
    y: np.ndarray
    counterparty_ids: np.ndarray
    invoice_ids: np.ndarray


def build_features(payments: pd.DataFrame, min_history: int = 3) -> RiskDataset:
    """Engineer RFM-style features from each counterparty's PRIOR history.

    Features (all computed on invoices issued strictly earlier than the target):
      - recency:   days_since_last_invoice
      - frequency: n_prior_invoices
      - monetary:  log_amount, amount_vs_counterparty_mean
      - behaviour: prior_default_rate, mean/std/max prior delay, delay_trend

    `delay_trend` is the OLS slope of delay against invoice sequence number,
    included because a counterparty whose delays are *lengthening* is a
    different risk from one with the same mean delay that is stable. A model
    seeing only the mean cannot distinguish them.

    NO LOOK-AHEAD
    -------------
    Rows are processed in issue order per counterparty and every statistic is
    accumulated from strictly-earlier invoices only. This is the difference
    between an honest AUC and a meaningless one.

    Args:
        payments: the generator's payment frame.
        min_history: invoices required before a counterparty yields a row.
            Below this, the behavioural features are too noisy to be real
            signal and would mostly encode "this counterparty is new".

    Returns:
        `RiskDataset` with aligned X, y, and identifiers.
    """
    df = payments.sort_values(["counterparty_id", "issue_date"]).reset_index(drop=True)
    rows, labels, cps, invs = [], [], [], []

    for cp_id, grp in df.groupby("counterparty_id", sort=False):
        grp = grp.reset_index(drop=True)
        delays: list[float] = []
        defaults: list[int] = []
        amounts: list[float] = []
        last_issue: Optional[pd.Timestamp] = None

        for i in range(len(grp)):
            row = grp.iloc[i]
            if len(defaults) >= min_history:
                d = np.array(delays, dtype=float) if delays else np.array([0.0])
                seq = np.arange(len(d), dtype=float)
                if len(d) >= 2 and np.ptp(seq) > 0:
                    trend = float(np.polyfit(seq, d, 1)[0])
                else:
                    trend = 0.0
                mean_amt = float(np.mean(amounts)) if amounts else float(row["amount"])
                rows.append(
                    {
                        "n_prior_invoices": float(len(defaults)),
                        "prior_default_rate": float(np.mean(defaults)),
                        "mean_prior_delay": float(d.mean()),
                        "std_prior_delay": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                        "max_prior_delay": float(d.max()),
                        "delay_trend": trend,
                        "days_since_last_invoice": float(
                            (row["issue_date"] - last_issue).days
                        )
                        if last_issue is not None
                        else 0.0,
                        "log_amount": float(np.log1p(row["amount"])),
                        "amount_vs_counterparty_mean": float(
                            row["amount"] / mean_amt if mean_amt > 0 else 1.0
                        ),
                        "payment_terms_days": float(
                            (row["due_date"] - row["issue_date"]).days
                        ),
                    }
                )
                labels.append(0 if bool(row["paid"]) else 1)
                cps.append(str(cp_id))
                invs.append(str(row["invoice_id"]))

            # Accumulate AFTER emitting the row, so this invoice never informs
            # its own prediction.
            defaults.append(0 if bool(row["paid"]) else 1)
            if bool(row["paid"]):
                delays.append(float(row["delay_days"]))
            amounts.append(float(row["amount"]))
            last_issue = row["issue_date"]

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return RiskDataset(
        X=X,
        y=np.array(labels, dtype=int),
        counterparty_ids=np.array(cps),
        invoice_ids=np.array(invs),
    )


def rule_baseline_score(X: pd.DataFrame) -> np.ndarray:
    """The deck's rules approach, made concrete so it can be measured.

    score ∝ mean_prior_delay * log_amount, normalized to [0, 1].

    This is the "days overdue x amount" heuristic the architecture plan names
    as the baseline. It is a genuine attempt, not a straw man: delay and
    exposure are the two things a human credit controller actually looks at.
    Its weakness is that it is monotone in two variables and cannot represent
    "long delays but never actually defaults", which is a common and
    commercially important pattern.
    """
    raw = X["mean_prior_delay"].to_numpy() * X["log_amount"].to_numpy()
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)


@dataclass
class FittedRiskModel:
    """A fitted model plus everything needed to score and explain."""

    name: RiskModelName
    model: object
    scaler: Optional[StandardScaler]
    performance: ModelPerformance
    calibration: list[CalibrationBucket]
    coef_ci: Optional[dict[str, tuple[float, float, float]]] = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.name is RiskModelName.RULE_BASELINE:
            return rule_baseline_score(X)
        # Pass the DataFrame through unchanged when there is no scaler: the GBM
        # was fitted on named columns, and handing it a bare array triggers a
        # feature-name mismatch warning and silently relies on column order.
        Xv = self.scaler.transform(X) if self.scaler is not None else X
        return self.model.predict_proba(Xv)[:, 1]


def calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_buckets: int = 10
) -> list[CalibrationBucket]:
    """Predicted vs. observed default rate, bucketed by predicted probability.

    Quantile buckets rather than equal-width: with a skewed score distribution
    most equal-width bins would be empty, and an empty bin's "observed rate" is
    not a measurement. Buckets with fewer than 2 observations are dropped
    rather than reported as noise.
    """
    edges = np.quantile(y_prob, np.linspace(0, 1, n_buckets + 1))
    edges = np.unique(edges)
    out: list[CalibrationBucket] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if i == len(edges) - 2 else y_prob < hi)
        if mask.sum() < 2:
            continue
        out.append(
            CalibrationBucket(
                bucket_lower=float(lo),
                bucket_upper=float(hi),
                n=int(mask.sum()),
                mean_predicted=float(y_prob[mask].mean()),
                observed_rate=float(y_true[mask].mean()),
            )
        )
    return out


def _logistic_coef_ci(
    model: LogisticRegression, X_scaled: np.ndarray, confidence: float = 0.95
) -> dict[str, tuple[float, float, float]]:
    """Wald confidence intervals for logistic coefficients.

    Method: the observed information matrix for logistic regression is
    X' W X with W = diag(p(1-p)); its inverse is the asymptotic covariance of
    the coefficients, and se = sqrt(diag(.)). The interval is
    beta +/- z * se.

    CAVEAT, stated because it matters for how these intervals should be read:
    this is the *unpenalized* Wald interval, while the fitted coefficients are
    L2-penalized. Penalization shrinks coefficients toward zero and reduces
    their true variance, so these intervals are conservative (too wide) and
    are not centred on an unbiased estimate. They are reported as an
    indication of which features are precisely estimated, not as exact
    frequentist coverage. A bootstrap would give honest intervals under
    penalization and is the right upgrade if these are ever used for
    inference rather than explanation.
    """
    p = model.predict_proba(X_scaled)[:, 1]
    W = p * (1 - p)
    Xd = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
    try:
        # pinv rather than inv: a zero-variance feature standardizes to an
        # all-zero column, which makes X'WX exactly singular and sends every
        # interval to NaN. The pseudo-inverse degrades gracefully instead,
        # returning a zero row for the degenerate feature.
        cov = np.linalg.pinv(Xd.T * W @ Xd)
        diag = np.diag(cov)[1:]
        # Numerical noise can push a diagonal entry very slightly negative;
        # clip before the square root so it becomes 0 rather than NaN.
        se = np.sqrt(np.clip(diag, 0.0, None))
    except np.linalg.LinAlgError:
        se = np.full(X_scaled.shape[1], np.nan)
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    coefs = model.coef_[0]
    return {
        name: (float(c), float(c - z * s), float(c + z * s))
        for name, c, s in zip(FEATURE_NAMES, coefs, se)
    }


def train_models(
    data: RiskDataset,
    *,
    test_size: float = 0.3,
    seed: int = 42,
) -> dict[RiskModelName, FittedRiskModel]:
    """Fit the rules baseline, logistic regression and GBM on the same split.

    The split is **stratified by label and grouped nowhere** — deliberately a
    random split rather than a temporal one, because the features already
    encode time-ordering and the question here is "can these features separate
    defaults", not "does the relationship drift". A temporal split would
    conflate the two, and the drift question is answered separately by the
    Section C backtest.

    Args:
        data: output of `build_features`.
        test_size: held-out fraction.
        seed: reproducibility.

    Returns:
        Mapping model name -> `FittedRiskModel`, each carrying its own
        performance and calibration measured on the same test set.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.X, data.y, test_size=test_size, random_state=seed, stratify=data.y
    )

    out: dict[RiskModelName, FittedRiskModel] = {}

    def _score(name: RiskModelName, prob_te: np.ndarray, model, scaler, ci=None):
        perf = ModelPerformance(
            model_name=name,
            roc_auc=float(roc_auc_score(y_te, prob_te)),
            brier_score=float(brier_score_loss(y_te, prob_te)),
            log_loss=float(log_loss(y_te, np.clip(prob_te, 1e-9, 1 - 1e-9))),
            n_train=int(len(y_tr)),
            n_test=int(len(y_te)),
        )
        out[name] = FittedRiskModel(
            name=name,
            model=model,
            scaler=scaler,
            performance=perf,
            calibration=calibration_curve(y_te, prob_te),
            coef_ci=ci,
        )

    # --- rules baseline ---
    _score(RiskModelName.RULE_BASELINE, rule_baseline_score(X_te), None, None)

    # --- logistic regression (L2) ---
    scaler = StandardScaler().fit(X_tr)
    Xtr_s, Xte_s = scaler.transform(X_tr), scaler.transform(X_te)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # class_weight is deliberately NOT "balanced".
        #
        # Balancing improves ranking on an imbalanced problem, but it does so
        # by reweighting the likelihood, which shifts the fitted intercept and
        # destroys the probability scale. Measured on this data: balanced
        # weighting predicted 0.38-0.60 where the observed default rate was
        # 0.08-0.22, and more than doubled the Brier score (0.244 vs 0.108).
        #
        # Calibration matters more than ranking here because the output is
        # consumed as an actual probability — it feeds the Monte Carlo engine
        # and, through it, the optimizer. A well-ranked but badly scaled
        # probability would silently corrupt RaR. Ranking quality is still
        # reported via ROC-AUC, which is invariant to monotone rescaling.
        logit = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=2000,
            random_state=seed,
        ).fit(Xtr_s, y_tr)
    _score(
        RiskModelName.LOGISTIC_L2,
        logit.predict_proba(Xte_s)[:, 1],
        logit,
        scaler,
        _logistic_coef_ci(logit, Xtr_s),
    )

    # --- gradient-boosted trees ---
    gbm = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        max_depth=3,          # shallow: the dataset is small and defaults rare
        l2_regularization=1.0,
        random_state=seed,
    ).fit(X_tr, y_tr)
    _score(RiskModelName.GBM, gbm.predict_proba(X_te)[:, 1], gbm, None)

    return out


def compare_to_baseline(
    models: dict[RiskModelName, FittedRiskModel], candidate: RiskModelName
) -> BaselineLift:
    """Measure whether the added complexity earned its place.

    Reported as-is in both directions. A model that fails to beat
    (days overdue x amount) should be described as failing to beat it.
    """
    base = models[RiskModelName.RULE_BASELINE].performance.roc_auc
    cand = models[candidate].performance.roc_auc
    lift = cand - base
    if lift > 0.05:
        verdict = (
            f"{candidate.value} beats the rules baseline by {lift:+.3f} AUC — "
            "the complexity is earned."
        )
    elif lift > 0.0:
        verdict = (
            f"{candidate.value} edges the baseline by only {lift:+.3f} AUC. "
            "Marginal; the rules baseline remains a defensible choice given it "
            "needs no training data and no model governance."
        )
    else:
        verdict = (
            f"{candidate.value} does NOT beat the rules baseline "
            f"({lift:+.3f} AUC). Reported as measured: on this data the added "
            "complexity is not justified."
        )
    return BaselineLift(
        baseline_roc_auc=base,
        model_roc_auc=cand,
        auc_lift=lift,
        verdict=verdict,
    )


def explain_prediction(
    fitted: FittedRiskModel, x_row: pd.DataFrame
) -> list[FeatureContribution]:
    """Per-feature attribution for a single scored invoice.

    Logistic: contribution = coefficient x standardized value, which is the
    feature's additive effect on the log-odds and is exactly interpretable.
    Confidence intervals on the coefficient are carried through.

    GBM: SHAP values via `shap.TreeExplainer` when available. SHAP is used
    rather than impurity-based feature importance because importance is a
    global, model-level statistic and cannot answer "why is *this* invoice
    scored high", which is the question a user reviewing one decision asks.
    Falls back to a permutation-free zero attribution if shap is unavailable,
    and says so rather than inventing numbers.
    """
    contribs: list[FeatureContribution] = []

    if fitted.name is RiskModelName.LOGISTIC_L2 and fitted.scaler is not None:
        xs = fitted.scaler.transform(x_row)[0]
        coefs = fitted.model.coef_[0]
        for i, name in enumerate(FEATURE_NAMES):
            c = float(coefs[i] * xs[i])
            ci = fitted.coef_ci.get(name) if fitted.coef_ci else None
            contribs.append(
                FeatureContribution(
                    feature=name,
                    value=float(x_row.iloc[0][name]),
                    contribution=c,
                    ci_lower=float(ci[1] * xs[i]) if ci and np.isfinite(ci[1]) else None,
                    ci_upper=float(ci[2] * xs[i]) if ci and np.isfinite(ci[2]) else None,
                    direction="increases_risk" if c > 0 else "decreases_risk",
                )
            )
    elif fitted.name is RiskModelName.GBM:
        try:
            import shap

            explainer = shap.TreeExplainer(fitted.model)
            vals = np.asarray(explainer.shap_values(x_row))
            if vals.ndim == 3:
                vals = vals[0, :, 1]
            elif vals.ndim == 2:
                vals = vals[0]
            for i, name in enumerate(FEATURE_NAMES):
                c = float(vals[i])
                contribs.append(
                    FeatureContribution(
                        feature=name,
                        value=float(x_row.iloc[0][name]),
                        contribution=c,
                        direction="increases_risk" if c > 0 else "decreases_risk",
                    )
                )
        except Exception:
            return []
    else:
        # Rules baseline: the "explanation" is the rule itself.
        for name in ("mean_prior_delay", "log_amount"):
            contribs.append(
                FeatureContribution(
                    feature=name,
                    value=float(x_row.iloc[0][name]),
                    contribution=float(x_row.iloc[0][name]),
                    direction="increases_risk",
                )
            )

    contribs.sort(key=lambda c: abs(c.contribution), reverse=True)
    return contribs


def rationale_from_contributions(
    probability: float, contributions: list[FeatureContribution], top_k: int = 3
) -> str:
    """Generate the human-readable rationale FROM the attributions.

    §2.1 requires `RiskScore.rationale` to be generated from
    `feature_contributions` rather than written by a rule template. That
    distinction matters: a template sentence can stay confident while the
    model changes underneath it, whereas this sentence cannot be produced
    without the model actually having attributed the score that way.
    """
    if not contributions:
        return f"Estimated default probability {probability:.1%}. No attribution available."
    parts = []
    for c in contributions[:top_k]:
        word = "raises" if c.direction == "increases_risk" else "lowers"
        parts.append(f"{c.feature.replace('_', ' ')} ({c.value:,.2f}) {word} the estimate")
    return (
        f"Estimated default probability {probability:.1%}. Main drivers: "
        + "; ".join(parts)
        + "."
    )
