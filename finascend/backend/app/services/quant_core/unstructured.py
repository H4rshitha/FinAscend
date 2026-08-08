"""A.6 — Unstructured data: receipt text classification and grounded seasonality.

Two capabilities, both chosen to demonstrate that unstructured input is
handled with a method rather than a regex.

1. **Vendor / category extraction by embedding similarity.** OCR'd receipt
   text is embedded and matched against a small labelled reference set by
   cosine similarity. Regex was rejected: it fails on the exact cases that
   matter (OCR noise, abbreviations, vendor names never seen before), and a
   rule list cannot generalize to a vendor it has no rule for. Similarity
   matching degrades gracefully — an unknown vendor lands near its closest
   known category and reports a low confidence rather than silently matching
   nothing.

   The embedding here is a character n-gram TF-IDF vector, not a neural
   sentence embedding. That is a deliberate trade, stated plainly: character
   n-grams are robust to OCR character-level corruption ("INV0ICE" vs
   "INVOICE"), need no model download or GPU, and are reproducible offline.
   A transformer embedding would capture semantics that TF-IDF cannot — that
   "cab fare" and "taxi journey" mean the same thing — and would be the right
   upgrade if semantic generalization mattered more than OCR robustness. The
   interface is written so the vectorizer can be swapped without touching
   callers.

2. **Seasonality grounded in STL decomposition.** The architecture plan's
   `market_intelligence` module described "seasonal trends" from "macro
   signals", which is unfalsifiable. This replaces it with
   `statsmodels.tsa.seasonal.STL`, which decomposes the actual series into
   trend, seasonal and residual components, and quantifies how much variance
   the seasonal component explains. A claim about seasonality that comes with
   a variance share can be checked; one that comes from a black box cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.seasonal import STL

# A small labelled reference set. Deliberately small: the point is to show
# that a handful of exemplars generalizes via similarity, which is what makes
# this different from an exhaustive rule table.
DEFAULT_REFERENCE_SET: dict[str, list[str]] = {
    "rent": ["monthly office rent", "premises lease payment", "shop rental invoice"],
    "payroll": ["salary disbursement", "staff wages", "employee payroll run"],
    "utilities": ["electricity bill", "water supply charges", "internet broadband bill"],
    "logistics": ["courier charges", "freight transport invoice", "delivery services"],
    "raw_materials": ["steel sheet purchase", "raw material supply", "component order"],
    "professional_fees": ["consulting services", "audit fee", "legal retainer invoice"],
    "travel": ["taxi fare receipt", "flight booking", "hotel accommodation"],
    "tax": ["gst payment challan", "income tax remittance", "tds deposit"],
}


@dataclass(frozen=True)
class CategoryPrediction:
    """A predicted category plus the evidence for it."""

    text: str
    category: str
    confidence: float
    runner_up: Optional[str]
    runner_up_score: float
    is_uncertain: bool


class ReceiptClassifier:
    """Classify OCR'd receipt text by similarity to a reference set.

    Method: character n-gram TF-IDF (`sklearn.TfidfVectorizer`,
    `analyzer="char_wb"`, n-grams 3-5) followed by cosine similarity against
    each reference exemplar; the predicted category is the argmax.

    `char_wb` rather than word n-grams because OCR errors corrupt characters
    within words. A word-level model treats "INV0ICE" as a completely unseen
    token, while a character model still matches most of its n-grams.
    """

    def __init__(
        self,
        reference_set: Optional[dict[str, list[str]]] = None,
        *,
        uncertainty_margin: float = 0.05,
    ) -> None:
        self.reference_set = reference_set or DEFAULT_REFERENCE_SET
        self.uncertainty_margin = uncertainty_margin
        self._labels: list[str] = []
        self._corpus: list[str] = []
        for cat, examples in self.reference_set.items():
            for ex in examples:
                self._corpus.append(ex)
                self._labels.append(cat)
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=1
        )
        self._reference_matrix = self._vectorizer.fit_transform(self._corpus)

    def classify(self, texts: Sequence[str]) -> list[CategoryPrediction]:
        """Classify each text, reporting confidence and the runner-up.

        The runner-up is returned because the *margin* between top two is what
        indicates reliability. A top score of 0.4 is confident when the
        runner-up is 0.1 and a coin-flip when the runner-up is 0.39, and
        reporting only the winner hides that distinction — exactly the kind of
        false certainty this project is meant to avoid.
        """
        X = self._vectorizer.transform(list(texts))
        sims = cosine_similarity(X, self._reference_matrix)

        out: list[CategoryPrediction] = []
        for i, text in enumerate(texts):
            # Best similarity per category, not per exemplar, so a category
            # with more exemplars is not mechanically favoured.
            by_cat: dict[str, float] = {}
            for j, lab in enumerate(self._labels):
                by_cat[lab] = max(by_cat.get(lab, 0.0), float(sims[i, j]))
            ranked = sorted(by_cat.items(), key=lambda kv: kv[1], reverse=True)
            top, top_score = ranked[0]
            runner, runner_score = (ranked[1] if len(ranked) > 1 else (None, 0.0))
            out.append(
                CategoryPrediction(
                    text=text,
                    category=top,
                    confidence=top_score,
                    runner_up=runner,
                    runner_up_score=runner_score,
                    is_uncertain=(top_score - runner_score) < self.uncertainty_margin,
                )
            )
        return out


@dataclass(frozen=True)
class SeasonalDecomposition:
    """STL output plus the variance shares that make it checkable."""

    period: int
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    seasonal_variance_share: float
    trend_variance_share: float
    residual_variance_share: float
    interpretation: str


def decompose_seasonality(
    series: pd.Series, *, period: int = 7, robust: bool = True
) -> SeasonalDecomposition:
    """Decompose a series into trend + seasonal + residual via STL.

    Method: `statsmodels.tsa.seasonal.STL` (Cleveland et al. 1990), a
    LOESS-based decomposition. Chosen over classical additive decomposition
    because STL allows the seasonal component to *evolve* over time, which a
    fixed seasonal-index method cannot, and because `robust=True` downweights
    outliers so a single shock does not distort the whole seasonal estimate —
    directly relevant here, since the A.0 generator injects regime shocks.

    Variance shares are reported so the seasonality claim is quantified rather
    than asserted. If the seasonal component explains 2% of variance, saying
    "this business is highly seasonal" is false, and the number says so.

    Args:
        series: the observed series; must be longer than 2 * period.
        period: seasonal period (7 for weekly structure in daily data).
        robust: use robust LOESS weights.

    Returns:
        `SeasonalDecomposition` with components and variance attribution.
    """
    y = pd.Series(series).astype(float).reset_index(drop=True)
    if len(y) < 2 * period:
        raise ValueError(f"need at least {2 * period} points for STL at period={period}")

    result = STL(y, period=period, robust=robust).fit()
    trend = np.asarray(result.trend, dtype=float)
    seasonal = np.asarray(result.seasonal, dtype=float)
    resid = np.asarray(result.resid, dtype=float)

    # Variance shares are normalized to sum to 1. They are NOT an exact
    # variance decomposition — the components are correlated, so their
    # variances do not add to the total. This is a share-of-explained-
    # variation summary, and calling it that rather than "variance explained"
    # is the honest description.
    variances = np.array([np.var(seasonal), np.var(trend), np.var(resid)])
    shares = variances / variances.sum() if variances.sum() > 0 else np.zeros(3)

    seasonal_share = float(shares[0])
    if seasonal_share > 0.30:
        verdict = "strongly seasonal"
    elif seasonal_share > 0.10:
        verdict = "moderately seasonal"
    else:
        verdict = "weakly seasonal — seasonality claims would be overstated"

    return SeasonalDecomposition(
        period=period,
        trend=trend,
        seasonal=seasonal,
        residual=resid,
        seasonal_variance_share=seasonal_share,
        trend_variance_share=float(shares[1]),
        residual_variance_share=float(shares[2]),
        interpretation=(
            f"STL(period={period}, robust={robust}): seasonal component accounts for "
            f"{seasonal_share:.1%} of component variation, trend {shares[1]:.1%}, "
            f"residual {shares[2]:.1%}. Assessment: {verdict}."
        ),
    )


def compare_periods(
    series: pd.Series,
    pre_index: Sequence[int],
    post_index: Sequence[int],
) -> dict[str, float]:
    """Welch's t-test comparing two periods of a series.

    Used instead of asserting that "activity slowed after X". Welch's variant
    (`equal_var=False`) rather than Student's, because the two periods
    routinely have different variances — a downturn changes volatility as well
    as level, and assuming equal variance would understate the p-value exactly
    when the claim is most interesting.

    Returns:
        Dict with both means, the difference, the t statistic, the p-value and
        Cohen's d. The effect size is included because with a few hundred
        daily observations almost any difference reaches significance, and a
        p-value alone would overstate how much the change matters.
    """
    from scipy import stats as _st

    a = np.asarray(pd.Series(series).iloc[list(pre_index)], dtype=float)
    b = np.asarray(pd.Series(series).iloc[list(post_index)], dtype=float)
    t_stat, p = _st.ttest_ind(a, b, equal_var=False)

    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    cohens_d = float((b.mean() - a.mean()) / pooled_sd) if pooled_sd > 0 else 0.0

    return {
        "pre_mean": float(a.mean()),
        "post_mean": float(b.mean()),
        "difference": float(b.mean() - a.mean()),
        "t_statistic": float(t_stat),
        "p_value": float(p),
        "cohens_d": cohens_d,
        "n_pre": float(len(a)),
        "n_post": float(len(b)),
    }
