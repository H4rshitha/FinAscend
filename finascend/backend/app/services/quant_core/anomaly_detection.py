"""A.5 — Anomaly and duplicate detection: DBSCAN plus a statistical layer.

The architecture plan specifies DBSCAN for ambiguity resolution. That is kept.
What is added here is a second and a third opinion, because a single
unsupervised detector on unlabelled data cannot be validated — there is no
ground truth to score it against, so "it found 40 anomalies" is not a result.
Comparing detectors that fail in *different* ways is the closest thing to
validation available without labels, and where they disagree is informative.

THREE DETECTORS
---------------
1. **Robust z-score (MAD-based).** Univariate, on amount. Uses the median and
   the median absolute deviation rather than mean and standard deviation,
   because the mean and SD are themselves dragged toward the outliers being
   hunted — a single 100x transaction inflates the SD enough to hide itself.
   This is the masking effect, and it is why the textbook z-score is the wrong
   tool for the job it is most often used for.

2. **Isolation Forest.** Multivariate, unsupervised. Isolates points by random
   splitting; anomalies need fewer splits to isolate. Chosen over a
   density-based alternative because it does not require a distance metric
   over mixed-scale features and it handles the multivariate case that the
   robust z-score cannot see at all.

3. **DBSCAN.** Density clustering, used for near-duplicate resolution rather
   than pure outlier flagging: duplicates form tight clusters, and a genuine
   one-off lands in the noise label (-1).

They answer different questions, so disagreement is expected and is reported
rather than reconciled away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.schemas.quant import AnomalyFlag, DetectorComparison

# 0.6745 is the 0.75 quantile of the standard normal. Dividing the MAD by it
# makes the MAD a consistent estimator of sigma for Gaussian data, so a robust
# z-score is on the same scale as an ordinary z-score and the familiar
# thresholds (2, 3) still mean what a reader expects.
MAD_TO_SIGMA = 0.6745


def robust_z_scores(values: Sequence[float]) -> np.ndarray:
    """MAD-based robust z-scores.

        z_i = 0.6745 * (x_i - median(x)) / MAD(x)
        MAD = median(|x - median(x)|)

    Chosen over the standard z-score because the standard version uses the
    mean and SD, both of which are corrupted by the outliers it is meant to
    detect. The MAD has a 50% breakdown point: half the data would have to be
    contaminated before it misleads.

    Degenerate case: if more than half the values are identical the MAD is
    zero and the score is undefined. Zeros are returned rather than infinities,
    because "every point is infinitely anomalous" is not a useful answer.
    """
    x = np.asarray(values, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad <= 0:
        return np.zeros_like(x)
    return MAD_TO_SIGMA * (x - med) / mad


@dataclass(frozen=True)
class AnomalyReport:
    flags: list[AnomalyFlag]
    comparison: DetectorComparison


def detect_anomalies(
    df: pd.DataFrame,
    *,
    amount_column: str = "amount",
    feature_columns: Optional[Sequence[str]] = None,
    id_column: str = "invoice_id",
    z_threshold: float = 3.5,
    contamination: float = 0.05,
    seed: int = 42,
) -> AnomalyReport:
    """Run all three detectors and report where they agree and disagree.

    Args:
        df: transactions to screen.
        amount_column: column for the univariate robust z-score.
        feature_columns: columns for the multivariate detectors; defaults to
            all numeric columns.
        id_column: identifier carried into the flags.
        z_threshold: |robust z| above which a point is flagged. 3.5 is the
            conventional MAD-based cutoff (Iglewicz & Hoaglin 1993), chosen
            rather than 3.0 because the MAD scaling makes 3.5 the value with
            comparable false-positive behaviour on Gaussian data.
        contamination: Isolation Forest's expected anomaly fraction. This is
            the detector's weak point — it is an assumption, not an estimate,
            and it directly sets how many points get flagged.
        seed: reproducibility for Isolation Forest.

    Returns:
        `AnomalyReport` with per-record flags and a detector comparison.
    """
    if feature_columns is None:
        feature_columns = [
            c for c in df.select_dtypes(include=[np.number]).columns
        ]
    feats = df[list(feature_columns)].fillna(0.0).to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(feats)

    # --- 1. robust z on amount ---
    rz = robust_z_scores(df[amount_column].to_numpy(dtype=float))
    rz_flag = np.abs(rz) > z_threshold

    # --- 2. Isolation Forest ---
    iso = IsolationForest(
        contamination=contamination, random_state=seed, n_estimators=200
    ).fit(scaled)
    iso_pred = iso.predict(scaled)          # -1 anomaly, +1 normal
    iso_score = iso.score_samples(scaled)   # lower = more anomalous
    iso_flag = iso_pred == -1

    # --- 3. DBSCAN for near-duplicate structure ---
    db = DBSCAN(eps=0.5, min_samples=3).fit(scaled)
    labels = db.labels_

    ids = df[id_column].astype(str).to_numpy() if id_column in df.columns else np.arange(len(df)).astype(str)
    flags = [
        AnomalyFlag(
            record_id=str(ids[i]),
            robust_z=float(rz[i]),
            flagged_by_robust_z=bool(rz_flag[i]),
            flagged_by_isolation_forest=bool(iso_flag[i]),
            isolation_forest_score=float(iso_score[i]),
            dbscan_label=int(labels[i]),
        )
        for i in range(len(df))
    ]

    both = int(np.sum(rz_flag & iso_flag))
    rz_only = int(np.sum(rz_flag & ~iso_flag))
    iso_only = int(np.sum(iso_flag & ~rz_flag))
    union = both + rz_only + iso_only
    jaccard = both / union if union else 1.0

    commentary = (
        f"Robust z flagged {int(rz_flag.sum())}, Isolation Forest flagged "
        f"{int(iso_flag.sum())}, {both} in common (Jaccard {jaccard:.2f}). "
        "The disagreement is structural, not noise: the robust z-score is "
        "univariate on amount, so it can only ever flag unusually large or "
        "small values. Isolation Forest is multivariate, so it flags records "
        "that are unremarkable on every single axis but implausible in "
        "combination — a normal amount on unusual terms from a counterparty "
        "with an unusual history. Those are different failure modes and each "
        "detector is blind to the other's. Low agreement is therefore the "
        "expected result and would be more suspicious if it were high. "
        f"DBSCAN assigned {int(np.sum(labels == -1))} records to noise; those "
        "are the genuine one-offs as opposed to near-duplicates."
    )

    return AnomalyReport(
        flags=flags,
        comparison=DetectorComparison(
            n_records=len(df),
            n_robust_z_only=rz_only,
            n_isolation_forest_only=iso_only,
            n_both=both,
            jaccard_agreement=float(jaccard),
            commentary=commentary,
        ),
    )
