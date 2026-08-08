"""Section B — credit risk scorer: a thin caller into A.4.

Per the architecture plan's Section B table, this is NOT a standalone rule
table any more. It delegates to `quant_core.risk_scoring` and its only real
job is assembling the fitted model's output into the `RiskScore` schema,
including the explainability fields §2.1 requires to reach the API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Optional
from uuid import UUID, uuid4

import pandas as pd

from app.schemas.core import RiskScore
from app.schemas.quant import RiskModelName
from app.services.quant_core.risk_scoring import (
    FittedRiskModel,
    RiskDataset,
    compare_to_baseline,
    explain_prediction,
    rationale_from_contributions,
)


def training_set_version(data: RiskDataset, model_name: RiskModelName) -> str:
    """Identify the fitted artifact, not just the config.

    §2.1 replaced `weighting_scheme_version: str = "v1"` with something that
    identifies a *fitted* model. A version string alone cannot: the same "v1"
    config fitted to different data produces a different scorer, and an audit
    trail that cannot distinguish them is not an audit trail.
    """
    digest = sha256(
        pd.util.hash_pandas_object(data.X, index=True).values.tobytes()
    ).hexdigest()[:12]
    return f"{model_name.value}:train-{digest}:n{len(data.X)}"


def score_obligation(
    *,
    business_id: UUID,
    fitted: FittedRiskModel,
    models: dict[RiskModelName, FittedRiskModel],
    data: RiskDataset,
    row_index: int,
    obligation_id: Optional[UUID] = None,
    counterparty_id: Optional[UUID] = None,
    days_until_due: float = 0.0,
    amount: float = 0.0,
    max_amount: float = 1.0,
) -> RiskScore:
    """Produce a full `RiskScore` for one obligation, with its explanation.

    `urgency_score` and `impact_score` are computed rather than assigned,
    per the §2.1 provenance rule:
      - urgency: proximity to the due date, mapped to [0,1] over a 90-day
        window. An overdue obligation saturates at 1.0.
      - impact: this obligation's share of the largest exposure in the book.

    `composite_score` combines default probability with impact multiplicatively
    rather than as a weighted sum, because expected loss IS probability times
    exposure — an additive blend would let a large but safe obligation score
    like a small but doomed one, which is not what the number is meant to mean.
    """
    row = data.X.iloc[[row_index]]
    prob = float(fitted.predict_proba(row)[0])
    contributions = explain_prediction(fitted, row)

    urgency = float(min(1.0, max(0.0, 1.0 - (days_until_due / 90.0))))
    impact = float(min(1.0, amount / max_amount)) if max_amount > 0 else 0.0
    composite = float(min(1.0, prob * (0.5 + 0.5 * impact)))

    return RiskScore(
        id=uuid4(),
        business_id=business_id,
        obligation_id=obligation_id,
        counterparty_id=counterparty_id,
        urgency_score=urgency,
        impact_score=impact,
        default_probability=prob,
        composite_score=composite,
        model_name=fitted.name,
        weighting_scheme_version=training_set_version(data, fitted.name),
        feature_contributions=contributions,
        baseline_comparison=compare_to_baseline(models, fitted.name)
        if fitted.name is not RiskModelName.RULE_BASELINE
        else None,
        rationale=rationale_from_contributions(prob, contributions),
        computed_at=datetime.now(timezone.utc),
    )
