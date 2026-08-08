"""Turn extracted receipt fields into domain records, then screen for duplicates.

This is the join between the OCR path and the rest of the system. The chain is

    image -> ocr_service -> A.6 ReceiptClassifier -> normalize -> A.5 dedup

and each stage is a function on the previous stage's *declared* output type, so
none of them knows how the image was read. Swapping EasyOCR for a cloud OCR
changes the first arrow and nothing else.

TWO RULES THAT MATTER MORE THAN THE PLUMBING
--------------------------------------------
**A record is never fabricated to fill a gap.** If OCR could not read the total,
`normalize` refuses to build a record rather than defaulting the amount to zero
or guessing from the subtotal. A zero-amount outflow is not a harmless
placeholder — it enters the ledger, it reaches the optimizer, and it changes an
allocation while looking like data. Refusal surfaces as a `NormalizationError`
that names the missing field.

**Everything carries its provenance.** `source_type` is `RECEIPT_OCR`, and the
confidence that survived extraction travels with the record, so a downstream
consumer can tell a crisp scan from a guess off a blurred photo. A record that
cannot say how it was obtained is indistinguishable from one that was typed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional, Sequence
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL

import numpy as np
import pandas as pd

from app.schemas.base import SourceType
from app.schemas.core import Outflow
from app.services.ingestion.ocr_service import ReceiptFields
from app.services.quant_core.unstructured import CategoryPrediction, ReceiptClassifier

# Payment terms assumed when the receipt states none. Named, not inlined,
# because it is an assumption about a document rather than something read off
# it — and the receipt template says "Payment due within 30 days".
DEFAULT_TERMS_DAYS = 30

# Below this the classifier's own margin says the category is a coin flip, so
# the record is still created but flagged for review rather than silently
# filed under a category nobody checked.
REVIEW_CONFIDENCE = 0.20


class NormalizationError(ValueError):
    """A required field was unreadable, so no record can honestly be built."""


@dataclass(frozen=True)
class NormalizedReceipt:
    """An `Outflow` plus the evidence trail that produced it."""

    outflow: Outflow
    category_prediction: CategoryPrediction
    extraction_confidence: float
    needs_review: bool
    review_reasons: list[str]
    fields: ReceiptFields


def _deterministic_id(business_id: UUID, vendor: str, invoice_no: str) -> UUID:
    """Stable id from (business, vendor, invoice number).

    Deterministic rather than random so that re-ingesting the same receipt
    produces the same id. That makes exact re-submission idempotent and leaves
    the A.5 near-duplicate screen to do the harder job it is actually for:
    catching the same expense entered twice with a different invoice number or
    a slightly different amount.
    """
    return uuid5(NAMESPACE_URL, f"finascend/receipt/{business_id}/{vendor}/{invoice_no}")


def normalize(
    fields: ReceiptFields,
    *,
    business_id: UUID,
    classifier: Optional[ReceiptClassifier] = None,
    currency_check: bool = True,
) -> NormalizedReceipt:
    """Build an `Outflow` from extracted fields, or refuse and say why.

    A receipt evidences money *going out*, so it normalizes to `Outflow`. The
    inbound direction is not inferred from receipt text: an invoice the business
    ISSUED and a receipt it RECEIVED can look nearly identical to OCR, and
    getting the sign wrong would flip a liability into an asset. Direction is
    the caller's to state, and this function only handles the payable case it
    can actually justify.

    Args:
        fields: output of `ocr_service.extract_fields`.
        business_id: owning business.
        classifier: A.6 classifier; constructed with defaults if omitted.
        currency_check: cross-check the total against subtotal + tax when both
            were read, and flag the record when they disagree.

    Raises:
        NormalizationError: when vendor, total or date could not be read.
    """
    missing = [
        name
        for name, value in (
            ("vendor_name", fields.vendor_name),
            ("total_amount", fields.total_amount),
            ("issue_date", fields.issue_date),
        )
        if value is None
    ]
    if missing:
        raise NormalizationError(
            f"cannot build a record: OCR did not recover {', '.join(missing)}. "
            "Refusing rather than defaulting — a placeholder amount reaches the "
            "optimizer looking like data."
        )

    clf = classifier or ReceiptClassifier()
    # The classifier is given the whole OCR text, not just the vendor name.
    # The line-item description ("Monthly office premises rent") is what carries
    # the category signal; the vendor name alone is a proper noun the reference
    # set has never seen.
    prediction = clf.classify([fields.raw_text.replace("\n", " ")])[0]

    reasons: list[str] = []
    if prediction.is_uncertain:
        reasons.append(
            f"category margin {prediction.confidence - prediction.runner_up_score:.3f} "
            f"between {prediction.category} and {prediction.runner_up}"
        )
    if prediction.confidence < REVIEW_CONFIDENCE:
        reasons.append(f"low category confidence {prediction.confidence:.3f}")

    extraction_conf = (
        float(np.mean(list(fields.field_confidence.values())))
        if fields.field_confidence
        else 0.0
    )
    if extraction_conf < 0.60:
        reasons.append(f"low OCR field confidence {extraction_conf:.3f}")

    if currency_check and fields.tax_amount is not None and fields.tax_amount > 0:
        # A receipt is internally redundant: total = subtotal + tax. When the
        # tax line was read, the total can be checked against it instead of
        # trusted. This catches the specific OCR failure that matters most — a
        # lost decimal point, which is wrong by a factor of 100 and otherwise
        # looks like a perfectly plausible number.
        #
        # The band is [1%, 40%] and the lower bound does the work. Indian GST
        # slabs are 5/12/18/28%, so any NON-ZERO tax line implying 0.15% means
        # one of the two figures is out by orders of magnitude. An earlier
        # version used a lower bound of 0, which made the check vacuous against
        # exactly the failure it exists for: a total inflated 100x leaves the
        # implied rate near zero, not above the ceiling. The zero-tax case is
        # excluded above rather than absorbed into the band, because a genuinely
        # exempt receipt is a different thing from a misread one.
        implied_subtotal = fields.total_amount - fields.tax_amount
        if implied_subtotal <= 0:
            reasons.append("tax exceeds total — one of the two was misread")
        else:
            implied_rate = float(fields.tax_amount / implied_subtotal)
            if not 0.01 <= implied_rate <= 0.40:
                reasons.append(
                    f"implied tax rate {implied_rate:.2%} is outside the "
                    "plausible 1–40% band — total or tax was misread"
                )

    invoice_no = fields.invoice_number or f"UNKNOWN-{uuid4().hex[:8]}"
    if fields.invoice_number is None:
        reasons.append("invoice number unreadable; a synthetic reference was assigned")

    outflow = Outflow(
        id=_deterministic_id(business_id, fields.vendor_name, invoice_no),
        business_id=business_id,
        amount=Decimal(fields.total_amount).quantize(Decimal("0.01")),
        due_date=fields.issue_date + timedelta(days=DEFAULT_TERMS_DAYS),
        paid_date=None,
        counterparty_id=None,
        counterparty_name=fields.vendor_name,
        category=prediction.category,
        source_type=SourceType.RECEIPT_OCR,
        source_reference=invoice_no,
        is_recurring=False,
        is_duplicate_of=None,
        created_at=datetime.now(timezone.utc),
    )

    return NormalizedReceipt(
        outflow=outflow,
        category_prediction=prediction,
        extraction_confidence=extraction_conf,
        needs_review=bool(reasons),
        review_reasons=reasons,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# A.5 duplicate screening
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DuplicateFinding:
    record_id: str
    is_exact_duplicate: bool
    duplicate_of: Optional[str]
    dbscan_label: int
    robust_z: float
    flagged_by_isolation_forest: bool
    reason: str


def screen_for_duplicates(
    records: Sequence[NormalizedReceipt],
    *,
    seed: int = 42,
    amount_tolerance: float = 0.01,
    day_tolerance: int = 3,
) -> list[DuplicateFinding]:
    """Screen normalized receipts for duplicates via A.5, in two passes.

    TWO PASSES, BECAUSE THEY CATCH DIFFERENT THINGS
    -----------------------------------------------
    1. **Exact match** on (vendor, amount, due date within `day_tolerance`).
       This is the common real case — the same receipt photographed twice, or
       submitted by two people — and it needs no clustering to find. Running a
       density model at it would be using a statistical method where an
       equality check is both correct and explainable.
    2. **`anomaly_detection.detect_anomalies`** (A.5: robust z, Isolation
       Forest, DBSCAN) over the numeric feature space. This is what catches the
       near-duplicates pass 1 cannot see: the same expense entered twice with
       a transposed amount or a different invoice number. DBSCAN's noise label
       (-1) marks genuine one-offs; a shared cluster label marks records that
       sit on top of each other.

    Reporting both is deliberate. A cluster label alone does not distinguish
    "identical" from "similar", and the exact check alone misses everything
    that was retyped rather than re-photographed.
    """
    if not records:
        return []

    rows = []
    for r in records:
        o = r.outflow
        rows.append(
            {
                "invoice_id": str(o.id),
                "amount": float(o.amount),
                "due_epoch_days": float(pd.Timestamp(o.due_date).value // 86_400_000_000_000),
                "vendor_hash": float(abs(hash(o.counterparty_name)) % 10_000),
                "extraction_confidence": r.extraction_confidence,
            }
        )
    df = pd.DataFrame(rows)

    # --- pass 1: exact/near-exact key match ---
    exact_of: dict[int, int] = {}
    for i in range(len(records)):
        for j in range(i):
            a, b = records[i].outflow, records[j].outflow
            if a.counterparty_name != b.counterparty_name:
                continue
            if abs(float(a.amount) - float(b.amount)) > amount_tolerance:
                continue
            if abs((a.due_date - b.due_date).days) > day_tolerance:
                continue
            exact_of[i] = j
            break

    # --- pass 2: A.5 ---
    from app.services.quant_core.anomaly_detection import detect_anomalies

    report = detect_anomalies(
        df,
        amount_column="amount",
        feature_columns=["amount", "due_epoch_days", "vendor_hash"],
        id_column="invoice_id",
        seed=seed,
    )

    findings = []
    for i, flag in enumerate(report.flags):
        dup = exact_of.get(i)
        if dup is not None:
            reason = (
                f"exact duplicate of {records[dup].outflow.source_reference}: same "
                f"vendor, amount within {amount_tolerance}, due date within "
                f"{day_tolerance} days"
            )
        elif flag.dbscan_label >= 0:
            reason = (
                f"near-duplicate candidate: DBSCAN placed it in cluster "
                f"{flag.dbscan_label} with other records rather than in noise"
            )
        else:
            reason = "no duplicate signal: DBSCAN assigned it to noise (a one-off)"
        findings.append(
            DuplicateFinding(
                record_id=flag.record_id,
                is_exact_duplicate=dup is not None,
                duplicate_of=str(records[dup].outflow.id) if dup is not None else None,
                dbscan_label=flag.dbscan_label,
                robust_z=flag.robust_z,
                flagged_by_isolation_forest=flag.flagged_by_isolation_forest,
                reason=reason,
            )
        )
    return findings


def ingest_receipt_images(
    images: Sequence[object],
    *,
    business_id: UUID,
    engine: Optional[str] = None,
    classifier: Optional[ReceiptClassifier] = None,
) -> tuple[list[NormalizedReceipt], list[DuplicateFinding], list[str]]:
    """The whole chain: images in, records and duplicate findings out.

    Returns:
        (records, duplicate findings, rejection messages). Rejections are
        returned rather than raised so one unreadable receipt in a batch does
        not discard the rest, and so the caller can see exactly which images
        failed and why.
    """
    from app.services.ingestion.ocr_service import extract_fields, resolve_engine

    eng = resolve_engine(engine)
    clf = classifier or ReceiptClassifier()

    records: list[NormalizedReceipt] = []
    rejections: list[str] = []
    for idx, image in enumerate(images):
        try:
            fields = extract_fields(eng.read(image))
            records.append(normalize(fields, business_id=business_id, classifier=clf))
        except NormalizationError as exc:
            rejections.append(f"image {idx}: {exc}")

    return records, screen_for_duplicates(records), rejections
