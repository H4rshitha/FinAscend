"""Integration: a receipt image goes in, a checked domain record comes out.

These tests run the whole chain — image, OCR, A.6 classification, normalization,
A.5 duplicate screening — against receipts whose contents are known exactly, so
each stage is scored rather than merely executed.

They are marked `ocr` and skipped when the engine's weights are unavailable,
because a missing model download is an environment problem and should not read
as a code failure. Everything that does not need the engine (normalization,
refusal behaviour, duplicate screening) is tested without it.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from uuid import uuid4

import numpy as np
import pytest

from app.schemas.base import SourceType
from app.services.ingestion.normalizer import (
    NormalizationError,
    ingest_receipt_images,
    normalize,
    screen_for_duplicates,
)
from app.services.ingestion.ocr_service import (
    OcrLine,
    OcrResult,
    ReceiptFields,
    extract_fields,
    merge_into_rows,
    resolve_engine,
)
from app.services.ingestion.receipt_generator import (
    Difficulty,
    generate_receipt,
    render_clean,
    make_truth,
)

BUSINESS = uuid4()


def _engine_or_skip():
    try:
        engine = resolve_engine("easyocr")
        engine._get_reader()
        return engine
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"EasyOCR weights unavailable: {exc}")


# ---------------------------------------------------------------------------
# The end-to-end claim
# ---------------------------------------------------------------------------

@pytest.mark.ocr
@pytest.mark.parametrize("tier", [Difficulty.CLEAN, Difficulty.MODERATE])
def test_image_in_record_out_matches_ground_truth(tier):
    """A receipt image becomes an `Outflow` whose fields match what it said.

    CLEAN and MODERATE only. The HARD tier is deliberately excluded from the
    pass/fail assertion because its measured total-amount accuracy is 50% —
    asserting on it would either encode a failing expectation or force a
    tolerance so loose it checks nothing. HARD is *measured* in
    `accuracy_report`, which is the right place for a number that is not a
    binary. Hiding a 50% result behind a lenient assertion here would be the
    dishonest option.
    """
    engine = _engine_or_skip()
    rng = np.random.default_rng(11)
    truth, image = generate_receipt(0, tier, rng)

    records, duplicates, rejections = ingest_receipt_images(
        [image], business_id=BUSINESS, engine="easyocr"
    )

    assert not rejections, f"receipt was rejected: {rejections}"
    assert len(records) == 1
    record = records[0]
    out = record.outflow

    assert out.counterparty_name == truth.vendor_name
    assert out.amount == truth.total_amount.quantize(Decimal("0.01"))
    assert out.due_date == truth.issue_date + timedelta(days=30)
    assert out.source_type == SourceType.RECEIPT_OCR.value
    assert out.business_id == BUSINESS

    # A.6 classification must land on the category the vendor template used.
    assert record.category_prediction.category == truth.category, (
        f"classified {record.category_prediction.category!r}, "
        f"expected {truth.category!r} from text: {record.fields.raw_text[:120]!r}"
    )

    assert len(duplicates) == 1
    assert duplicates[0].is_exact_duplicate is False


@pytest.mark.ocr
def test_the_same_receipt_twice_is_caught_as_a_duplicate():
    """Two copies of one receipt must be flagged, and only the second one."""
    _engine_or_skip()
    rng = np.random.default_rng(4)
    _truth, image = generate_receipt(2, Difficulty.CLEAN, rng)

    records, duplicates, _ = ingest_receipt_images(
        [image, image], business_id=BUSINESS, engine="easyocr"
    )
    assert len(records) == 2

    flagged = [d for d in duplicates if d.is_exact_duplicate]
    assert len(flagged) == 1, (
        "exactly one of an identical pair should be flagged — flagging both "
        "would leave nothing to keep, flagging neither defeats the screen"
    )
    assert flagged[0].duplicate_of is not None

    # The deterministic id makes exact re-submission idempotent, which is a
    # separate guarantee from the near-duplicate screen.
    assert records[0].outflow.id == records[1].outflow.id


# ---------------------------------------------------------------------------
# Behaviour that must not depend on the engine
# ---------------------------------------------------------------------------

def _fields(**overrides) -> ReceiptFields:
    base = dict(
        vendor_name="Sunrise Properties Pvt Ltd",
        invoice_number="INV-1234-56",
        issue_date=date(2026, 3, 4),
        total_amount=Decimal("118000.00"),
        tax_amount=Decimal("18000.00"),
        raw_text="Sunrise Properties Pvt Ltd Monthly office premises rent TOTAL INR 118000",
        field_confidence={"vendor_name": 0.95, "total_amount": 0.93},
        engine="stub",
        mean_ocr_confidence=0.94,
    )
    base.update(overrides)
    return ReceiptFields(**base)


@pytest.mark.parametrize("missing", ["vendor_name", "total_amount", "issue_date"])
def test_normalize_refuses_rather_than_inventing_a_missing_field(missing):
    """An unreadable required field must abort the record, not default it.

    The failure this prevents is silent: a total defaulted to zero is a
    perfectly valid `Outflow` that reaches the optimizer and changes an
    allocation while looking like data. Refusal is recoverable; a fabricated
    number is not, because nothing downstream knows to doubt it.
    """
    with pytest.raises(NormalizationError, match=missing):
        normalize(_fields(**{missing: None}), business_id=BUSINESS)


def test_lost_decimal_point_is_caught_by_the_tax_cross_check():
    """A total that is 100x wrong must be flagged via total = subtotal + tax.

    This is the specific OCR failure the HARD tier produces most often, and it
    is undetectable from the total alone — 11,800,000 is as parseable as
    118,000. The receipt's own internal redundancy is what catches it.
    """
    ok = normalize(_fields(), business_id=BUSINESS)
    assert not any("tax rate" in r for r in ok.review_reasons)

    corrupted = normalize(
        _fields(total_amount=Decimal("11800000.00")), business_id=BUSINESS
    )
    assert corrupted.needs_review
    assert any("tax rate" in r for r in corrupted.review_reasons), (
        f"expected an implausible-tax-rate flag, got {corrupted.review_reasons}"
    )


def test_missing_invoice_number_is_flagged_not_fatal():
    """The invoice number is recoverable, so it degrades rather than aborts."""
    record = normalize(_fields(invoice_number=None), business_id=BUSINESS)
    assert record.needs_review
    assert any("invoice number" in r for r in record.review_reasons)
    assert record.outflow.source_reference.startswith("UNKNOWN-")


def test_near_duplicate_with_a_different_invoice_number_is_screened():
    """The same expense retyped must still be caught by the A.5 pass."""
    a = normalize(_fields(), business_id=BUSINESS)
    b = normalize(_fields(invoice_number="INV-9999-99"), business_id=BUSINESS)
    others = [
        normalize(
            _fields(
                invoice_number=f"INV-100{i}-00",
                total_amount=Decimal(f"{40000 + i * 9000}.00"),
                tax_amount=Decimal("1000.00"),
                vendor_name=f"Other Vendor {i}",
                issue_date=date(2026, 5, 1) + timedelta(days=i * 11),
            ),
            business_id=BUSINESS,
        )
        for i in range(6)
    ]

    findings = screen_for_duplicates([a, b] + others)
    by_id = {f.record_id: f for f in findings}
    assert by_id[str(b.outflow.id)].is_exact_duplicate, (
        "same vendor, same amount, same date but a different invoice number "
        "must still be caught — that is what the screen is for"
    )
    assert by_id[str(b.outflow.id)].duplicate_of == str(a.outflow.id)


# ---------------------------------------------------------------------------
# Layout post-processing, which the amount fields depend on
# ---------------------------------------------------------------------------

def test_merge_into_rows_reassembles_a_fragmented_amount():
    """`159` `312` `98` on one visual row must read back as 159312.98.

    The detector splits a number wherever a separator blurs away, so this is
    the mechanism behind most HARD-tier total failures. Tested directly on
    synthetic boxes so it does not depend on reproducing a specific OCR error.
    """
    lines = [
        OcrLine("TOTAL", 0.99, (360, 700, 430, 726)),
        OcrLine("INR", 0.92, (510, 700, 550, 726)),
        OcrLine("159", 0.96, (560, 701, 600, 727)),
        OcrLine("312", 1.00, (605, 701, 645, 727)),
        OcrLine("98", 1.00, (650, 701, 680, 727)),
        OcrLine("Payment due within 30 days", 0.90, (44, 800, 400, 826)),
    ]
    merged = merge_into_rows(lines)
    assert len(merged) == 2, f"expected 2 visual rows, got {[m.text for m in merged]}"
    assert merged[0].text == "TOTAL INR 159 312 98"

    fields = extract_fields(OcrResult(engine="stub", lines=merged, elapsed_ms=0.0))
    assert fields.total_amount == Decimal("159312.98")


def test_merge_into_rows_is_robust_to_page_skew():
    """A rotated row must stay one row, not shatter into several.

    At 7 degrees a single row drifts further vertically across the page than a
    row is tall, so grouping on raw y splits it. The observed symptom was a
    vendor name coming back as "Properties Pvt Ltd Sunrise".
    """
    angle = np.deg2rad(7.0)
    lines = []
    for k, word in enumerate(["Sunrise", "Properties", "Pvt", "Ltd"]):
        x = 44 + k * 150
        y = 60 + x * np.tan(angle)
        lines.append(OcrLine(word, 0.95, (x, y, x + 140, y + 30), angle=angle))

    merged = merge_into_rows(lines)
    assert len(merged) == 1, f"skewed row shattered into {len(merged)} rows"
    assert merged[0].text == "Sunrise Properties Pvt Ltd"


def test_invoice_label_is_not_mistaken_for_the_invoice_number():
    """"Invoice No:" must not itself parse as an invoice number.

    Regression: a \\w-based pattern combined with a global glyph-confusion pass
    read the label as INV + "oice" + "No" and produced "INV-0ice-N0".
    """
    lines = [
        OcrLine("Invoice No: INV-3505-74", 0.9, (44, 100, 500, 126)),
        OcrLine("Date: 13/03/2026", 0.9, (44, 140, 500, 166)),
        OcrLine("TOTAL INR 184,151.98", 0.9, (44, 200, 500, 226)),
    ]
    fields = extract_fields(OcrResult(engine="stub", lines=lines, elapsed_ms=0.0))
    assert fields.invoice_number == "INV-3505-74"
    assert fields.issue_date == date(2026, 3, 13)
    assert fields.total_amount == Decimal("184151.98")


def test_gstin_is_not_read_as_a_tax_amount():
    """"GSTIN: 29AABCU9603R1ZM" is a registration number, not a rupee figure."""
    lines = [
        OcrLine("GSTIN: 29AABCU9603R1ZM", 0.8, (44, 100, 500, 126)),
        OcrLine("GST 18% 24,301.98", 0.9, (44, 140, 500, 166)),
    ]
    fields = extract_fields(OcrResult(engine="stub", lines=lines, elapsed_ms=0.0))
    assert fields.tax_amount == Decimal("24301.98")
