"""Receipt-OCR endpoints backing the frontend's ingestion demo.

Every response here is computed on the request. Samples are rendered by the
generator, OCR runs live, and the resulting record is the same `Outflow` the
integration tests assert against — nothing is canned, and the ground truth is
returned alongside the extraction so the page can show whether each field was
actually right rather than merely present.
"""

from __future__ import annotations

import io
from typing import Any, Optional
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.core.security import TokenPayload, current_user
from app.services.ingestion.normalizer import (
    NormalizationError,
    normalize,
    screen_for_duplicates,
)
from app.services.ingestion.ocr_service import extract_fields, resolve_engine
from app.services.ingestion.receipt_generator import (
    Difficulty,
    generate_receipt,
    make_truth,
)

router = APIRouter()

DEMO_BUSINESS = UUID("00000000-0000-0000-0000-0000000000de")
SAMPLE_SEED = 7
N_SAMPLES_PER_TIER = 6


def _sample(tier: Difficulty, index: int):
    """Regenerate one sample deterministically.

    The generator advances a single RNG per tier, so a sample is only
    reproducible by replaying the draws up to its index. Doing that here keeps
    the endpoint stateless — no corpus is held in memory or on disk between
    requests, and the image the user sees is byte-identical to the one scored
    in `accuracy_report`.
    """
    rng = np.random.default_rng(SAMPLE_SEED)
    truth = image = None
    for i in range(index + 1):
        truth, image = generate_receipt(i, tier, rng)
    return truth, image


@router.get("/ingestion/receipts/samples", tags=["ingestion"])
def list_samples(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    """List the selectable demo receipts, with their ground truth."""
    out = []
    for tier in (Difficulty.CLEAN, Difficulty.MODERATE, Difficulty.HARD):
        for i in range(N_SAMPLES_PER_TIER):
            truth = _sample(tier, i)[0]
            out.append(
                {
                    "id": f"{tier.value}-{i}",
                    "difficulty": tier.value,
                    "index": i,
                    "truth": {
                        "vendor_name": truth.vendor_name,
                        "invoice_number": truth.invoice_number,
                        "issue_date": truth.issue_date.isoformat(),
                        "total_amount": float(truth.total_amount),
                        "tax_amount": float(truth.tax_amount),
                        "category": truth.category,
                    },
                }
            )
    return {
        "samples": out,
        "note": (
            "Ground truth is returned so the UI can mark each extracted field "
            "right or wrong. It is never used by the extractor."
        ),
    }


@router.get("/ingestion/receipts/{sample_id}/image", tags=["ingestion"])
def sample_image(sample_id: str, user: TokenPayload = Depends(current_user)) -> Response:
    """Render one sample receipt as a PNG."""
    tier, _, idx = sample_id.rpartition("-")
    try:
        _truth, image = _sample(Difficulty(tier), int(idx))
    except (ValueError, KeyError):
        raise HTTPException(
            status_code=404,
            detail={"error_code": "not_found", "message": f"no sample {sample_id!r}",
                    "details": {}},
        )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


def _pipeline_response(image, truth=None) -> dict[str, Any]:
    """Run OCR -> A.6 classify -> normalize -> A.5 dedup and describe each stage."""
    engine = resolve_engine()
    result = engine.read(image)
    fields = extract_fields(result)

    stages: dict[str, Any] = {
        "ocr": {
            "engine": result.engine,
            "n_regions": len(result.lines),
            "mean_confidence": round(result.mean_confidence, 4),
            "elapsed_ms": round(result.elapsed_ms, 1),
            "text": result.text,
            "lines": [
                {"text": l.text, "confidence": round(l.confidence, 4), "bbox": l.bbox}
                for l in result.lines
            ],
        },
        "extraction": {
            "vendor_name": fields.vendor_name,
            "invoice_number": fields.invoice_number,
            "issue_date": fields.issue_date.isoformat() if fields.issue_date else None,
            "total_amount": float(fields.total_amount) if fields.total_amount else None,
            "tax_amount": float(fields.tax_amount) if fields.tax_amount else None,
            "field_confidence": {k: round(v, 4) for k, v in fields.field_confidence.items()},
        },
    }

    try:
        record = normalize(fields, business_id=DEMO_BUSINESS)
    except NormalizationError as exc:
        stages["classification"] = None
        stages["record"] = None
        stages["rejected"] = {
            "reason": str(exc),
            "why_this_is_correct": (
                "A required field was unreadable. The pipeline refuses to build "
                "a record rather than defaulting the amount — a placeholder "
                "reaches the optimizer looking like data."
            ),
        }
        stages["duplicate_screen"] = None
        return stages

    p = record.category_prediction
    stages["classification"] = {
        "category": p.category,
        "confidence": round(p.confidence, 4),
        "runner_up": p.runner_up,
        "runner_up_score": round(p.runner_up_score, 4),
        "margin": round(p.confidence - p.runner_up_score, 4),
        "is_uncertain": p.is_uncertain,
        "method": "char_wb 3-5 n-gram TF-IDF, cosine similarity to the A.6 reference set",
    }
    o = record.outflow
    stages["record"] = {
        "id": str(o.id),
        "counterparty_name": o.counterparty_name,
        "amount": float(o.amount),
        "currency": o.currency,
        "due_date": o.due_date.isoformat(),
        "category": o.category,
        "source_type": o.source_type,
        "source_reference": o.source_reference,
        "needs_review": record.needs_review,
        "review_reasons": record.review_reasons,
        "extraction_confidence": round(record.extraction_confidence, 4),
    }
    finding = screen_for_duplicates([record])[0]
    stages["duplicate_screen"] = {
        "is_exact_duplicate": finding.is_exact_duplicate,
        "dbscan_label": finding.dbscan_label,
        "robust_z": round(finding.robust_z, 4),
        "flagged_by_isolation_forest": finding.flagged_by_isolation_forest,
        "reason": finding.reason,
        "caveat": (
            "Screened against this request only. A real screen runs against the "
            "existing ledger; a single-record DBSCAN can only ever report noise."
        ),
    }
    stages["rejected"] = None

    if truth is not None:
        stages["truth"] = {
            "vendor_name": truth.vendor_name,
            "invoice_number": truth.invoice_number,
            "issue_date": truth.issue_date.isoformat(),
            "total_amount": float(truth.total_amount),
            "category": truth.category,
        }
        stages["correct"] = {
            "vendor_name": fields.vendor_name == truth.vendor_name,
            "invoice_number": fields.invoice_number == truth.invoice_number,
            "issue_date": fields.issue_date == truth.issue_date,
            "total_amount": (
                fields.total_amount is not None
                and float(fields.total_amount) == float(truth.total_amount)
            ),
            "category": p.category == truth.category,
        }
    return stages


@router.post("/ingestion/receipts/sample/{sample_id}/process", tags=["ingestion"])
def process_sample(
    sample_id: str, user: TokenPayload = Depends(current_user)
) -> dict[str, Any]:
    """Run the full chain on a generated sample and score it against truth."""
    tier, _, idx = sample_id.rpartition("-")
    try:
        truth, image = _sample(Difficulty(tier), int(idx))
    except (ValueError, KeyError):
        raise HTTPException(
            status_code=404,
            detail={"error_code": "not_found", "message": f"no sample {sample_id!r}",
                    "details": {}},
        )
    return _pipeline_response(image, truth)


@router.post("/ingestion/receipts/upload", tags=["ingestion"])
async def process_upload(
    file: UploadFile = File(...), user: TokenPayload = Depends(current_user)
) -> dict[str, Any]:
    """Run the full chain on an uploaded image.

    No ground truth is available for an upload, so the response omits the
    correctness block rather than inventing one — the page shows what was
    extracted and how confident each field was, and nothing more.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "empty_upload", "message": "no image data",
                    "details": {}},
        )
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "unreadable_image",
                "message": "could not decode the uploaded file as an image",
                "details": {"exception": type(exc).__name__},
            },
        )
    return _pipeline_response(image)


# ===========================================================================
# Bank statements — the second ingestion channel
# ===========================================================================
#
# Documents come in through OCR above; account data comes in here. Both
# converge on the same domain records, and the statement path carries a check
# the OCR path cannot: a statement asserts its own running balance, so a
# mis-parse is detectable from the file alone.

from datetime import date, timedelta  # noqa: E402

from app.services.ingestion.bank_api_client import (  # noqa: E402
    HttpStatementProvider,
    LocalReferenceProvider,
    ProviderError,
    RateCapExceeded,
    SyncEngine,
)
from app.services.ingestion.statement_generator import (  # noqa: E402
    Dialect,
    generate_statement,
)
from app.services.ingestion.statement_parser import (  # noqa: E402
    StatementParseError,
    ingest_statement,
    parse_statement,
)

# Module-level so the rate cap and idempotency cache actually persist between
# requests. A per-request engine would reset the counter every call, which is
# the same as having no cap at all.
SYNC_ENGINE = SyncEngine()

STATEMENT_SAMPLE_SEED = 5
STATEMENT_SAMPLE_ROWS = 45


@router.get("/ingestion/statements/dialects", tags=["ingestion"])
def list_statement_dialects(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    """The bank layouts the parser is exercised against.

    Listed because "handles any bank" is a claim, and this is the set on which
    it is actually tested. Every one of these produces an identical ledger from
    the same seed, which is what the cross-dialect test asserts.
    """
    return {
        "dialects": [
            {
                "id": d.value,
                "description": desc,
            }
            for d, desc in (
                (Dialect.SIMPLE_DEBIT_CREDIT,
                 "two amount columns, dd/mm/yyyy, header on row 1"),
                (Dialect.SIGNED_AMOUNT,
                 "one signed amount column, dd-Mon-yyyy dates"),
                (Dialect.AMOUNT_WITH_DR_CR_FLAG,
                 "one positive amount column plus a separate Dr/Cr indicator"),
                (Dialect.INDIAN_BANK_PREAMBLE,
                 "account preamble above the header, ISO dates, lakh grouping, "
                 "trailing 'Cr' on the balance"),
                (Dialect.US_NO_BALANCE,
                 "mm/dd/yyyy, parenthesised negatives, NO balance column — so the "
                 "reconciliation check is unavailable and says so"),
                (Dialect.AMBIGUOUS_MMDD,
                 "every day <= 12, so dd/mm and mm/dd both parse; resolved by the "
                 "column's chronological order"),
            )
        ],
        "note": (
            "The column mapping is chosen by whichever assignment satisfies the "
            "running-balance identity, not by header names — which is why a "
            "statement with swapped Debit/Credit headers still parses correctly."
        ),
    }


@router.get("/ingestion/statements/sample", tags=["ingestion"])
def statement_sample(
    dialect: Dialect = Query(Dialect.SIMPLE_DEBIT_CREDIT),
    user: TokenPayload = Depends(current_user),
) -> dict[str, Any]:
    """A generated statement in the requested dialect, with its ground truth."""
    truth, text = generate_statement(
        seed=STATEMENT_SAMPLE_SEED, dialect=dialect, n_rows=STATEMENT_SAMPLE_ROWS
    )
    return {
        "dialect": dialect.value,
        "csv": text,
        "truth": {
            "n_rows": len(truth.rows),
            "total_debits": float(truth.total_debits),
            "total_credits": float(truth.total_credits),
            "opening_balance": float(truth.opening_balance),
            "closing_balance": float(truth.closing_balance),
            "date_format": truth.date_format,
        },
        "note": "Truth is returned so the UI can score the parse. The parser never sees it.",
    }


def _statement_response(text: str, account_reference: str) -> dict[str, Any]:
    """Parse, then describe every stage — including a refusal."""
    try:
        result, inflows, outflows = ingest_statement(
            text, business_id=DEMO_BUSINESS, account_reference=account_reference
        )
    except StatementParseError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "unparseable_statement",
                "message": str(exc),
                "details": {},
            },
        )

    return {
        "mapping": {
            "convention": result.convention,
            "date_format": result.date_format,
            "columns": [a.model_dump() for a in result.column_assignments],
        },
        "reconciliation": result.reconciliation.model_dump(),
        "totals": {
            "n_rows": len(result.rows),
            "total_debits": float(result.total_debits),
            "total_credits": float(result.total_credits),
            "opening_balance": (
                float(result.opening_balance) if result.opening_balance is not None else None
            ),
            "closing_balance": (
                float(result.closing_balance) if result.closing_balance is not None else None
            ),
        },
        "rows": [r.model_dump(mode="json") for r in result.rows[:200]],
        "records": {
            "n_inflows": len(inflows),
            "n_outflows": len(outflows),
            "sample_inflows": [i.model_dump(mode="json") for i in inflows[:5]],
            "sample_outflows": [o.model_dump(mode="json") for o in outflows[:5]],
        },
        "rejected": result.rejected,
        "rejection_reason": result.rejection_reason,
        "warnings": result.warnings,
    }


@router.post("/ingestion/statements/parse", tags=["ingestion"])
async def parse_statement_upload(
    file: UploadFile = File(...),
    account_reference: str = Query("UPLOAD"),
    user: TokenPayload = Depends(current_user),
) -> dict[str, Any]:
    """Parse an uploaded bank statement CSV into domain records.

    A statement that fails its own running-balance identity comes back with
    `rejected = true` and no records, rather than as a ledger the optimizer
    would treat as fact.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "empty_upload", "message": "no file data",
                    "details": {}},
        )
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return _statement_response(text, account_reference)


@router.post("/ingestion/statements/sync", tags=["ingestion"])
def sync_statements(
    dialect: Dialect = Query(Dialect.SIMPLE_DEBIT_CREDIT),
    days: int = Query(60, ge=1, le=365),
    page_size: int = Query(25, ge=5, le=200),
    n_rows: int = Query(60, ge=1, le=500),
    base_url: Optional[str] = Query(
        None,
        description="Point at a real statement API. Omitted -> the in-process "
                    "reference provider, which is labelled as such in the response.",
    ),
    api_key: Optional[str] = Query(None),
    user: TokenPayload = Depends(current_user),
) -> dict[str, Any]:
    """Pull a statement window over the API channel, then parse it.

    With no `base_url` this runs against `LocalReferenceProvider` — real
    pagination, real rate-cap accounting, real idempotency, over data this repo
    generates. The response's `provider_kind` says `local_reference` so it can
    never be read as evidence of a live bank integration.

    Supply `base_url` and the same code path speaks real HTTP to it.
    """
    until = date.today()
    since = until - timedelta(days=days)

    provider = (
        HttpStatementProvider(base_url, api_key=api_key)
        if base_url
        else LocalReferenceProvider(
            seed=STATEMENT_SAMPLE_SEED, dialect=dialect,
            n_rows=n_rows, page_size=page_size,
        )
    )

    try:
        result, inflows, outflows = SYNC_ENGINE.sync(
            provider, business_id=DEMO_BUSINESS, account_reference="DEMO-ACCOUNT",
            since=since, until=until,
        )
    except RateCapExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error_code": "rate_cap_exceeded",
                "message": str(exc),
                "details": SYNC_ENGINE.rate_cap.status(
                    f"{provider.name}:{DEMO_BUSINESS}"
                ).model_dump(mode="json"),
            },
        )
    except ProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "provider_unavailable",
                "message": str(exc),
                "details": {"provider": provider.name},
            },
        )

    payload = result.model_dump(mode="json")
    payload["records"] = {
        "n_inflows": len(inflows),
        "n_outflows": len(outflows),
        "sample_inflows": [i.model_dump(mode="json") for i in inflows[:5]],
        "sample_outflows": [o.model_dump(mode="json") for o in outflows[:5]],
    }
    return payload


@router.get("/ingestion/providers", tags=["ingestion"])
def list_providers(user: TokenPayload = Depends(current_user)) -> dict[str, Any]:
    """What can be pulled from, and what is honestly behind each one."""
    return {
        "providers": [
            {
                "name": "local_reference",
                "kind": "local_reference",
                "available": True,
                "description": (
                    "Serves statements from this repository's own generator, "
                    "in-process. Exercises pagination, rate capping and "
                    "idempotency for real. It is NOT a bank."
                ),
            },
            {
                "name": "http_open_banking",
                "kind": "http_open_banking",
                "available": True,
                "description": (
                    "Real HTTP client — cursor pagination, idempotency keys, "
                    "rolling-window rate cap, exponential backoff with full "
                    "jitter honouring Retry-After. Pass base_url to use it. "
                    "Tested against a live local server over a real socket."
                ),
            },
            {
                "name": "decentro / plaid",
                "kind": None,
                "available": False,
                "description": (
                    "Not implemented. No credentials exist for this project, and "
                    "a connector that cannot be executed would be worse than "
                    "absent. HttpStatementProvider is the integration point."
                ),
            },
        ],
        "rate_cap_default": {
            "calls_allowed": SYNC_ENGINE.rate_cap.calls_allowed,
            "window_days": SYNC_ENGINE.rate_cap.window_days,
        },
    }
