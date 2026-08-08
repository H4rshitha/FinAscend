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
