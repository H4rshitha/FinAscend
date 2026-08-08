"""Receipt OCR behind one engine-agnostic interface.

The signature is deliberately generic. `OcrEngine.read(image) -> OcrResult` says
nothing about which engine produced the text, so swapping EasyOCR for Google
Vision, Azure Document Intelligence or Textract later is a new class and one
entry in `_ENGINES` — no caller changes. `extract_fields` consumes `OcrResult`
rather than any engine's native output for the same reason: the field
extractor and the accuracy harness are the parts worth keeping, and they must
not be coupled to a vendor's response shape.

WHY EasyOCR
-----------
The requirement was no credentials and no separately-installed binary.

  * **EasyOCR — chosen.** `pip install easyocr` is the whole installation. It
    is a CRAFT text detector plus a CRNN recognizer on PyTorch, runs on CPU,
    and needs no API key. It handles the rotated and low-contrast phone-photo
    cases the HARD tier is built from noticeably better than a threshold-based
    engine, because detection is learned rather than driven by page
    binarization.
  * **pytesseract — the documented fallback.** Rejected as the default because
    `pip install pytesseract` installs only a *wrapper*; the Tesseract binary
    is a separate OS-level install, which is exactly the dependency that was
    ruled out. It is still implemented and selectable, because EasyOCR's
    footprint is real and worth naming: PyTorch pulls roughly 2 GB of wheels
    and the recognition models are downloaded on first use. On a size- or
    network-constrained host, set `FINASCEND_OCR_ENGINE=tesseract`.

The cost of EasyOCR is paid once at import and once at model download; the cost
of a cloud engine would be paid per request and per credential. For a system
whose other components are all reproducible offline, that trade favours the
local engine.

MODEL DOWNLOAD
--------------
EasyOCR fetches its detector and recognizer weights on first construction. That
is a network dependency, and it fails loudly here rather than silently
returning empty text — an OCR engine that reads nothing and reports success
would make the accuracy harness report zeros and look like an extraction bug.
"""

from __future__ import annotations

import io
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union

import numpy as np

ImageInput = Union[str, Path, bytes, np.ndarray, "object"]


# ---------------------------------------------------------------------------
# Engine-agnostic result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OcrLine:
    """One detected text region."""

    text: str
    confidence: float
    # (x_min, y_min, x_max, y_max) in pixels. Kept because reading order and
    # column alignment carry real signal for receipts — the total is the number
    # to the RIGHT of the word TOTAL, not merely the next number in the stream.
    bbox: tuple[float, float, float, float]
    # Baseline angle of this region in radians, from the detector's quad. Zero
    # for engines that only report axis-aligned boxes.
    angle: float = 0.0

    @property
    def y_center(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


def merge_into_rows(lines: Sequence[OcrLine], tolerance: float = 0.6) -> list[OcrLine]:
    """Group detected regions into visual rows, then order them for reading.

    WHY THIS IS NECESSARY, NOT COSMETIC
    -----------------------------------
    A text detector emits *regions*, and it splits a region wherever it sees a
    gap — including the gaps a thousands separator and a decimal point create.
    On a degraded image the grand total `INR 159,312.98` comes back as four
    separate regions: `TOTAL`, `INR`, `159`, `312`, `98`. Parsed region by
    region, that total is unrecoverable; parsed as the visual row it actually
    is, it reads back exactly. The same grouping is what puts a label and its
    value together when they sit in two columns.

    ROTATION
    --------
    Grouping on raw `y` fails as soon as the page is skewed: at 7 degrees a
    single row drifts further vertically across the page width than a row is
    tall, so one row shatters into several and reading order scrambles (the
    observed symptom was a vendor name coming back as "Properties Pvt Ltd
    Sunrise"). The page angle is therefore estimated as the median of the
    detector's own per-region baseline angles, and rows are grouped on the
    **deskewed** coordinate

        y' = y * cos(theta) - x * sin(theta)

    which is constant along a row whatever the skew. Deskewing the coordinates
    is far cheaper than deskewing and re-OCRing the image, and it uses
    information the detector already produced.
    """
    if not lines:
        return []

    theta = float(np.median([l.angle for l in lines]))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    med_h = float(np.median([max(l.height, 1.0) for l in lines]))

    def row_coord(l: OcrLine) -> float:
        x = (l.bbox[0] + l.bbox[2]) / 2.0
        return l.y_center * cos_t - x * sin_t

    ordered = sorted(lines, key=row_coord)
    rows: list[list[OcrLine]] = [[ordered[0]]]
    for line in ordered[1:]:
        if abs(row_coord(line) - row_coord(rows[-1][-1])) <= tolerance * med_h:
            rows[-1].append(line)
        else:
            rows.append([line])

    merged: list[OcrLine] = []
    for row in rows:
        row.sort(key=lambda l: l.bbox[0])
        merged.append(
            OcrLine(
                text=" ".join(l.text for l in row),
                # The weakest region governs: a row is only as trustworthy as
                # its least confident piece, and averaging would let one
                # crisp word hide a garbled amount beside it.
                confidence=float(min(l.confidence for l in row)),
                bbox=(
                    min(l.bbox[0] for l in row), min(l.bbox[1] for l in row),
                    max(l.bbox[2] for l in row), max(l.bbox[3] for l in row),
                ),
                angle=theta,
            )
        )
    return merged


@dataclass(frozen=True)
class OcrResult:
    """Everything an engine returns, in a shape no engine is special in."""

    engine: str
    lines: list[OcrLine]
    elapsed_ms: float
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        return float(np.mean([l.confidence for l in self.lines])) if self.lines else 0.0


class OcrEngine(Protocol):
    """The whole contract. A cloud engine implements exactly this."""

    name: str

    def read(self, image: ImageInput) -> OcrResult:
        """Extract text regions from one image."""
        ...


def _to_array(image: ImageInput) -> np.ndarray:
    """Normalize every accepted input form to an RGB uint8 array."""
    from PIL import Image

    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, (str, Path)):
        return np.asarray(Image.open(image).convert("RGB"))
    if isinstance(image, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    raise TypeError(f"unsupported image input: {type(image)!r}")


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

class EasyOcrEngine:
    """EasyOCR (CRAFT detector + CRNN recognizer) on CPU.

    The `Reader` is constructed lazily and cached on the instance: building it
    loads ~100 MB of weights, which must not happen at import time or every
    test collection pays for it.
    """

    name = "easyocr"

    def __init__(self, languages: Sequence[str] = ("en",), gpu: bool = False) -> None:
        self.languages = list(languages)
        self.gpu = gpu
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr  # imported here so the module loads without torch

            self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        return self._reader

    def read(self, image: ImageInput) -> OcrResult:
        arr = _to_array(image)
        t0 = time.perf_counter()
        raw = self._get_reader().readtext(arr, detail=1, paragraph=False)
        elapsed = (time.perf_counter() - t0) * 1000.0

        lines = []
        for bbox, text, conf in raw:
            pts = np.asarray(bbox, dtype=float)
            # EasyOCR returns a quadrilateral in clockwise order from the
            # top-left, so pts[1] - pts[0] is the baseline direction and gives
            # this region's skew for free.
            top_edge = pts[1] - pts[0]
            angle = float(np.arctan2(top_edge[1], top_edge[0]))
            lines.append(
                OcrLine(
                    text=str(text),
                    confidence=float(conf),
                    bbox=(
                        float(pts[:, 0].min()), float(pts[:, 1].min()),
                        float(pts[:, 0].max()), float(pts[:, 1].max()),
                    ),
                    angle=angle,
                )
            )
        return OcrResult(
            engine=self.name, lines=merge_into_rows(lines), elapsed_ms=elapsed
        )


class TesseractOcrEngine:
    """pytesseract fallback. Requires the Tesseract BINARY, not just the wheel.

    Kept because EasyOCR's PyTorch footprint is a legitimate reason to choose
    otherwise, and because having a second engine behind the same interface is
    what demonstrates the interface is actually engine-agnostic.
    """

    name = "tesseract"

    def __init__(self, config: str = "--psm 6") -> None:
        self.config = config

    def read(self, image: ImageInput) -> OcrResult:
        import pytesseract
        from PIL import Image

        arr = _to_array(image)
        t0 = time.perf_counter()
        data = pytesseract.image_to_data(
            Image.fromarray(arr), config=self.config,
            output_type=pytesseract.Output.DICT,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0

        # Tesseract emits per-word boxes; regroup into lines so the result is
        # shaped like every other engine's.
        grouped: dict[tuple, list[int]] = {}
        for i, txt in enumerate(data["text"]):
            if not str(txt).strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            grouped.setdefault(key, []).append(i)

        lines = []
        for idxs in grouped.values():
            words = [str(data["text"][i]) for i in idxs]
            confs = [float(data["conf"][i]) for i in idxs if float(data["conf"][i]) >= 0]
            xs = [data["left"][i] for i in idxs]
            ys = [data["top"][i] for i in idxs]
            x2 = [data["left"][i] + data["width"][i] for i in idxs]
            y2 = [data["top"][i] + data["height"][i] for i in idxs]
            lines.append(
                OcrLine(
                    text=" ".join(words),
                    confidence=float(np.mean(confs) / 100.0) if confs else 0.0,
                    bbox=(float(min(xs)), float(min(ys)), float(max(x2)), float(max(y2))),
                )
            )
        return OcrResult(
            engine=self.name, lines=merge_into_rows(lines), elapsed_ms=elapsed
        )


_ENGINES = {"easyocr": EasyOcrEngine, "tesseract": TesseractOcrEngine}
_ENGINE_CACHE: dict[str, OcrEngine] = {}


def resolve_engine(name: Optional[str] = None) -> OcrEngine:
    """Return the configured engine, cached so weights load once per process.

    Resolution order: explicit argument, then `FINASCEND_OCR_ENGINE`, then
    EasyOCR. A cloud engine slots in by registering in `_ENGINES`.
    """
    key = (name or os.environ.get("FINASCEND_OCR_ENGINE") or "easyocr").lower()
    if key not in _ENGINES:
        raise ValueError(f"unknown OCR engine {key!r}; have {sorted(_ENGINES)}")
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = _ENGINES[key]()
    return _ENGINE_CACHE[key]


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReceiptFields:
    """Structured fields recovered from OCR text, each with its own confidence.

    Per-field confidence rather than one document-level score: OCR routinely
    reads a vendor name perfectly and mangles the total, and a single blended
    number cannot express that. The normalizer downstream needs to know which
    fields to trust.
    """

    vendor_name: Optional[str]
    invoice_number: Optional[str]
    issue_date: Optional[date]
    total_amount: Optional[Decimal]
    tax_amount: Optional[Decimal]
    raw_text: str
    field_confidence: dict[str, float]
    engine: str
    mean_ocr_confidence: float


# OCR confuses these glyph pairs constantly, and a receipt's digits are exactly
# where it matters. Applied only inside numeric candidates — applying it to the
# vendor name would corrupt legitimate letters.
_DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0",
                              "l": "1", "I": "1", "|": "1",
                              "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2"})

_BOILERPLATE = {
    "tax invoice", "receipt", "invoice", "tax invoice / receipt", "gstin",
    "description", "amount", "date", "subtotal", "total", "bill",
}


def _fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _contains_label(text: str, label: str, threshold: float = 0.78) -> bool:
    """Fuzzy word-level label match, because OCR corrupts labels too.

    Exact matching would fail on 'T0TAL' and 'lnvoice', which are the readings
    a degraded image actually produces — the whole reason regex-only extraction
    was rejected in A.6.
    """
    low = text.lower()
    if label in low:
        return True
    return any(_fuzzy(w, label) >= threshold for w in re.findall(r"[a-z0-9]+", low))


_RUN_RE = re.compile(r"\d[\d.,\s]*\d")
MIN_AMOUNT_DIGITS = 3


def _run_to_decimal(run: str) -> Optional[Decimal]:
    """Interpret one numeric run, reassembling groups the detector split.

    A receipt amount arrives as digit groups separated by thousands marks,
    decimal points, or — once the detector has cut the region at those marks —
    by nothing but whitespace. `159,312.98`, `159 312 98` and `184 ,151.98` are
    all the same number and all three occur in the corpus.

    Two rules, in order:

    1. If an explicit decimal point survives, split on the LAST one: everything
       before is the integer part with its separators stripped, everything
       after is the fraction. Last rather than first, because a mis-read
       thousands separator can add a spurious earlier point.
    2. Otherwise, use group *shape*. Currency grouping is regular: after the
       leading group every full group is exactly 3 digits, so a trailing group
       of exactly 2 can only be a fractional part. `159 | 312 | 98` is
       159312.98; `135 | 011` is 135011.

    Rule 2 is a genuine inference from layout regularity rather than a guess,
    but it is not infallible — a two-digit trailing group that really is a
    thousands group (impossible in standard grouping, possible under OCR
    corruption) would be misread by a factor of 100. That failure mode is
    stated rather than hidden, and it is why HARD-tier totals are reported
    separately instead of pooled into one accuracy number.
    """
    groups = re.findall(r"\d+", run)
    if not groups or sum(len(g) for g in groups) < MIN_AMOUNT_DIGITS:
        return None

    try:
        if "." in run:
            head, _, tail = run.rpartition(".")
            whole = "".join(re.findall(r"\d+", head))
            frac = "".join(re.findall(r"\d+", tail))
            if not whole:
                return None
            return Decimal(f"{whole}.{frac}" if frac else whole)

        if (
            len(groups) > 1
            and len(groups[-1]) == 2
            and all(len(g) == 3 for g in groups[1:-1])
            and len(groups[0]) <= 3
        ):
            return Decimal("".join(groups[:-1]) + "." + groups[-1])
        return Decimal("".join(groups))
    except InvalidOperation:
        return None


def _parse_amount(text: str) -> Optional[Decimal]:
    """Pull a currency amount out of a noisy line.

    Two passes: the strict one first, then a retry with glyph confusions
    repaired (`l35,Oll` -> `135,011`). Candidates are ranked by DIGIT COUNT
    rather than by magnitude, because the failure to avoid is a truncated read
    ("135" out of "135,011") and the longest well-formed number on the line is
    the amount. Ranking by magnitude would instead reward the glyph-repair pass
    for inventing extra digits.
    """
    candidates: list[tuple[int, Decimal]] = []
    for variant in (text, text.translate(_DIGIT_FIXES)):
        for m in _RUN_RE.finditer(variant):
            value = _run_to_decimal(m.group(0))
            if value is not None and value > 0:
                candidates.append((len(re.findall(r"\d", m.group(0))), value))
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c[0], c[1]))[1]


def _parse_date(text: str) -> Optional[date]:
    """Parse dd/mm/yyyy and the separators OCR turns it into."""
    t = text.replace(" ", "")
    # OCR reads '/' as '1', '7', 'I' or '|'; accept any non-digit separator and
    # also the run-together form.
    for pattern, fmt in (
        (r"(\d{2})[^\d](\d{2})[^\d](\d{4})", "%d/%m/%Y"),
        (r"(\d{2})(\d{2})(\d{4})", "%d/%m/%Y"),
        (r"(\d{4})[^\d](\d{2})[^\d](\d{2})", "%Y/%m/%d"),
    ):
        m = re.search(pattern, t)
        if not m:
            continue
        try:
            return datetime.strptime("/".join(m.groups()), fmt).date()
        except ValueError:
            continue
    return None


_GSTIN_RE = re.compile(r"\d{2}[A-Z]{5}\d{4}[A-Z]", re.I)
# The prefix is matched tolerantly ([I1l|]NV covers "1NV", "lNV", "InV") and
# the groups require real digits. An earlier version ran the whole line through
# the glyph-confusion table first, which turned the legitimate "I" of "INV"
# into "1" so the literal prefix never matched; and it accepted \w for the
# groups, so the label "Invoice No:" itself parsed as INV + "oice" + "No" and
# yielded the invoice number "INV-0ice-N0".
_INVOICE_RE = re.compile(r"[I1l|]NV[\s\-–_]?(\d{3,4})[\s\-–_]?(\d{2})(?!\d)", re.I)


def _value_after(
    lines: Sequence[OcrLine], idx: int, parser, lookahead: int = 3
) -> tuple[Optional[object], float]:
    """Read a labelled value from the label's line, or from the lines after it.

    Detectors emit a text *region* per visual block, and on a two-column
    receipt "Date:" and "24/05/2026" are two regions, not one line. Parsing
    only the label's own line therefore misses every field on a layout where
    the value sits in its own box — which is most of them. Searching forward in
    reading order handles both layouts with one rule, and the small lookahead
    bounds how far a label can reach so a missing value cannot silently adopt
    an unrelated number further down the page.

    Returns:
        (value or None, confidence of the line the value came from)
    """
    for j in range(idx, min(idx + lookahead + 1, len(lines))):
        value = parser(lines[j].text)
        if value is not None:
            return value, lines[j].confidence
    return None, 0.0


def extract_fields(result: OcrResult) -> ReceiptFields:
    """Recover the four scored fields from an `OcrResult`.

    Label-anchored rather than position-anchored: each field is found by
    locating its *label* fuzzily and reading the value beside or below it. A
    fixed template ("the total is on line 17") would score well on this
    generator and fail on the first real receipt with a different layout, which
    would make the accuracy numbers meaningless as evidence.

    Every field may come back `None`. A missing field is a correct answer when
    OCR genuinely failed to read it, and is scored as a miss rather than
    guessed at — a fabricated total is far worse than an absent one, because it
    enters the ledger silently and nothing downstream can tell it was invented.
    """
    lines = result.lines
    conf: dict[str, float] = {}

    # --- vendor: the first substantial non-boilerplate line at the top ---
    vendor = None
    for l in lines[:5]:
        s = l.text.strip()
        if len(s) < 6:
            continue
        if any(_fuzzy(s, b) > 0.75 for b in _BOILERPLATE):
            continue
        if re.fullmatch(r"[\d\W]+", s):
            continue
        vendor = s
        conf["vendor_name"] = l.confidence
        break

    # --- invoice number ---
    # The pattern requires DIGITS in both groups. An earlier version accepted
    # \w and matched the label itself: "Invoice No:" parsed as INV + "oice" +
    # "No" and produced the invoice number "INV-0ice-N0".
    def _parse_invoice(text: str) -> Optional[str]:
        for variant in (text, text.translate(_DIGIT_FIXES)):
            m = _INVOICE_RE.search(variant)
            if m:
                return f"INV-{m.group(1)}-{m.group(2)}"
        return None

    invoice_no, c = None, 0.0
    for i, l in enumerate(lines):
        if _contains_label(l.text, "invoice") or "inv" in l.text.lower():
            invoice_no, c = _value_after(lines, i, _parse_invoice)
            if invoice_no:
                conf["invoice_number"] = c
                break
    if invoice_no is None:
        invoice_no = _parse_invoice(result.text)
        if invoice_no:
            conf["invoice_number"] = result.mean_confidence * 0.8

    # --- date ---
    issue_date, c = None, 0.0
    for i, l in enumerate(lines):
        if _contains_label(l.text, "date") or _contains_label(l.text, "dated"):
            issue_date, c = _value_after(lines, i, _parse_date)
            if issue_date:
                conf["issue_date"] = c
                break
    if issue_date is None:
        for l in lines:
            issue_date = _parse_date(l.text)
            if issue_date:
                conf["issue_date"] = l.confidence * 0.8
                break

    # --- total: prefer the TOTAL block, and never the SUBTOTAL decoy ---
    total, best_rank = None, -1.0
    for i, l in enumerate(lines):
        low = l.text.lower()
        if not _contains_label(l.text, "total"):
            continue
        rank = 1.0
        if "sub" in low:
            rank -= 0.6          # subtotal is a decoy, not the answer
        if any(k in low for k in ("inr", "rs", "₹")):
            rank += 0.5          # the grand total carries the currency mark
        # A currency mark may also sit on the line BETWEEN the label and the
        # figure ("TOTAL" / "INR" / "159,312.98"), so credit that too.
        if i + 1 < len(lines) and any(
            k in lines[i + 1].text.lower() for k in ("inr", "rs", "₹")
        ):
            rank += 0.5
        amount, c = _value_after(lines, i, _parse_amount)
        if amount is None or rank <= best_rank:
            continue
        best_rank, total = rank, amount
        conf["total_amount"] = c

    # --- tax ---
    tax = None
    for i, l in enumerate(lines):
        # "GSTIN" fuzzy-matches "gst" but labels a registration number, not an
        # amount. Skipping it explicitly stops the 15-character alphanumeric
        # GSTIN from being read as a rupee figure.
        if _fuzzy(l.text.strip().rstrip(":"), "gstin") > 0.8:
            continue
        if not (_contains_label(l.text, "gst") or _contains_label(l.text, "tax")):
            continue
        candidate, c = _value_after(
            lines, i, lambda t: None if _GSTIN_RE.search(t) else _parse_amount(t)
        )
        if candidate is not None:
            tax = candidate
            conf["tax_amount"] = c
            break

    return ReceiptFields(
        vendor_name=vendor,
        invoice_number=invoice_no,
        issue_date=issue_date,
        total_amount=total,
        tax_amount=tax,
        raw_text=result.text,
        field_confidence=conf,
        engine=result.engine,
        mean_ocr_confidence=result.mean_confidence,
    )


def read_receipt(image: ImageInput, engine: Optional[str] = None) -> ReceiptFields:
    """Convenience: OCR an image and extract its fields in one call."""
    return extract_fields(resolve_engine(engine).read(image))
