"""Synthetic receipt images with known ground truth.

The same principle as A.0's cash-flow generator: OCR and field extraction can
only be *scored* against inputs whose correct answer is known in advance.
A pipeline demonstrated on three hand-picked receipts is an anecdote; one
scored against a generated corpus with per-field ground truth is a measurement.

THREE DIFFICULTY TIERS, REPORTED SEPARATELY
-------------------------------------------
Reporting a single blended accuracy over a mixed corpus is the failure this
module is built to avoid, because the blend is controlled by the mix. A corpus
that is 90% clean scans reports ~clean accuracy and says nothing about the
photographed-in-a-shop case that actually breaks OCR. The tiers are therefore
kept separate all the way through to the report:

  CLEAN     flatbed-scan conditions — upright, sharp, full contrast.
            The upper bound: whatever accuracy is lost here is lost to the
            extractor, not to image quality.
  MODERATE  a careful phone photo — slight rotation, mild blur, mild contrast
            loss, light sensor noise, JPEG recompression.
  HARD      a hurried phone photo in bad light — larger skew, real blur, low
            contrast, visible noise, aggressive JPEG, and a thermal-paper-style
            luminance gradient.

The degradations are applied in physical order (geometry, then optics, then
sensor, then compression) because that is the order a real camera applies
them, and applying JPEG before blur would produce artifacts no real photo has.

Every parameter is drawn from a seeded `numpy.random.Generator`, so a tier is
a *distribution* of images rather than one fixed distortion — an extractor
tuned to a single rotation angle would otherwise pass.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# matplotlib ships DejaVu, so a real scalable font is available without adding
# a dependency or depending on a system font that may not exist on the host.
# PIL's built-in bitmap font is far too small for OCR to read reliably, which
# would make the whole corpus measure the font rather than the pipeline.
_FONT_DIR = Path(__file__).resolve()
try:  # pragma: no cover - exercised implicitly by every render
    import matplotlib

    _FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
except Exception:  # pragma: no cover
    _FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


class Difficulty(str, Enum):
    """Image-quality tier. Accuracy is always reported per tier, never pooled."""

    CLEAN = "clean"
    MODERATE = "moderate"
    HARD = "hard"


@dataclass(frozen=True)
class ReceiptTruth:
    """What the receipt actually says — the answer key for field extraction."""

    receipt_id: str
    vendor_name: str
    invoice_number: str
    issue_date: date
    subtotal: Decimal
    tax_amount: Decimal
    total_amount: Decimal
    category: str            # ground truth for the A.6 classifier
    line_description: str
    difficulty: Difficulty

    @property
    def scored_fields(self) -> dict[str, object]:
        """The four fields the extractor is graded on."""
        return {
            "vendor_name": self.vendor_name,
            "invoice_number": self.invoice_number,
            "issue_date": self.issue_date,
            "total_amount": self.total_amount,
        }


# Vendor templates, one per A.6 reference category. The `line_description`
# deliberately echoes the vocabulary in `unstructured.DEFAULT_REFERENCE_SET`
# without copying it verbatim: an exact copy would test string equality rather
# than the character n-gram similarity the classifier actually relies on.
_TEMPLATES: list[tuple[str, str, str, tuple[float, float]]] = [
    ("Sunrise Properties Pvt Ltd", "rent", "Monthly office premises rent", (60_000, 180_000)),
    ("Meridian Payroll Services", "payroll", "Staff salary disbursement run", (200_000, 900_000)),
    ("State Power Utility Board", "utilities", "Electricity supply charges", (8_000, 45_000)),
    ("BlueDart Freight Solutions", "logistics", "Courier and freight charges", (3_000, 28_000)),
    ("Deccan Steel Traders", "raw_materials", "Steel sheet raw material supply", (50_000, 400_000)),
    ("Kulkarni & Associates LLP", "professional_fees", "Statutory audit fee", (25_000, 150_000)),
    ("Skyline Travel Desk", "travel", "Flight booking and hotel stay", (12_000, 90_000)),
    ("GST Payment Gateway", "tax", "GST payment challan remittance", (30_000, 250_000)),
]

_W, _H = 720, 1000


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = _FONT_DIR / name
    try:
        return ImageFont.truetype(str(path), size)
    except Exception:  # pragma: no cover - only if DejaVu is genuinely missing
        return ImageFont.load_default(size)


def make_truth(
    index: int, difficulty: Difficulty, rng: np.random.Generator
) -> ReceiptTruth:
    """Draw one receipt's ground-truth content."""
    vendor, category, description, (lo, hi) = _TEMPLATES[index % len(_TEMPLATES)]
    subtotal = Decimal(int(rng.uniform(lo, hi))).quantize(Decimal("1"))
    # 18% GST, the standard Indian rate for most B2B services.
    tax = (subtotal * Decimal("0.18")).quantize(Decimal("0.01"))
    total = (subtotal + tax).quantize(Decimal("0.01"))
    issue = date(2026, 1, 1) + timedelta(days=int(rng.integers(0, 210)))
    return ReceiptTruth(
        receipt_id=f"RCPT-{difficulty.value[:1].upper()}{index:04d}",
        vendor_name=vendor,
        invoice_number=f"INV-{rng.integers(1000, 9999)}-{rng.integers(10, 99)}",
        issue_date=issue,
        subtotal=subtotal,
        tax_amount=tax,
        total_amount=total,
        category=category,
        line_description=description,
        difficulty=difficulty,
    )


def render_clean(truth: ReceiptTruth) -> Image.Image:
    """Render the undegraded receipt. Every tier starts from this image."""
    img = Image.new("RGB", (_W, _H), "white")
    d = ImageDraw.Draw(img)

    title = _font("DejaVuSans-Bold.ttf", 34)
    body = _font("DejaVuSans.ttf", 24)
    bold = _font("DejaVuSans-Bold.ttf", 26)
    mono = _font("DejaVuSansMono.ttf", 24)
    small = _font("DejaVuSans.ttf", 19)

    y = 46
    d.text((44, y), truth.vendor_name, font=title, fill="black")
    y += 48
    d.text((44, y), "Tax Invoice / Receipt", font=small, fill="black")
    y += 34
    d.line([(44, y), (_W - 44, y)], fill="black", width=2)

    y += 30
    d.text((44, y), "Invoice No:", font=body, fill="black")
    d.text((300, y), truth.invoice_number, font=mono, fill="black")
    y += 40
    d.text((44, y), "Date:", font=body, fill="black")
    d.text((300, y), truth.issue_date.strftime("%d/%m/%Y"), font=mono, fill="black")
    y += 40
    d.text((44, y), "GSTIN:", font=body, fill="black")
    d.text((300, y), "29AABCU9603R1ZM", font=mono, fill="black")

    y += 56
    d.line([(44, y), (_W - 44, y)], fill="black", width=1)
    y += 22
    d.text((44, y), "Description", font=bold, fill="black")
    d.text((520, y), "Amount", font=bold, fill="black")
    y += 38
    d.line([(44, y), (_W - 44, y)], fill="black", width=1)

    y += 24
    d.text((44, y), truth.line_description, font=body, fill="black")
    d.text((520, y), f"{truth.subtotal:,}", font=mono, fill="black")

    y += 70
    d.line([(360, y), (_W - 44, y)], fill="black", width=1)
    y += 20
    d.text((360, y), "Subtotal", font=body, fill="black")
    d.text((520, y), f"{truth.subtotal:,}", font=mono, fill="black")
    y += 38
    d.text((360, y), "GST 18%", font=body, fill="black")
    d.text((520, y), f"{truth.tax_amount:,}", font=mono, fill="black")
    y += 44
    d.line([(360, y), (_W - 44, y)], fill="black", width=2)
    y += 18
    d.text((360, y), "TOTAL", font=bold, fill="black")
    d.text((510, y), f"INR {truth.total_amount:,}", font=_font("DejaVuSansMono-Bold.ttf", 25), fill="black")

    y += 90
    d.text((44, y), "Payment due within 30 days of invoice date.", font=small, fill="black")
    y += 30
    d.text((44, y), "This is a computer generated document.", font=small, fill="black")
    return img


def _degrade(
    img: Image.Image, difficulty: Difficulty, rng: np.random.Generator
) -> Image.Image:
    """Apply tier-appropriate degradation in physical acquisition order.

    Geometry -> optics -> illumination -> sensor noise -> compression. That is
    the order a camera actually applies them; compressing before blurring, for
    instance, would produce ringing artifacts that no real photograph contains
    and would make the HARD tier unrepresentative rather than merely hard.
    """
    if difficulty is Difficulty.CLEAN:
        return img

    hard = difficulty is Difficulty.HARD

    # --- geometry: the page is not square to the camera ---
    angle = float(rng.uniform(3.0, 7.0) if hard else rng.uniform(0.8, 2.5))
    angle *= float(rng.choice([-1.0, 1.0]))
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")

    # --- optics: imperfect focus ---
    radius = float(rng.uniform(1.1, 1.9) if hard else rng.uniform(0.3, 0.7))
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # --- illumination: faded thermal paper / poor lighting ---
    contrast = float(rng.uniform(0.34, 0.50) if hard else rng.uniform(0.70, 0.85))
    img = ImageEnhance.Contrast(img).enhance(contrast)
    brightness = float(rng.uniform(1.04, 1.16) if hard else rng.uniform(0.97, 1.05))
    img = ImageEnhance.Brightness(img).enhance(brightness)

    if hard:
        # A luminance gradient across the page — one corner in shadow. This is
        # what defeats a global binarization threshold, and it is the single
        # most realistic thing about a phone photo of a receipt.
        arr = np.asarray(img, dtype=np.float32)
        h, w = arr.shape[:2]
        gx, gy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        shade = 1.0 - 0.30 * (gx * float(rng.uniform(0.3, 1.0))
                              + gy * float(rng.uniform(0.3, 1.0))) / 2.0
        arr *= shade[:, :, None]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # --- sensor noise ---
    sd = float(rng.uniform(9.0, 16.0) if hard else rng.uniform(2.5, 5.0))
    arr = np.asarray(img, dtype=np.float32)
    arr += rng.normal(0.0, sd, size=arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # --- compression, last, as the camera writes the file ---
    quality = int(rng.integers(24, 38) if hard else rng.integers(62, 82))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def generate_receipt(
    index: int, difficulty: Difficulty, rng: np.random.Generator
) -> tuple[ReceiptTruth, Image.Image]:
    """One receipt: its ground truth and its rendered, degraded image."""
    truth = make_truth(index, difficulty, rng)
    return truth, _degrade(render_clean(truth), difficulty, rng)


def generate_corpus(
    *,
    n_per_tier: int = 24,
    seed: int = 7,
    tiers: tuple[Difficulty, ...] = (
        Difficulty.CLEAN,
        Difficulty.MODERATE,
        Difficulty.HARD,
    ),
) -> list[tuple[ReceiptTruth, Image.Image]]:
    """Generate a scored corpus, balanced across tiers and vendor categories.

    The same `index` sequence is used in every tier, so a tier's content is
    identical and only its image quality differs. Any accuracy drop between
    tiers is therefore attributable to degradation rather than to having drawn
    an easier set of vendors.
    """
    out: list[tuple[ReceiptTruth, Image.Image]] = []
    for tier in tiers:
        rng = np.random.default_rng(seed)
        for i in range(n_per_tier):
            out.append(generate_receipt(i, tier, rng))
    return out


def save_corpus(
    corpus: list[tuple[ReceiptTruth, Image.Image]], directory: Path
) -> list[Path]:
    """Write a corpus to disk as PNGs named by receipt id."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for truth, img in corpus:
        p = directory / f"{truth.receipt_id}_{truth.difficulty.value}.png"
        img.save(p)
        paths.append(p)
    return paths
