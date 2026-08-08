"""Parse a bank statement export into domain records, whatever bank wrote it.

    csv text -> header location -> column-role inference -> reconciliation
             -> ParsedStatementRow -> Inflow / Outflow

THE CENTRAL IDEA: THE MAPPING IS SELECTED BY THE STATEMENT'S OWN ARITHMETIC
--------------------------------------------------------------------------
The hard part of statement parsing is not reading numbers, it is deciding what
the columns *mean*. Header names are unreliable — "Withdrawal (Dr)", "Debit",
"DR Amount" and "Amount" with a separate flag all denote the same thing, and
some exports label the balance column "Amount" too.

Rather than guess from names and hope, this parser **enumerates the plausible
mappings and picks the one that satisfies the running-balance identity**

    balance[i] == balance[i-1] + credit[i] - debit[i]

on every row. That inverts the usual design: the identity is normally used to
validate a mapping chosen by other means, and here it *is* the means. This
matters because the two failures that a name-based parser cannot detect —
debit and credit swapped, and a sign convention inverted — are exactly the two
the identity rejects instantly, while both produce a completely well-formed
ledger that is wrong in every row.

Header evidence is still used, for two things it is genuinely good for:
breaking ties between mappings that reconcile equally well, and supplying the
per-column confidence reported in the result.

WHEN THERE IS NO BALANCE COLUMN
-------------------------------
Some exports omit it. Then the identity is unavailable, the parser falls back
to header and value-shape evidence, and the reconciliation report says
`checkable = False`. It does not say `passed = True`. An unchecked parse and a
verified parse are different claims and the schema keeps them distinct.

REFUSAL
-------
Like `normalize()` on the receipt path, this refuses rather than emitting a
partially-trusted ledger. A statement that cannot be reconciled is returned
with `rejected = True` and a diagnosis, not as rows the optimizer will treat
as fact.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Iterable, Optional, Sequence
from uuid import UUID, uuid5, NAMESPACE_URL

from app.schemas.base import SourceType
from app.schemas.core import Inflow, Outflow
from app.schemas.solvency import (
    AmountConvention,
    ColumnAssignment,
    ColumnRole,
    ParsedStatementRow,
    ReconciliationReport,
    StatementParseResult,
)

# Absolute currency tolerance per row. One paisa, not a percentage: the
# identity is exact arithmetic on quantized decimals, so any residual larger
# than a rounding unit is a real disagreement rather than accumulated error.
DEFAULT_TOLERANCE = 0.011

# Header vocabulary per role. Matched fuzzily, because exports vary in wording
# far more than in meaning.
_HEADER_HINTS: dict[ColumnRole, tuple[str, ...]] = {
    ColumnRole.DATE: ("date", "txn date", "posted date", "transaction date", "tran date"),
    ColumnRole.VALUE_DATE: ("value date", "val date", "effective date"),
    ColumnRole.DESCRIPTION: (
        "description", "narration", "particulars", "details", "memo",
        "transaction details", "remarks",
    ),
    ColumnRole.REFERENCE: ("reference", "ref no", "cheque", "chq", "cheque/ref no", "utr"),
    ColumnRole.DEBIT: ("debit", "withdrawal", "withdrawal (dr)", "dr", "paid out", "dr amount"),
    ColumnRole.CREDIT: ("credit", "deposit", "deposit (cr)", "cr", "paid in", "cr amount"),
    ColumnRole.SIGNED_AMOUNT: ("amount", "value", "txn amount", "transaction amount"),
    ColumnRole.BALANCE: ("balance", "running balance", "closing balance", "bal"),
}

_INDICATOR_TOKENS = {"dr", "cr", "d", "c", "debit", "credit", "db"}

# Ordered by decreasing specificity: a 4-digit leading group can only be ISO,
# and an alphabetic month can only be the %b forms, so those never contend.
_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d",
    "%d-%b-%Y", "%d %b %Y", "%d-%B-%Y",
    "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
    "%d/%m/%y", "%m/%d/%y",
)

# Formats that are indistinguishable on a column whose days are all <= 12.
_AMBIGUOUS_PAIRS = {
    frozenset({"%d/%m/%Y", "%m/%d/%Y"}),
    frozenset({"%d-%m-%Y", "%m-%d-%Y"}),
    frozenset({"%d/%m/%y", "%m/%d/%y"}),
}


class StatementParseError(ValueError):
    """The file could not be read as a statement at all."""


def _fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _header_score(header: str, role: ColumnRole) -> float:
    """How much the header text supports this role, in [0, 1].

    Substring containment scores high because bank headers decorate rather than
    rename ("Withdrawal (Dr)" contains "withdrawal"); fuzzy ratio catches the
    abbreviations that do rename.
    """
    h = header.lower().strip()
    if not h:
        return 0.0
    best = 0.0
    for hint in _HEADER_HINTS.get(role, ()):
        if h == hint:
            return 1.0
        if hint in h:
            best = max(best, 0.85)
        best = max(best, _fuzzy(h, hint))
    return best


# ---------------------------------------------------------------------------
# Cell parsing
# ---------------------------------------------------------------------------

_PAREN_RE = re.compile(r"^\((.*)\)$")
_TRAILING_MARK_RE = re.compile(r"\b(cr|dr)\.?$", re.I)
_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


def parse_amount(cell: str) -> Optional[Decimal]:
    """Read one currency cell, or None when it is blank or not a number.

    Handles, in order:

      * blank / '-' / 'NIL'  -> None (an empty debit cell means "not a debit",
        which is different from a debit of zero, and collapsing the two is how
        a sparse-column pair stops being distinguishable);
      * parentheses as negation, the US export convention;
      * a trailing `Cr` / `Dr` marker, the Indian convention, which sets the
        sign and is then removed;
      * comma separators under BOTH western (1,234,567.89) and Indian lakh
        (12,34,567.89) grouping. Stripping the separator handles both, which is
        why the parser never needs to know which country wrote the file — a
        positional de-grouping routine would have to.
    """
    s = (cell or "").strip()
    if not s or s in {"-", "--", "NIL", "nil", "N/A"}:
        return None

    s = re.sub(r"(?i)^(inr|rs\.?|usd|eur|₹|\$|€)\s*", "", s).strip()

    sign = Decimal(1)
    m = _PAREN_RE.match(s)
    if m:
        sign, s = Decimal(-1), m.group(1).strip()

    mark = _TRAILING_MARK_RE.search(s)
    if mark:
        if mark.group(1).lower() == "dr":
            sign = -sign
        s = _TRAILING_MARK_RE.sub("", s).strip()

    s = s.replace(",", "").replace(" ", "")
    if s.startswith("-"):
        sign, s = -sign, s[1:]
    if s.startswith("+"):
        s = s[1:]

    if not _NUMERIC_RE.match(s):
        return None
    try:
        return (sign * Decimal(s)).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def _parse_date_with(cell: str, fmt: str) -> Optional[date]:
    s = (cell or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, fmt).date()
    except ValueError:
        return None


def infer_date_format(values: Sequence[str]) -> tuple[Optional[str], list[str]]:
    """Choose the date format from the whole column's evidence.

    THREE FILTERS, IN ORDER
    -----------------------
    1. **Parses every non-empty cell.** A format that fails one row is wrong,
       not "mostly right" — statements do not mix date formats.
    2. **Yields a non-decreasing sequence.** Statements are in posting order,
       so a format that reads the column into shuffled dates has swapped the
       day and month fields. This is what resolves dd/mm vs mm/dd on most real
       files without needing a single day above 12.
    3. **Genuine ambiguity is reported, not resolved.** When two formats both
       parse everything and both come out ordered — which happens when every
       day in the file is <= 12 — the column truly does not say which is meant.
       The parser returns its preferred reading AND a warning naming the other,
       rather than picking silently. Silently picking is how a statement
       reconciles perfectly with every date wrong.

    Returns:
        (format or None, warnings)
    """
    nonempty = [v for v in values if (v or "").strip()]
    if not nonempty:
        return None, ["date column was empty"]

    viable: list[str] = []
    for fmt in _DATE_FORMATS:
        parsed = [_parse_date_with(v, fmt) for v in nonempty]
        if any(p is None for p in parsed):
            continue
        viable.append(fmt)

    if not viable:
        return None, ["no candidate date format parsed every row"]

    def ordered(fmt: str) -> bool:
        ds = [_parse_date_with(v, fmt) for v in nonempty]
        return all(a <= b for a, b in zip(ds, ds[1:]))

    ordered_fmts = [f for f in viable if ordered(f)]
    pool = ordered_fmts or viable
    chosen = pool[0]

    warnings: list[str] = []
    if not ordered_fmts:
        warnings.append(
            f"no candidate date format produced a chronologically ordered column; "
            f"using {chosen!r}, but the posting order should be checked"
        )
    for other in pool[1:]:
        if frozenset({chosen, other}) in _AMBIGUOUS_PAIRS:
            warnings.append(
                f"date column is genuinely ambiguous between {chosen!r} and {other!r} "
                f"— every day in the file is <= 12, so the column itself cannot say "
                f"which is meant. Reading it as {chosen!r}. If this statement is from "
                f"a US bank the dates are month-first and are currently wrong."
            )
            break
    return chosen, warnings


# ---------------------------------------------------------------------------
# Header location and column typing
# ---------------------------------------------------------------------------

def _read_rows(text: str) -> list[list[str]]:
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    return [row for row in csv.reader(io.StringIO(text), dialect)]


def locate_header(rows: Sequence[Sequence[str]], max_scan: int = 25) -> int:
    """Find the row that names the columns, past any account preamble.

    Scored rather than assumed-to-be-row-0, because several bank exports print
    account number, currency and opening balance above the table. A parser that
    takes row 0 reads "Account Statement" as a column name and every subsequent
    inference is built on it.

    The score rewards a row whose cells look like header vocabulary and whose
    width matches the rows beneath it — a preamble line is usually narrower
    than the table it precedes.
    """
    widths = Counter(len(r) for r in rows if len(r) > 1)
    modal_width = widths.most_common(1)[0][0] if widths else 0

    best_idx, best_score = 0, -1.0
    for i, row in enumerate(rows[:max_scan]):
        if len(row) < 2:
            continue
        vocab = sum(
            max(_header_score(cell, role) for role in _HEADER_HINTS)
            for cell in row
            if (cell or "").strip()
        )
        # A header row's own cells are labels, so almost none of them should
        # parse as a number. This is what stops a data row from outscoring the
        # real header on vocabulary alone.
        numeric_cells = sum(1 for c in row if parse_amount(c) is not None)
        score = vocab - 2.0 * numeric_cells + (1.5 if len(row) == modal_width else 0.0)
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


@dataclass(frozen=True)
class _Column:
    index: int
    header: str
    values: list[str]

    def numeric_fraction(self) -> float:
        nonempty = [v for v in self.values if (v or "").strip()]
        if not nonempty:
            return 0.0
        return sum(1 for v in nonempty if parse_amount(v) is not None) / len(nonempty)

    def fill_fraction(self) -> float:
        return (
            sum(1 for v in self.values if (v or "").strip()) / len(self.values)
            if self.values else 0.0
        )

    def is_indicator(self) -> bool:
        vals = {(v or "").strip().lower() for v in self.values if (v or "").strip()}
        return bool(vals) and vals <= _INDICATOR_TOKENS

    def is_date_like(self) -> bool:
        nonempty = [v for v in self.values if (v or "").strip()]
        if not nonempty:
            return False
        fmt, _ = infer_date_format(nonempty)
        return fmt is not None


# ---------------------------------------------------------------------------
# Candidate mappings, scored by reconciliation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Candidate:
    convention: AmountConvention
    debit_idx: Optional[int]
    credit_idx: Optional[int]
    amount_idx: Optional[int]
    indicator_idx: Optional[int]
    balance_idx: Optional[int]

    def header_support(self, cols: Sequence[_Column]) -> float:
        """Prior from header names, used to break reconciliation ties."""
        s = 0.0
        if self.debit_idx is not None:
            s += _header_score(cols[self.debit_idx].header, ColumnRole.DEBIT)
        if self.credit_idx is not None:
            s += _header_score(cols[self.credit_idx].header, ColumnRole.CREDIT)
        if self.amount_idx is not None:
            s += _header_score(cols[self.amount_idx].header, ColumnRole.SIGNED_AMOUNT)
        if self.balance_idx is not None:
            s += _header_score(cols[self.balance_idx].header, ColumnRole.BALANCE)
        return s


def _signed_pairs(
    cand: _Candidate, cols: Sequence[_Column], n_rows: int
) -> list[tuple[Decimal, Decimal]]:
    """Resolve each row to (debit, credit) under this candidate mapping."""
    out: list[tuple[Decimal, Decimal]] = []
    zero = Decimal("0.00")
    for i in range(n_rows):
        if cand.convention is AmountConvention.SEPARATE_DEBIT_CREDIT:
            d = parse_amount(cols[cand.debit_idx].values[i]) or zero
            c = parse_amount(cols[cand.credit_idx].values[i]) or zero
            out.append((abs(d), abs(c)))
        elif cand.convention is AmountConvention.SIGNED_SINGLE_COLUMN:
            a = parse_amount(cols[cand.amount_idx].values[i]) or zero
            out.append((-a, zero) if a < 0 else (zero, a))
        else:  # AMOUNT_WITH_INDICATOR
            a = abs(parse_amount(cols[cand.amount_idx].values[i]) or zero)
            flag = (cols[cand.indicator_idx].values[i] or "").strip().lower()
            out.append((a, zero) if flag in {"dr", "d", "debit", "db"} else (zero, a))
    return out


def _reconcile(
    pairs: Sequence[tuple[Decimal, Decimal]],
    balances: Optional[Sequence[Optional[Decimal]]],
    tolerance: float,
) -> tuple[ReconciliationReport, list[bool]]:
    """Check `balance[i] == balance[i-1] + credit - debit` across the file."""
    n = len(pairs)
    if balances is None or any(b is None for b in balances) or n == 0:
        return (
            ReconciliationReport(
                checkable=False, n_rows=n, n_rows_reconciled=0,
                max_absolute_residual=0.0, mean_absolute_residual=0.0,
                tolerance=tolerance, passed=False,
                diagnosis="statement carried no usable balance column, so the "
                          "running-balance identity could not be evaluated",
            ),
            [True] * n,
        )

    residuals: list[float] = []
    flags: list[bool] = []
    failing: list[int] = []
    # Row 0 is checked against the implied opening balance rather than skipped,
    # by reconstructing balance[-1] = balance[0] - credit[0] + debit[0]; that is
    # trivially satisfied, so row 0 is reported as reconciled but contributes no
    # evidence. Every later row is a genuine constraint.
    for i in range(1, n):
        d, c = pairs[i]
        expected = balances[i - 1] + c - d
        r = abs(float(expected - balances[i]))
        residuals.append(r)
        ok = r <= tolerance
        flags.append(ok)
        if not ok and len(failing) < 20:
            failing.append(i)

    flags = [True] + flags
    max_r = max(residuals) if residuals else 0.0
    mean_r = sum(residuals) / len(residuals) if residuals else 0.0
    passed = not failing and bool(residuals)

    diagnosis = None
    if not passed and residuals:
        diagnosis = _diagnose(pairs, residuals, tolerance)

    return (
        ReconciliationReport(
            checkable=True,
            n_rows=n,
            n_rows_reconciled=sum(flags),
            max_absolute_residual=max_r,
            mean_absolute_residual=mean_r,
            tolerance=tolerance,
            passed=passed,
            failing_row_indices=failing,
            diagnosis=diagnosis,
        ),
        flags,
    )


def _diagnose(
    pairs: Sequence[tuple[Decimal, Decimal]], residuals: Sequence[float], tolerance: float
) -> str:
    """Name the most likely cause from the shape of the residuals.

    'The file does not reconcile' is not actionable. These three patterns cover
    the mistakes that actually happen, and each has a signature that noise does
    not:

      * every residual is exactly twice that row's amount -> the direction is
        inverted (debit and credit swapped, or the sign convention read
        backwards). Doubling is the signature: correcting a flow of `x` in the
        wrong direction moves the balance by `2x`.
      * one or two rows fail and the rest are clean -> a localized bad cell,
        not a mapping error.
      * residuals grow monotonically -> a row is missing from the file, or the
        statement spans a page break that dropped a line.
    """
    doubling = 0
    for i, r in enumerate(residuals, start=1):
        amt = float(pairs[i][0] + pairs[i][1])
        if amt > 0 and abs(r - 2.0 * amt) <= max(tolerance, 0.01 * amt):
            doubling += 1
    if residuals and doubling / len(residuals) > 0.8:
        return (
            "every failing row's residual is twice its own amount, which is the "
            "signature of an inverted direction: debit and credit are swapped, or "
            "the signed column's sign convention is backwards"
        )

    n_bad = sum(1 for r in residuals if r > tolerance)
    if n_bad <= max(2, len(residuals) // 20):
        return (
            f"{n_bad} of {len(residuals)} rows fail while the rest reconcile exactly "
            "— this is a localized bad cell, not a column-mapping error"
        )

    if all(b >= a - tolerance for a, b in zip(residuals, residuals[1:])):
        return (
            "residuals increase monotonically down the file, which is what a MISSING "
            "ROW produces: every balance after the gap is off by the same omitted "
            "amount. Check for a dropped line at a page break"
        )
    return "residual pattern does not match a known single cause; the mapping is suspect"


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------

def parse_statement(
    text: str,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    opening_balance: Optional[Decimal] = None,
) -> StatementParseResult:
    """Parse a bank statement CSV export into `StatementParseResult`.

    Args:
        text: raw file contents.
        tolerance: absolute per-row currency tolerance for the identity.
        opening_balance: balance before the first row, when known from a
            preamble. Only used to report `opening_balance`; the identity is
            evaluated row-to-row and does not depend on it.

    Returns:
        A result that is either usable (`rejected = False`) or refused with a
        stated reason. Never a partially-trusted ledger.

    Raises:
        StatementParseError: the file is not tabular at all.
    """
    rows = _read_rows(text)
    if not rows:
        raise StatementParseError("file is empty")

    h_idx = locate_header(rows)
    header = [c.strip() for c in rows[h_idx]]
    width = len(header)
    body = [r for r in rows[h_idx + 1:] if any((c or "").strip() for c in r)]
    body = [list(r) + [""] * (width - len(r)) for r in body if len(r) <= width or True]
    body = [r[:width] for r in body]
    if not body:
        raise StatementParseError("no data rows beneath the header")

    cols = [
        _Column(index=j, header=header[j] if j < len(header) else "",
                values=[r[j] for r in body])
        for j in range(width)
    ]
    n = len(body)
    warnings: list[str] = []

    # --- date column ---
    date_cols = [c for c in cols if c.is_date_like() and c.fill_fraction() > 0.9]
    if not date_cols:
        raise StatementParseError("no column parses as a date on every row")
    # Prefer the one whose header says so; otherwise the leftmost.
    date_col = max(
        date_cols,
        key=lambda c: (_header_score(c.header, ColumnRole.DATE), -c.index),
    )
    date_fmt, date_warnings = infer_date_format(date_col.values)
    warnings.extend(date_warnings)
    if date_fmt is None:
        raise StatementParseError("could not infer a date format for the date column")

    # --- typed column pools ---
    numeric = [
        c for c in cols
        if c is not date_col and c.numeric_fraction() > 0.95 and c.fill_fraction() > 0.0
        and not c.is_indicator()
    ]
    indicators = [c for c in cols if c.is_indicator()]
    if not numeric:
        raise StatementParseError("no numeric column found")

    # --- enumerate candidate mappings ---
    candidates: list[_Candidate] = []
    balance_options: list[Optional[int]] = [c.index for c in numeric] + [None]

    for bal in balance_options:
        pool = [c for c in numeric if c.index != bal]
        # (a) two sparse complementary columns -> debit / credit, both orders
        for a in pool:
            for b in pool:
                if a.index == b.index:
                    continue
                candidates.append(
                    _Candidate(AmountConvention.SEPARATE_DEBIT_CREDIT,
                               a.index, b.index, None, None, bal)
                )
        # (b) one signed column
        for a in pool:
            candidates.append(
                _Candidate(AmountConvention.SIGNED_SINGLE_COLUMN,
                           None, None, a.index, None, bal)
            )
        # (c) one amount column plus an indicator
        for a in pool:
            for ind in indicators:
                candidates.append(
                    _Candidate(AmountConvention.AMOUNT_WITH_INDICATOR,
                               None, None, a.index, ind.index, bal)
                )

    if not candidates:
        raise StatementParseError("no plausible column mapping could be formed")

    # --- score every candidate by the identity, tie-broken by header support ---
    scored = []
    for cand in candidates:
        try:
            pairs = _signed_pairs(cand, cols, n)
        except (IndexError, TypeError):
            continue
        bals = (
            [parse_amount(cols[cand.balance_idx].values[i]) for i in range(n)]
            if cand.balance_idx is not None else None
        )
        report, flags = _reconcile(pairs, bals, tolerance)
        # A candidate that reconciles but assigns every row a zero amount is
        # degenerate — it satisfies the identity vacuously on a constant
        # balance column. Requiring real movement rules it out.
        movement = sum(1 for d, c in pairs if d > 0 or c > 0)
        scored.append((cand, pairs, bals, report, flags, movement))

    checkable = [s for s in scored if s[3].checkable]
    if checkable:
        best = max(
            checkable,
            key=lambda s: (
                s[3].passed,
                s[5] / max(n, 1) > 0.5,
                -s[3].mean_absolute_residual,
                s[0].header_support(cols),
            ),
        )
    else:
        # No balance column anywhere: fall back to header + shape evidence.
        best = max(
            scored,
            key=lambda s: (s[5] / max(n, 1) > 0.5, s[0].header_support(cols)),
        )
        warnings.append(
            "no balance column: the mapping was chosen from header and value-shape "
            "evidence alone and has NOT been verified against the running-balance "
            "identity"
        )

    cand, pairs, bals, report, flags, _mv = best

    # --- descriptive columns ---
    used = {date_col.index, cand.debit_idx, cand.credit_idx, cand.amount_idx,
            cand.indicator_idx, cand.balance_idx}
    text_cols = [c for c in cols if c.index not in used and c.numeric_fraction() < 0.5]
    desc_col = max(
        text_cols, key=lambda c: (_header_score(c.header, ColumnRole.DESCRIPTION),
                                  c.fill_fraction()),
        default=None,
    )
    ref_col = max(
        [c for c in text_cols if desc_col is None or c.index != desc_col.index],
        key=lambda c: _header_score(c.header, ColumnRole.REFERENCE),
        default=None,
    )

    # --- assignments, with confidence relative to the runner-up role ---
    assignments = _build_assignments(cols, cand, date_col, desc_col, ref_col)

    # --- rows ---
    parsed_rows: list[ParsedStatementRow] = []
    zero = Decimal("0.00")
    for i in range(n):
        d, c = pairs[i]
        posted = _parse_date_with(date_col.values[i], date_fmt)
        if posted is None:
            continue
        parsed_rows.append(
            ParsedStatementRow(
                row_index=i,
                posted_date=posted,
                description=(desc_col.values[i].strip() if desc_col else ""),
                reference=(ref_col.values[i].strip() or None) if ref_col else None,
                debit=d.quantize(Decimal("0.01")),
                credit=c.quantize(Decimal("0.01")),
                balance=(bals[i] if bals else None),
                reconciled=flags[i],
            )
        )

    total_debits = sum((r.debit for r in parsed_rows), zero)
    total_credits = sum((r.credit for r in parsed_rows), zero)

    rejected = report.checkable and not report.passed
    rejection_reason = None
    if rejected:
        rejection_reason = (
            f"statement does not reconcile: {report.n_rows - report.n_rows_reconciled} "
            f"of {report.n_rows} rows break the running-balance identity "
            f"(max residual {report.max_absolute_residual:.2f}). {report.diagnosis} "
            "Refusing to emit records — a misparsed statement produces a "
            "well-formed ledger that is wrong in every row."
        )

    closing = None
    for r in reversed(parsed_rows):
        if r.balance is not None:
            closing = r.balance
            break

    implied_opening = opening_balance
    if implied_opening is None and parsed_rows and parsed_rows[0].balance is not None:
        implied_opening = (
            parsed_rows[0].balance - parsed_rows[0].credit + parsed_rows[0].debit
        )

    return StatementParseResult(
        dialect_name=None,
        convention=cand.convention,
        date_format=date_fmt,
        column_assignments=assignments,
        rows=parsed_rows,
        reconciliation=report,
        opening_balance=implied_opening,
        closing_balance=closing,
        total_credits=total_credits,
        total_debits=total_debits,
        rejected=rejected,
        rejection_reason=rejection_reason,
        warnings=warnings,
    )


def _build_assignments(
    cols: Sequence[_Column],
    cand: _Candidate,
    date_col: _Column,
    desc_col: Optional[_Column],
    ref_col: Optional[_Column],
) -> list[ColumnAssignment]:
    """Describe the winning mapping, with per-column confidence and evidence.

    Confidence is the winning role's header score normalized against the best
    competing role's, so a column named "Debit" assigned to DEBIT reports high
    confidence while one named "Amount" assigned to DEBIT by reconciliation
    alone reports low confidence — correctly. Low confidence beside a passing
    reconciliation is not a problem; low confidence beside a failing one names
    the suspect.
    """
    roles: dict[int, ColumnRole] = {date_col.index: ColumnRole.DATE}
    if cand.debit_idx is not None:
        roles[cand.debit_idx] = ColumnRole.DEBIT
    if cand.credit_idx is not None:
        roles[cand.credit_idx] = ColumnRole.CREDIT
    if cand.amount_idx is not None:
        roles[cand.amount_idx] = ColumnRole.SIGNED_AMOUNT
    if cand.balance_idx is not None:
        roles[cand.balance_idx] = ColumnRole.BALANCE
    if desc_col is not None:
        roles[desc_col.index] = ColumnRole.DESCRIPTION
    if ref_col is not None:
        roles.setdefault(ref_col.index, ColumnRole.REFERENCE)

    out: list[ColumnAssignment] = []
    for c in cols:
        role = roles.get(c.index, ColumnRole.IGNORED)
        own = _header_score(c.header, role) if role in _HEADER_HINTS else 0.0
        rival = max(
            (_header_score(c.header, r) for r in _HEADER_HINTS if r is not role),
            default=0.0,
        )
        if role is ColumnRole.IGNORED:
            conf, evidence = 1.0, "no role matched; column carried to no record"
        elif own <= 0.0:
            conf = 0.25
            evidence = (
                "header gave no support; the role was determined by the "
                "running-balance identity, which is the stronger evidence"
            )
        else:
            conf = min(1.0, own / (own + rival) if (own + rival) > 0 else own)
            evidence = f"header match {own:.2f} against best rival role {rival:.2f}"
        out.append(
            ColumnAssignment(
                source_name=c.header, column_index=c.index,
                role=role, confidence=float(conf), evidence=evidence,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Statement rows -> domain records
# ---------------------------------------------------------------------------

_NARRATIVE_NOISE = re.compile(
    r"\b(NEFT|IMPS|RTGS|UPI|ACH|CHQ|DEP|CR|DR|REF|TXN|BATCH|PAYMENT|BILL|CHALLAN|"
    r"DEBIT|CREDIT|AC|LOAN|EMI|SALARY|GST|RENT|ELECTRICITY|UTILITIES)\b",
    re.I,
)

# Any whitespace-delimited token containing a digit. This is what removes
# `INV449182`, `P2M` and bare reference numbers in one rule. A word-boundary
# alternation cannot do it: there is no boundary between the `V` and the `4` of
# `INV449182`, so `\bINV\b` never matches and the reference survives into the
# counterparty name. That was a real bug, caught by the extraction test.
_TOKEN_WITH_DIGIT = re.compile(r"\S*\d\S*")


def counterparty_from_narrative(description: str) -> str:
    """Recover a counterparty name from a bank narrative.

    Bank narratives are rail metadata plus a name: `NEFT DR ACME SUPPLIES
    INV449182`. Stripping the known rail tokens and the reference numbers
    leaves the name. This is deliberately a *heuristic on a known vocabulary*
    rather than the A.6 embedding classifier, because the target here is a
    proper noun the reference set has never seen — the classifier's job is
    category, and asking it for identity would be using it outside what it was
    fitted for.

    Returns `"UNKNOWN"` rather than guessing when nothing survives; an
    unattributed transaction is a real outcome and the ledger should say so.
    """
    # Order matters. Separators are split FIRST so that `UPI/P2M/ORION` becomes
    # three tokens; only then can the digit rule drop `P2M` while keeping
    # `ORION`. Applied the other way round, the whole slash-joined run counts as
    # one token containing a digit and the vendor name is destroyed with it.
    s = re.sub(r"[/\\|]", " ", description or "")
    s = _TOKEN_WITH_DIGIT.sub(" ", s)
    s = _NARRATIVE_NOISE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" -_.")
    return s.title() if len(s) >= 3 else "UNKNOWN"


def _record_id(business_id: UUID, kind: str, row: ParsedStatementRow, acct: str) -> UUID:
    """Stable id from the transaction's own identity.

    Deterministic for the same reason the receipt path is: re-ingesting a
    statement that overlaps a previous pull must not double-post. Bank
    statements overlap constantly — a monthly pull that fetches 35 days to be
    safe re-delivers five days of rows every time — so idempotency here is not
    an edge case, it is the normal path.
    """
    key = (
        f"finascend/statement/{business_id}/{acct}/{kind}/{row.posted_date.isoformat()}/"
        f"{row.debit}/{row.credit}/{row.description}"
    )
    return uuid5(NAMESPACE_URL, key)


def to_records(
    result: StatementParseResult,
    *,
    business_id: UUID,
    account_reference: str,
    source_type: SourceType = SourceType.BANK_STATEMENT,
) -> tuple[list[Inflow], list[Outflow]]:
    """Convert reconciled statement rows into `Inflow` / `Outflow` records.

    ON `Inflow.certainty`
    ---------------------
    §2.1 struck the `certainty = 1.0` *default* because asserting perfect
    certainty about a **receivable** — money expected but not arrived — is the
    least defensible claim in the schema. A settled bank credit is a different
    object: the money is in the account, `received_date` is set, and
    `is_receivable` is False. Certainty 1.0 here is an observation, not an
    assumption, and it is set explicitly at the one place that can justify it
    rather than restored as a default. A statement line for a *pending* credit
    would not qualify and is not emitted.

    Raises:
        ValueError: the parse was rejected. Records are never built from a
            statement that failed its own arithmetic.
    """
    if result.rejected:
        raise ValueError(
            f"refusing to build records from a rejected parse: {result.rejection_reason}"
        )

    now = datetime.now(timezone.utc)
    inflows: list[Inflow] = []
    outflows: list[Outflow] = []

    for row in result.rows:
        if not row.reconciled:
            # A single unreconciled row is dropped rather than admitted, and the
            # parse-level report still names its index, so the omission is
            # visible rather than silent.
            continue
        name = counterparty_from_narrative(row.description)
        if row.credit > 0:
            inflows.append(
                Inflow(
                    id=_record_id(business_id, "in", row, account_reference),
                    business_id=business_id,
                    amount=row.credit,
                    expected_date=row.posted_date,
                    received_date=row.posted_date,
                    counterparty_name=name,
                    source_type=source_type,
                    source_reference=row.reference or account_reference,
                    is_receivable=False,
                    certainty=1.0,
                    created_at=now,
                )
            )
        elif row.debit > 0:
            outflows.append(
                Outflow(
                    id=_record_id(business_id, "out", row, account_reference),
                    business_id=business_id,
                    amount=row.debit,
                    due_date=row.posted_date,
                    paid_date=row.posted_date,
                    counterparty_name=name,
                    category="uncategorized",
                    source_type=source_type,
                    source_reference=row.reference or account_reference,
                    created_at=now,
                )
            )
    return inflows, outflows


def ingest_statement(
    text: str,
    *,
    business_id: UUID,
    account_reference: str,
    source_type: SourceType = SourceType.BANK_STATEMENT,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[StatementParseResult, list[Inflow], list[Outflow]]:
    """Whole chain: text in, parse result and domain records out.

    Records come back empty when the parse was rejected, rather than the
    rejection being raised — one bad statement in a batch should not discard
    the batch, and the caller can see exactly what failed and why.
    """
    result = parse_statement(text, tolerance=tolerance)
    if result.rejected:
        return result, [], []
    ins, outs = to_records(
        result, business_id=business_id, account_reference=account_reference,
        source_type=source_type,
    )
    return result, ins, outs
