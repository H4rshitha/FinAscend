"""Synthetic bank statements with known ground truth, in several bank dialects.

The same principle as A.0's cash-flow generator and the receipt generator: a
parser can only be *scored* against inputs whose correct answer is known in
advance. A parser demonstrated on one exported CSV is an anecdote.

WHY DIALECTS, NOT DIFFICULTY TIERS
----------------------------------
The receipt corpus varies by image quality, because that is what breaks OCR. A
statement is already machine-readable, so image quality is not the adversary —
**layout disagreement between banks is**. There is no standard. The same three
facts (when, how much, which way) are expressed as:

  * one signed amount column, or two columns named Debit/Credit, or one amount
    column plus a separate Dr/Cr marker;
  * dates as dd/mm/yyyy, mm/dd/yyyy, dd-Mon-yyyy or yyyy-mm-dd;
  * amounts with thousands separators, with Indian lakh grouping, with a
    trailing `Cr`, wrapped in parentheses for negatives, or bare;
  * headers that may be on row 1 or after a block of account preamble.

A parser that handles one bank's export is not a parser, it is a mapping. The
generator therefore emits the same underlying ledger through several dialects,
so a correct parser must produce the *identical* records from all of them.
That equivalence is the strongest available test and it needs no golden files.

THE TRAP THAT IS DELIBERATELY SET
---------------------------------
`AMBIGUOUS_MMDD` emits dates where the day is <= 12 far more often than chance,
so dd/mm and mm/dd are both parseable and only one is right. A parser that
guesses from the first row passes by luck; one that infers the format from the
whole column's evidence — and refuses when the column genuinely cannot
disambiguate — is what the test suite requires. Silently defaulting to one
convention is the failure mode this exists to catch, because it produces a
statement that reconciles perfectly with every date wrong.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, Sequence

import numpy as np

# Grouped by direction so the generated descriptions carry real category signal
# rather than being lorem text — the same reason A.6's classifier is given the
# whole receipt line rather than the vendor name alone.
_DEBIT_NARRATIVES = [
    ("ACH DR SALARY BATCH {ref}", "payroll"),
    ("NEFT DR {vendor} INV{ref}", "vendor_payment"),
    ("RENT PAYMENT {vendor} {ref}", "rent"),
    ("GST CHALLAN {ref}", "tax"),
    ("EMI DEBIT LOAN AC {ref}", "loan_emi"),
    ("UPI/P2M/{vendor}/{ref}", "vendor_payment"),
    ("ELECTRICITY BILL {vendor} {ref}", "utilities"),
]

_CREDIT_NARRATIVES = [
    ("NEFT CR {vendor} INV{ref}", "customer_receipt"),
    ("IMPS CR {vendor} {ref}", "customer_receipt"),
    ("UPI/P2M/{vendor}/CR/{ref}", "customer_receipt"),
    ("CHQ DEP {ref}", "customer_receipt"),
]

_VENDORS = [
    "SUNRISE PROPERTIES", "ACME SUPPLIES", "NORTHWIND TRADING", "VERTEX LOGISTICS",
    "BLUEPEAK SERVICES", "ORION TEXTILES", "KAVERI FOODS", "MERIDIAN TECH",
    "GREENFIELD AGRO", "CASTLE HARDWARE",
]


class Dialect(str, Enum):
    """A bank export layout. Each is a real convention, not a synthetic variant."""

    # Two amount columns, dd/mm/yyyy, header on row 1. The straightforward case.
    SIMPLE_DEBIT_CREDIT = "simple_debit_credit"
    # One signed column; negatives are money out.
    SIGNED_AMOUNT = "signed_amount"
    # One positive amount column plus a Dr/Cr indicator. Direction is invisible
    # to a numeric heuristic, so this is the case a sign-based parser inverts.
    AMOUNT_WITH_DR_CR_FLAG = "amount_with_dr_cr_flag"
    # Account preamble above the header, ISO dates, lakh grouping, and a
    # trailing 'Cr' on the balance. Modelled on Indian bank exports.
    INDIAN_BANK_PREAMBLE = "indian_bank_preamble"
    # US-style mm/dd/yyyy with parenthesised negatives, no balance column at
    # all — so the reconciliation invariant is unavailable and the parser must
    # report that rather than claim a pass.
    US_NO_BALANCE = "us_no_balance"
    # dd/mm/yyyy where the day is usually <= 12. The disambiguation trap.
    AMBIGUOUS_MMDD = "ambiguous_mmdd"


@dataclass(frozen=True)
class StatementTruth:
    """The answer key: what the file means, independent of how it is written."""

    account_reference: str
    opening_balance: Decimal
    closing_balance: Decimal
    # (posted_date, description, debit, credit, running_balance, category)
    rows: list[tuple[date, str, Decimal, Decimal, Decimal, str]]
    dialect: Dialect
    date_format: str
    currency: str = "INR"

    @property
    def total_debits(self) -> Decimal:
        return sum((r[2] for r in self.rows), Decimal("0.00"))

    @property
    def total_credits(self) -> Decimal:
        return sum((r[3] for r in self.rows), Decimal("0.00"))


def _q(x: float) -> Decimal:
    """Quantize to paise. Every amount in the truth is exact by construction."""
    return Decimal(str(round(float(x), 2))).quantize(Decimal("0.01"))


def make_truth(
    *,
    seed: int,
    dialect: Dialect,
    n_rows: int = 40,
    start: date = date(2026, 1, 5),
    opening_balance: float = 850_000.0,
) -> StatementTruth:
    """Generate one statement's underlying ledger and its running balance.

    The running balance is computed here, in the truth, rather than by the
    parser. That is what makes the reconciliation check a genuine test: the
    file asserts an arithmetic relationship that the generator established
    independently, so a parser satisfying it has recovered the real numbers
    rather than re-derived them from its own reading.

    Args:
        seed: draws every amount, gap and narrative.
        dialect: affects the date distribution (see `AMBIGUOUS_MMDD`) but not
            the ledger itself — the same seed under two dialects yields the
            same economic facts, which is what the cross-dialect test asserts.
        n_rows: transaction count.
        start: first posting date.
        opening_balance: balance carried into the first row.

    Returns:
        A `StatementTruth` whose `rows` are in posting order.
    """
    # TWO STREAMS, AND THE CROSS-DIALECT TEST DEPENDS ON IT.
    #
    # `rng` draws the economic facts; `date_rng` draws only the posting dates.
    # They must be separate because AMBIGUOUS_MMDD consumes a different number
    # of date draws than the other dialects do. Sharing one stream would let
    # that desynchronization shift every subsequent amount, so "same seed, same
    # ledger, different layout" would silently stop holding — and with it
    # `test_all_dialects_recover_the_same_ledger`, which is the strongest
    # correctness assertion the parser has. This was not hypothetical: the
    # first version of this generator shared a stream and that test caught it.
    rng = np.random.default_rng(seed)
    date_rng = np.random.default_rng(seed + 5_000_003)
    rows: list[tuple[date, str, Decimal, Decimal, Decimal, str]] = []

    balance = _q(opening_balance)
    current = start

    for i in range(n_rows):
        # Gap in days. Statements cluster on weekdays; a 0-day gap produces the
        # same-date rows that break a parser sorting on date alone.
        if dialect is Dialect.AMBIGUOUS_MMDD:
            # Every day is <= 12, so dd/mm and mm/dd BOTH parse every cell and
            # the format cannot be settled cell by cell. The sequence still
            # ascends, as a real statement's does — which is the evidence a
            # correct parser is expected to use instead. Rolling the month over
            # rather than clamping the day keeps that ordering genuine.
            step = int(date_rng.integers(1, 4))
            if current.day + step > 12:
                nxt_m, nxt_y = (current.month % 12) + 1, current.year + (current.month // 12)
                current = date(nxt_y, nxt_m, int(date_rng.integers(1, 4)))
            else:
                current = date(current.year, current.month, current.day + step)
        else:
            current = current + timedelta(days=int(date_rng.integers(0, 4)))

        is_credit = bool(rng.random() < 0.42)
        template, category = (
            _CREDIT_NARRATIVES[int(rng.integers(len(_CREDIT_NARRATIVES)))]
            if is_credit
            else _DEBIT_NARRATIVES[int(rng.integers(len(_DEBIT_NARRATIVES)))]
        )
        desc = template.format(
            vendor=_VENDORS[int(rng.integers(len(_VENDORS)))],
            ref=f"{int(rng.integers(100000, 999999))}",
        )

        # Log-normal magnitudes: real transaction sizes are right-skewed, and a
        # uniform draw would produce a corpus with no large-value rows, which is
        # exactly where thousands-separator parsing fails.
        amount = _q(float(np.exp(rng.normal(10.2, 0.9))))
        debit, credit = (_q(0), amount) if is_credit else (amount, _q(0))

        balance = _q(balance + credit - debit)
        rows.append((current, desc, debit, credit, balance, category))

    return StatementTruth(
        account_reference=f"XXXXXX{int(rng.integers(1000, 9999))}",
        opening_balance=_q(opening_balance),
        closing_balance=balance,
        rows=rows,
        dialect=dialect,
        date_format=_DATE_FORMATS[dialect],
    )


_DATE_FORMATS: dict[Dialect, str] = {
    Dialect.SIMPLE_DEBIT_CREDIT: "%d/%m/%Y",
    Dialect.SIGNED_AMOUNT: "%d-%b-%Y",
    Dialect.AMOUNT_WITH_DR_CR_FLAG: "%d/%m/%Y",
    Dialect.INDIAN_BANK_PREAMBLE: "%Y-%m-%d",
    Dialect.US_NO_BALANCE: "%m/%d/%Y",
    Dialect.AMBIGUOUS_MMDD: "%d/%m/%Y",
}


def _indian_group(value: Decimal) -> str:
    """Format with lakh/crore grouping: 1234567.89 -> '12,34,567.89'.

    Indian grouping is 3 digits then 2s, not 3s throughout. A parser that
    strips commas positionally rather than treating them as separators reads
    this as a different number, which is why the corpus includes it.
    """
    neg = value < 0
    whole, _, frac = f"{abs(value):.2f}".partition(".")
    if len(whole) <= 3:
        grouped = whole
    else:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        grouped = ",".join(parts + [tail])
    return f"{'-' if neg else ''}{grouped}.{frac}"


def _western_group(value: Decimal) -> str:
    return f"{value:,.2f}"


def render(truth: StatementTruth) -> str:
    """Write the truth out as the CSV text a bank would export.

    Every dialect encodes the *same* ledger. Nothing about the economic content
    changes between renderings — only the column names, the direction
    convention, the date format and the number formatting. That is precisely
    what makes cross-dialect equality a valid correctness test.
    """
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    fmt = truth.date_format
    d = truth.dialect

    if d is Dialect.SIMPLE_DEBIT_CREDIT:
        w.writerow(["Txn Date", "Narration", "Cheque/Ref No", "Debit", "Credit", "Balance"])
        for dt, desc, deb, cr, bal, _ in truth.rows:
            w.writerow([
                dt.strftime(fmt), desc, "",
                _western_group(deb) if deb else "",
                _western_group(cr) if cr else "",
                _western_group(bal),
            ])

    elif d is Dialect.SIGNED_AMOUNT:
        w.writerow(["Date", "Description", "Amount", "Running Balance"])
        for dt, desc, deb, cr, bal, _ in truth.rows:
            signed = cr - deb
            w.writerow([dt.strftime(fmt), desc, _western_group(signed), _western_group(bal)])

    elif d is Dialect.AMOUNT_WITH_DR_CR_FLAG:
        w.writerow(["Value Date", "Particulars", "Amount", "Dr/Cr", "Balance"])
        for dt, desc, deb, cr, bal, _ in truth.rows:
            w.writerow([
                dt.strftime(fmt), desc,
                _western_group(cr if cr else deb),
                "CR" if cr else "DR",
                _western_group(bal),
            ])

    elif d is Dialect.INDIAN_BANK_PREAMBLE:
        # Preamble above the header: the header row is not row 1, so a parser
        # assuming it is will read account metadata as column names.
        w.writerow(["Account Statement"])
        w.writerow(["Account Number", truth.account_reference])
        w.writerow(["Currency", truth.currency])
        w.writerow(["Opening Balance", _indian_group(truth.opening_balance)])
        w.writerow([])
        w.writerow(["Date", "Transaction Details", "Withdrawal (Dr)", "Deposit (Cr)", "Closing Balance"])
        for dt, desc, deb, cr, bal, _ in truth.rows:
            w.writerow([
                dt.strftime(fmt), desc,
                _indian_group(deb) if deb else "",
                _indian_group(cr) if cr else "",
                f"{_indian_group(bal)} Cr",
            ])

    elif d is Dialect.US_NO_BALANCE:
        # No balance column: the reconciliation invariant is unavailable and a
        # parser must say so rather than report a pass it did not earn.
        w.writerow(["Posted Date", "Memo", "Amount"])
        for dt, desc, deb, cr, _bal, _ in truth.rows:
            signed = cr - deb
            cell = f"({_western_group(abs(signed))})" if signed < 0 else _western_group(signed)
            w.writerow([dt.strftime(fmt), desc, cell])

    elif d is Dialect.AMBIGUOUS_MMDD:
        w.writerow(["Date", "Details", "Debit", "Credit", "Balance"])
        for dt, desc, deb, cr, bal, _ in truth.rows:
            w.writerow([
                dt.strftime(fmt), desc,
                _western_group(deb) if deb else "",
                _western_group(cr) if cr else "",
                _western_group(bal),
            ])

    else:  # pragma: no cover - the enum is exhaustive above
        raise ValueError(f"unhandled dialect {d!r}")

    return buf.getvalue()


def generate_statement(
    *,
    seed: int,
    dialect: Dialect,
    n_rows: int = 40,
    opening_balance: float = 850_000.0,
) -> tuple[StatementTruth, str]:
    """Truth and rendered CSV text in one call."""
    truth = make_truth(
        seed=seed, dialect=dialect, n_rows=n_rows, opening_balance=opening_balance
    )
    return truth, render(truth)


def generate_corpus(
    *,
    seed: int = 11,
    dialects: Optional[Sequence[Dialect]] = None,
    n_per_dialect: int = 4,
    n_rows: int = 40,
) -> list[tuple[StatementTruth, str]]:
    """One statement per (dialect, index), each with its own derived seed."""
    out: list[tuple[StatementTruth, str]] = []
    for di, dialect in enumerate(dialects or list(Dialect)):
        for i in range(n_per_dialect):
            out.append(
                generate_statement(
                    seed=seed + 1000 * di + i, dialect=dialect, n_rows=n_rows
                )
            )
    return out
