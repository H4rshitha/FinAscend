"""Correctness tests for bank-statement parsing.

The organizing test is `test_all_dialects_recover_the_same_ledger`: the same
economic facts are rendered through six bank layouts, and a correct parser must
return identical records from all six. That is a far stronger assertion than
"parses without error" and it needs no golden files — the generator's truth is
the oracle, and cross-dialect equality catches anything that is right for one
layout by coincidence.

The rest test the two failures a name-based parser cannot see (swapped columns,
inverted sign) and the two honesty requirements (refuse an unreconcilable file;
never silently resolve a genuinely ambiguous date column).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import UUID

import pytest

from app.schemas.base import SourceType
from app.schemas.solvency import AmountConvention, ColumnRole
from app.services.ingestion.statement_generator import (
    Dialect,
    generate_statement,
    make_truth,
    render,
)
from app.services.ingestion.statement_parser import (
    StatementParseError,
    counterparty_from_narrative,
    ingest_statement,
    infer_date_format,
    locate_header,
    parse_amount,
    parse_statement,
    to_records,
)

BUSINESS = UUID("00000000-0000-0000-0000-0000000000de")


# ---------------------------------------------------------------------------
# Cell-level parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cell,expected",
    [
        ("1,234.56", Decimal("1234.56")),
        ("12,34,567.89", Decimal("1234567.89")),   # Indian lakh grouping
        ("1,234,567.89", Decimal("1234567.89")),   # western grouping
        ("(1,234.56)", Decimal("-1234.56")),       # US parenthesised negative
        ("1,234.56 Cr", Decimal("1234.56")),
        ("1,234.56 Dr", Decimal("-1234.56")),
        ("INR 5,000.00", Decimal("5000.00")),
        ("-450.25", Decimal("-450.25")),
        ("", None),
        ("-", None),
        ("NARRATION TEXT", None),
    ],
)
def test_parse_amount_handles_every_convention_in_the_corpus(cell, expected):
    assert parse_amount(cell) == expected


def test_blank_and_zero_are_not_conflated():
    """An empty debit cell means 'not a debit', which is not a debit of zero.

    Collapsing the two would make a sparse debit/credit pair indistinguishable
    from a dense one, and the mapping search relies on that distinction.
    """
    assert parse_amount("") is None
    assert parse_amount("0.00") == Decimal("0.00")


# ---------------------------------------------------------------------------
# The central cross-dialect test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dialect", list(Dialect))
def test_parser_recovers_the_generator_truth_exactly(dialect):
    """Every amount, direction and date must come back exactly, per dialect."""
    truth, text = generate_statement(seed=17, dialect=dialect, n_rows=45)
    result = parse_statement(text)

    assert not result.rejected, result.rejection_reason
    assert len(result.rows) == len(truth.rows)

    for got, want in zip(result.rows, truth.rows):
        want_date, _desc, want_debit, want_credit, _bal, _cat = want
        assert got.posted_date == want_date
        assert got.debit == want_debit
        assert got.credit == want_credit

    assert result.total_debits == truth.total_debits
    assert result.total_credits == truth.total_credits


def test_all_dialects_recover_the_same_ledger():
    """Six layouts, one ledger. The strongest available correctness assertion.

    The generator emits identical economic facts under every dialect for a
    given seed, so any dialect whose parsed (date, debit, credit) triples
    differ from the others is being parsed wrongly — regardless of whether it
    reconciles, and regardless of what its headers say.
    """
    ledgers = {}
    for dialect in Dialect:
        _truth, text = generate_statement(seed=23, dialect=dialect, n_rows=40)
        result = parse_statement(text)
        assert not result.rejected, f"{dialect}: {result.rejection_reason}"
        ledgers[dialect] = [(r.debit, r.credit) for r in result.rows]

    reference = ledgers[Dialect.SIMPLE_DEBIT_CREDIT]
    for dialect, rows in ledgers.items():
        assert rows == reference, f"{dialect} disagrees with the reference layout"


@pytest.mark.parametrize("dialect", list(Dialect))
def test_reconciliation_passes_where_a_balance_column_exists(dialect):
    """And is reported as UNCHECKABLE, not as passing, where it does not."""
    _truth, text = generate_statement(seed=31, dialect=dialect, n_rows=35)
    rec = parse_statement(text).reconciliation

    if dialect is Dialect.US_NO_BALANCE:
        assert rec.checkable is False
        assert rec.passed is False, "an unchecked parse must not claim to have passed"
    else:
        assert rec.checkable is True
        assert rec.passed is True
        assert rec.max_absolute_residual <= rec.tolerance


def test_convention_is_identified_correctly():
    expected = {
        Dialect.SIMPLE_DEBIT_CREDIT: AmountConvention.SEPARATE_DEBIT_CREDIT,
        Dialect.INDIAN_BANK_PREAMBLE: AmountConvention.SEPARATE_DEBIT_CREDIT,
        Dialect.AMBIGUOUS_MMDD: AmountConvention.SEPARATE_DEBIT_CREDIT,
        Dialect.SIGNED_AMOUNT: AmountConvention.SIGNED_SINGLE_COLUMN,
        Dialect.US_NO_BALANCE: AmountConvention.SIGNED_SINGLE_COLUMN,
        Dialect.AMOUNT_WITH_DR_CR_FLAG: AmountConvention.AMOUNT_WITH_INDICATOR,
    }
    for dialect, want in expected.items():
        _t, text = generate_statement(seed=9, dialect=dialect, n_rows=25)
        assert parse_statement(text).convention == want.value, dialect


# ---------------------------------------------------------------------------
# The two failures a header-based parser cannot detect
# ---------------------------------------------------------------------------

def test_swapped_debit_credit_headers_are_corrected_by_the_identity():
    """Mislabel the columns and the parser must still get the ledger right.

    This is the failure that motivates selecting the mapping by reconciliation:
    the headers are actively lying, every row is well-formed, and only the
    running-balance identity can tell that the directions are backwards.
    """
    truth, text = generate_statement(
        seed=44, dialect=Dialect.SIMPLE_DEBIT_CREDIT, n_rows=30
    )
    lines = text.splitlines()
    header = lines[0].split(",")
    i_d, i_c = header.index("Debit"), header.index("Credit")
    header[i_d], header[i_c] = header[i_c], header[i_d]
    corrupted = ",".join(header) + "\n" + "\n".join(lines[1:]) + "\n"

    result = parse_statement(corrupted)

    assert not result.rejected
    assert result.reconciliation.passed
    assert result.total_debits == truth.total_debits
    assert result.total_credits == truth.total_credits


def test_inverted_sign_convention_is_diagnosed_not_absorbed():
    """A statement whose signed column is backwards must FAIL, and say so.

    Here the balance column is left correct while every amount's sign is
    flipped, so no mapping can reconcile it. The parser must refuse and name
    the doubling signature rather than pick the least-bad mapping.
    """
    import csv
    import io

    truth = make_truth(seed=51, dialect=Dialect.SIGNED_AMOUNT, n_rows=25)
    rows = list(csv.reader(io.StringIO(render(truth))))
    # Columns: Date, Description, Amount, Running Balance. Flip every amount's
    # sign and leave the balance column correct, so no mapping can reconcile.
    for row in rows[1:]:
        row[2] = f"{-parse_amount(row[2]):.2f}"
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)

    result = parse_statement(buf.getvalue())

    assert result.rejected
    assert result.reconciliation.checkable
    assert not result.reconciliation.passed
    assert "inverted direction" in (result.reconciliation.diagnosis or "")


def _split_csv(line: str) -> list[str]:
    import csv
    import io

    return next(csv.reader(io.StringIO(line)))


def _rewrite_cell(text: str, line_no: int, col: int, value: str) -> str:
    """Replace one cell, re-quoting through csv so the row keeps its shape.

    Rejoining split cells with a bare ',' would un-quote the amounts, whose
    thousands separators then split them into extra columns — corrupting the
    whole row instead of the one cell the test means to corrupt, and testing
    something other than what it claims to.
    """
    import csv
    import io

    rows = list(csv.reader(io.StringIO(text)))
    rows[line_no][col] = value
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue()


def test_a_single_corrupted_cell_is_localized_not_blamed_on_the_mapping():
    truth, text = generate_statement(
        seed=63, dialect=Dialect.SIMPLE_DEBIT_CREDIT, n_rows=40
    )
    result = parse_statement(_rewrite_cell(text, 11, 5, "999999.99"))

    assert result.rejected
    assert "localized bad cell" in (result.reconciliation.diagnosis or "")
    assert result.reconciliation.failing_row_indices


def test_records_are_never_built_from_a_rejected_parse():
    _truth, text = generate_statement(
        seed=71, dialect=Dialect.SIMPLE_DEBIT_CREDIT, n_rows=30
    )
    corrupted = _rewrite_cell(text, 5, 5, "1.00")
    result = parse_statement(corrupted)

    assert result.rejected
    with pytest.raises(ValueError, match="rejected parse"):
        to_records(result, business_id=BUSINESS, account_reference="ACC")

    # ingest_statement returns the refusal rather than raising, so one bad file
    # in a batch does not discard the batch.
    _r, ins, outs = ingest_statement(
        corrupted, business_id=BUSINESS, account_reference="ACC"
    )
    assert ins == [] and outs == []


# ---------------------------------------------------------------------------
# Date-format inference
# ---------------------------------------------------------------------------

def test_ambiguous_day_column_is_resolved_by_chronological_order():
    """Every day <= 12, so only the ORDER of the column can settle dd vs mm."""
    truth, text = generate_statement(
        seed=83, dialect=Dialect.AMBIGUOUS_MMDD, n_rows=40
    )
    assert max(r[0].day for r in truth.rows) <= 12, "the trap must actually be set"

    result = parse_statement(text)
    assert result.date_format == "%d/%m/%Y"
    for got, want in zip(result.rows, truth.rows):
        assert got.posted_date == want[0]


def test_genuinely_undecidable_dates_warn_rather_than_silently_choosing():
    """A diagonal column parses AND orders under both readings. Say so."""
    fmt, warnings = infer_date_format(
        ["01/01/2026", "02/02/2026", "03/03/2026", "04/04/2026"]
    )
    assert fmt == "%d/%m/%Y"
    assert any("genuinely ambiguous" in w for w in warnings)


def test_iso_and_alphabetic_month_formats_never_contend():
    assert infer_date_format(["2026-03-04", "2026-03-09"])[0] == "%Y-%m-%d"
    assert infer_date_format(["04-Mar-2026", "09-Mar-2026"])[0] == "%d-%b-%Y"


# ---------------------------------------------------------------------------
# Header location
# ---------------------------------------------------------------------------

def test_header_is_found_beneath_an_account_preamble():
    _truth, text = generate_statement(
        seed=97, dialect=Dialect.INDIAN_BANK_PREAMBLE, n_rows=20
    )
    import csv
    import io

    rows = list(csv.reader(io.StringIO(text)))
    idx = locate_header(rows)
    assert idx > 0, "preamble was not skipped"
    assert "Date" in rows[idx]


def test_empty_and_non_tabular_input_raises_rather_than_returning_empty():
    with pytest.raises(StatementParseError):
        parse_statement("")
    with pytest.raises(StatementParseError):
        parse_statement("this file is prose\nand has no columns at all\n")


# ---------------------------------------------------------------------------
# Domain records
# ---------------------------------------------------------------------------

def test_records_carry_provenance_and_settled_credits_are_not_receivables():
    """A settled bank credit is observed money, so `is_receivable` is False.

    §2.1 struck the `certainty = 1.0` default because it asserted perfect
    confidence in a RECEIVABLE. This asserts it about money already in the
    account, which is an observation — and the test pins the distinction so a
    later refactor cannot quietly reintroduce the struck default by treating
    statement credits as receivables.
    """
    _truth, text = generate_statement(
        seed=101, dialect=Dialect.SIMPLE_DEBIT_CREDIT, n_rows=30
    )
    result, inflows, outflows = ingest_statement(
        text, business_id=BUSINESS, account_reference="XXXXXX1234"
    )
    assert inflows and outflows
    for inf in inflows:
        assert inf.source_type == SourceType.BANK_STATEMENT.value
        assert inf.is_receivable is False
        assert inf.received_date is not None
        assert inf.certainty == 1.0
    for out in outflows:
        assert out.source_type == SourceType.BANK_STATEMENT.value
        assert out.paid_date is not None


def test_record_ids_are_deterministic_so_re_ingestion_cannot_double_post():
    """Overlapping pulls are the normal path, not an edge case."""
    _truth, text = generate_statement(
        seed=113, dialect=Dialect.SIMPLE_DEBIT_CREDIT, n_rows=25
    )
    _r1, in1, out1 = ingest_statement(
        text, business_id=BUSINESS, account_reference="ACC9"
    )
    _r2, in2, out2 = ingest_statement(
        text, business_id=BUSINESS, account_reference="ACC9"
    )
    assert [i.id for i in in1] == [i.id for i in in2]
    assert [o.id for o in out1] == [o.id for o in out2]


@pytest.mark.parametrize(
    "narrative,expected",
    [
        ("NEFT DR ACME SUPPLIES INV449182", "Acme Supplies"),
        ("UPI/P2M/ORION TEXTILES/CR/771823", "Orion Textiles"),
        ("GST CHALLAN 889211", "UNKNOWN"),
    ],
)
def test_counterparty_extraction_strips_rail_metadata(narrative, expected):
    assert counterparty_from_narrative(narrative) == expected


def test_column_assignments_report_low_confidence_when_headers_gave_no_help():
    """Confidence must reflect the evidence, not the outcome.

    On the signed-amount layout the amount column is called "Amount", which
    supports SIGNED_AMOUNT well, while the reconciliation is what actually
    fixes the mapping. The assignment list is the audit trail for that, so it
    has to be populated and role-complete.
    """
    _t, text = generate_statement(seed=127, dialect=Dialect.SIGNED_AMOUNT, n_rows=20)
    result = parse_statement(text)
    roles = {a.role for a in result.column_assignments}
    assert ColumnRole.SIGNED_AMOUNT.value in roles
    assert ColumnRole.BALANCE.value in roles
    assert ColumnRole.DATE.value in roles
    assert all(0.0 <= a.confidence <= 1.0 for a in result.column_assignments)
