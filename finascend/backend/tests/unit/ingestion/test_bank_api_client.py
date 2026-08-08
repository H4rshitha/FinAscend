"""Tests for the API ingestion channel.

The HTTP tests run against a **real** `http.server` on a real socket in a
background thread — not a mocked transport. That is deliberate: the things
worth testing here are URL construction, header propagation, cursor
pagination, status handling and retry, and a mock that returns whatever it was
told to return would exercise the test's assumptions rather than the client's
behaviour. A local socket costs milliseconds and tests the actual code path.

The rate cap and idempotency tests use the local provider, since neither
depends on transport.
"""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from uuid import UUID

import pytest

from app.schemas.base import SourceType
from app.schemas.solvency import ProviderKind
from app.services.ingestion.bank_api_client import (
    DEFAULT_MONTHLY_CALL_CAP,
    HttpStatementProvider,
    IdempotencyCache,
    LocalReferenceProvider,
    ProviderError,
    RateCap,
    RateCapExceeded,
    SyncEngine,
    make_idempotency_key,
)
from app.services.ingestion.statement_generator import Dialect, generate_statement

BUSINESS = UUID("00000000-0000-0000-0000-0000000000de")
OTHER_BUSINESS = UUID("00000000-0000-0000-0000-0000000000ef")
SINCE, UNTIL = date(2026, 1, 1), date(2026, 3, 1)


# ---------------------------------------------------------------------------
# A real local statement server
# ---------------------------------------------------------------------------

class _State:
    """Mutable knobs the handler reads, so a test can shape the server."""

    def __init__(self) -> None:
        self.csv_text = ""
        self.page_size = 25
        self.n_header = 1
        self.fail_times = 0          # return 503 this many times, then succeed
        self.retry_after: str | None = None
        self.requests: list[dict] = []


def _make_server(state: _State) -> tuple[HTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):      # keep pytest output clean
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            state.requests.append(
                {
                    "path": parsed.path,
                    "query": {k: v[0] for k, v in qs.items()},
                    "idempotency_key": self.headers.get("Idempotency-Key"),
                    "authorization": self.headers.get("Authorization"),
                }
            )

            if state.fail_times > 0:
                state.fail_times -= 1
                self.send_response(503)
                if state.retry_after:
                    self.send_header("Retry-After", state.retry_after)
                self.end_headers()
                self.wfile.write(b"unavailable")
                return

            lines = state.csv_text.rstrip("\n").splitlines()
            head, data = lines[:state.n_header], lines[state.n_header:]
            offset = int(qs.get("cursor", ["0"])[0])
            page = data[offset:offset + state.page_size]
            nxt = offset + state.page_size

            body = json.dumps(
                {
                    "csv": "\n".join(head + page) + "\n",
                    "next_cursor": str(nxt) if nxt < len(data) else None,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


@pytest.fixture
def statement_server():
    state = _State()
    truth, text = generate_statement(
        seed=5, dialect=Dialect.SIMPLE_DEBIT_CREDIT, n_rows=60
    )
    state.csv_text = text
    state.n_header = len(text.rstrip("\n").splitlines()) - len(truth.rows)
    server, base_url = _make_server(state)
    try:
        yield state, base_url, truth
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# HTTP provider
# ---------------------------------------------------------------------------

def test_http_sync_pulls_every_page_and_reconciles_the_whole_window(statement_server):
    """Pagination must not break the running-balance check at page boundaries.

    The identity is evaluated across the assembled window, so a client that
    parsed page by page — or that concatenated repeated headers as data — would
    fail this even though every individual page looks fine.
    """
    state, base_url, truth = statement_server
    provider = HttpStatementProvider(base_url, api_key="test-key")
    engine = SyncEngine()

    result, inflows, outflows = engine.sync(
        provider, business_id=BUSINESS, account_reference="ACC1",
        since=SINCE, until=UNTIL,
    )

    assert result.provider_kind == ProviderKind.HTTP_OPEN_BANKING.value
    assert len(result.pages) == 3                     # 60 rows / page_size 25
    assert result.parse is not None
    assert not result.parse.rejected
    assert len(result.parse.rows) == len(truth.rows)
    assert result.parse.reconciliation.checkable
    assert result.parse.reconciliation.passed
    assert result.parse.total_debits == truth.total_debits
    assert result.parse.total_credits == truth.total_credits
    assert len(inflows) + len(outflows) == len(truth.rows)


def test_request_carries_window_cursor_auth_and_idempotency_key(statement_server):
    state, base_url, _truth = statement_server
    provider = HttpStatementProvider(base_url, api_key="test-key")
    SyncEngine().sync(
        provider, business_id=BUSINESS, account_reference="ACC1",
        since=SINCE, until=UNTIL,
    )

    first = state.requests[0]
    assert first["query"]["account"] == "ACC1"
    assert first["query"]["since"] == SINCE.isoformat()
    assert first["query"]["until"] == UNTIL.isoformat()
    assert "cursor" not in first["query"]
    assert first["authorization"] == "Bearer test-key"

    # The key must be on EVERY page, not just the first: a retry can land on
    # any page, and a key covering only page 1 leaves the rest unprotected.
    keys = {r["idempotency_key"] for r in state.requests}
    assert len(keys) == 1 and None not in keys
    assert state.requests[1]["query"]["cursor"] == "25"


def test_retryable_status_is_retried_and_then_succeeds(statement_server):
    state, base_url, truth = statement_server
    state.fail_times = 2
    provider = HttpStatementProvider(
        base_url, max_retries=3, backoff_base_seconds=0.001
    )

    result, _ins, _outs = SyncEngine().sync(
        provider, business_id=BUSINESS, account_reference="ACC1",
        since=SINCE, until=UNTIL,
    )
    assert result.pages[0].retries == 2
    assert not result.parse.rejected


def test_retry_after_header_is_honoured(statement_server):
    state, base_url, _truth = statement_server
    state.fail_times = 1
    state.retry_after = "0"          # server says "immediately"
    provider = HttpStatementProvider(
        base_url, max_retries=2, backoff_base_seconds=30.0
    )
    # If the header were ignored the backoff ceiling of 30s would make this
    # test hang; completing quickly is the assertion.
    result, _i, _o = SyncEngine().sync(
        provider, business_id=BUSINESS, account_reference="ACC1",
        since=SINCE, until=UNTIL,
    )
    assert result.pages[0].retries == 1


def test_exhausted_retries_raise_rather_than_returning_empty(statement_server):
    state, base_url, _truth = statement_server
    state.fail_times = 99
    provider = HttpStatementProvider(
        base_url, max_retries=2, backoff_base_seconds=0.001
    )
    with pytest.raises(ProviderError, match="HTTP 503"):
        SyncEngine().sync(
            provider, business_id=BUSINESS, account_reference="ACC1",
            since=SINCE, until=UNTIL,
        )


def test_unreachable_host_raises_provider_error():
    provider = HttpStatementProvider(
        "http://127.0.0.1:1", max_retries=1, backoff_base_seconds=0.001,
        timeout_seconds=0.5,
    )
    with pytest.raises(ProviderError):
        SyncEngine().sync(
            provider, business_id=BUSINESS, account_reference="ACC1",
            since=SINCE, until=UNTIL,
        )


# ---------------------------------------------------------------------------
# Local reference provider
# ---------------------------------------------------------------------------

def test_local_provider_is_labelled_as_a_reference_not_a_bank():
    """The label is the honesty mechanism and must survive into the response."""
    result, _i, _o = SyncEngine().sync(
        LocalReferenceProvider(seed=3, n_rows=50),
        business_id=BUSINESS, account_reference="ACC1",
        since=SINCE, until=UNTIL,
    )
    assert result.provider_kind == ProviderKind.LOCAL_REFERENCE.value
    assert result.provider == "local_reference"


@pytest.mark.parametrize(
    "dialect",
    [Dialect.SIMPLE_DEBIT_CREDIT, Dialect.INDIAN_BANK_PREAMBLE,
     Dialect.SIGNED_AMOUNT, Dialect.AMOUNT_WITH_DR_CR_FLAG],
)
def test_paginated_local_pull_reconciles_across_page_boundaries(dialect):
    result, ins, outs = SyncEngine().sync(
        LocalReferenceProvider(seed=8, dialect=dialect, n_rows=60, page_size=17),
        business_id=BUSINESS, account_reference="ACC1",
        since=SINCE, until=UNTIL,
    )
    assert len(result.pages) == 4
    assert result.parse.reconciliation.passed
    assert len(result.parse.rows) == 60
    assert len(ins) + len(outs) == 60
    assert all(r.source_type == SourceType.BANK_STATEMENT.value for r in ins + outs)


def test_page_ceiling_marks_the_window_incomplete_rather_than_silently_truncating():
    result, _i, _o = SyncEngine().sync(
        LocalReferenceProvider(seed=8, n_rows=200, page_size=10),
        business_id=BUSINESS, account_reference="ACC1",
        since=SINCE, until=UNTIL, max_pages=3,
    )
    assert len(result.pages) == 3
    assert any("INCOMPLETE" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_same_window_replays_from_cache_and_costs_no_budget():
    engine = SyncEngine()
    provider = LocalReferenceProvider(seed=3, n_rows=50, page_size=25)
    kwargs = dict(
        business_id=BUSINESS, account_reference="ACC1", since=SINCE, until=UNTIL
    )

    first, in1, out1 = engine.sync(provider, **kwargs)
    second, in2, out2 = engine.sync(provider, **kwargs)

    assert first.replayed_from_cache is False
    assert second.replayed_from_cache is True
    assert second.rate_cap.calls_used == first.rate_cap.calls_used
    assert [i.id for i in in1] == [i.id for i in in2]
    assert [o.id for o in out1] == [o.id for o in out2]


def test_a_different_window_is_a_different_request():
    engine = SyncEngine()
    provider = LocalReferenceProvider(seed=3, n_rows=50, page_size=25)
    engine.sync(provider, business_id=BUSINESS, account_reference="ACC1",
                since=SINCE, until=UNTIL)
    second, _i, _o = engine.sync(
        provider, business_id=BUSINESS, account_reference="ACC1",
        since=SINCE, until=UNTIL + timedelta(days=1),
    )
    assert second.replayed_from_cache is False


def test_idempotency_key_is_derived_not_random():
    """A random key would make every retry a new request — the double-post the
    mechanism exists to prevent."""
    args = dict(provider="p", account_reference="ACC1", since=SINCE,
                until=UNTIL, business_id=BUSINESS)
    assert make_idempotency_key(**args) == make_idempotency_key(**args)
    assert make_idempotency_key(**{**args, "business_id": OTHER_BUSINESS}) != \
        make_idempotency_key(**args)
    assert make_idempotency_key(**{**args, "account_reference": "ACC2"}) != \
        make_idempotency_key(**args)


# ---------------------------------------------------------------------------
# Rate cap
# ---------------------------------------------------------------------------

def test_rate_cap_blocks_once_the_window_budget_is_spent():
    cap = RateCap(calls_allowed=3, window_days=30)
    for _ in range(3):
        cap.consume("k")
    assert cap.status("k").exhausted
    assert cap.status("k").calls_remaining == 0
    with pytest.raises(RateCapExceeded):
        cap.consume("k")


def test_rate_cap_window_is_rolling_not_calendar():
    """A calendar reset would allow a full budget on the 31st and another on
    the 1st — twice the intended rate through the moment that matters."""
    cap = RateCap(calls_allowed=2, window_days=30)
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    cap.consume("k", now=now - timedelta(days=31))     # outside the window
    cap.consume("k", now=now - timedelta(days=2))
    status = cap.status("k", now=now)
    assert status.calls_used == 1, "the 31-day-old call should have aged out"
    cap.consume("k", now=now)
    with pytest.raises(RateCapExceeded):
        cap.consume("k", now=now)


def test_budget_is_charged_per_page_and_per_business():
    engine = SyncEngine(rate_cap=RateCap(calls_allowed=10, window_days=30))
    provider = LocalReferenceProvider(seed=3, n_rows=60, page_size=25)

    r1, _i, _o = engine.sync(provider, business_id=BUSINESS,
                             account_reference="ACC1", since=SINCE, until=UNTIL)
    assert r1.rate_cap.calls_used == 3          # one per page

    r2, _i, _o = engine.sync(provider, business_id=OTHER_BUSINESS,
                             account_reference="ACC1", since=SINCE, until=UNTIL)
    # Separate business, separate budget — one tenant cannot spend another's.
    assert r2.rate_cap.calls_used == 3


def test_failed_calls_still_consume_budget(statement_server):
    """Counting only successes would let a caller retry a broken endpoint
    without limit, which is exactly what a cap exists to prevent."""
    state, base_url, _truth = statement_server
    state.fail_times = 99
    engine = SyncEngine(rate_cap=RateCap(calls_allowed=5, window_days=30))
    provider = HttpStatementProvider(
        base_url, max_retries=1, backoff_base_seconds=0.001
    )
    with pytest.raises(ProviderError):
        engine.sync(provider, business_id=BUSINESS, account_reference="ACC1",
                    since=SINCE, until=UNTIL)
    assert engine.rate_cap.status(f"{provider.name}:{BUSINESS}").calls_used == 1


def test_default_cap_matches_the_documented_commitment():
    """The architecture plan promises ~30 calls/user/month. Same number, one
    place, so the promise and the enforcement cannot drift apart."""
    assert DEFAULT_MONTHLY_CALL_CAP == 30
    assert RateCap().status("k").calls_allowed == 30
