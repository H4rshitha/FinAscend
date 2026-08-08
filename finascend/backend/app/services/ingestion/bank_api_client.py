"""Pull bank statements over an API, then hand them to the statement parser.

    provider.fetch_page() -> csv text -> statement_parser -> Inflow / Outflow

This is the second ingestion channel beside receipt OCR. Where OCR recovers
data from a document, this recovers it from an account feed, and the two
converge on the same parser and the same domain records.

WHAT IS REAL HERE AND WHAT IS NOT — STATED UP FRONT
---------------------------------------------------
The **client** is real: cursor pagination, client-generated idempotency keys
with replay, a rolling-window rate cap, and retry with exponential backoff and
full jitter that honours `Retry-After`. Those are the parts that are hard to
get right and the parts that decide whether a sync is safe to re-run.

The **counterparty** is not a bank. This project has no Decentro or Plaid
credentials, and inventing a client that cannot be executed would be worse
than useless. So two providers ship:

  * `LocalReferenceProvider` — serves statements from this repo's own
    generator, in-process. Labelled `ProviderKind.LOCAL_REFERENCE` in every
    response so it can never be mistaken for a bank integration.
  * `HttpStatementProvider` — speaks real HTTP to a URL you configure. It is
    exercised in the test suite against a live local server over a real socket,
    which tests the pagination, retry, and rate-cap logic for real. Point it at
    a sandbox and it works; point it at nothing and it fails loudly.

The distinction is carried in the response rather than in a comment, because a
reader looking at a `StatementSyncResult` needs to know which one produced it.

WHY urllib AND NOT requests/httpx
---------------------------------
`urllib.request` is in the standard library, so this adds no dependency. The
things a third-party client is genuinely better at — connection pooling, HTTP/2,
async — buy nothing for a job that makes roughly thirty calls a month. httpx is
present in the environment as a transitive dependency of Starlette's test
client, and depending on a transitive is how a working install breaks later.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Optional, Protocol
from uuid import UUID

from app.schemas.base import SourceType
from app.schemas.core import Inflow, Outflow
from app.schemas.solvency import (
    ProviderKind,
    RateCapStatus,
    StatementSyncResult,
    SyncPage,
)
from app.services.ingestion.statement_generator import Dialect, generate_statement
from app.services.ingestion.statement_parser import (
    StatementParseError,
    parse_statement,
    to_records,
)

# The architecture plan's own feasibility claim. Named here so the number that
# enforces it and the number that was promised are the same object.
DEFAULT_MONTHLY_CALL_CAP = 30
DEFAULT_WINDOW_DAYS = 30

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    """The provider could not be reached, or refused, after retries."""


class RateCapExceeded(RuntimeError):
    """The call budget for the current window is spent."""


@dataclass(frozen=True)
class ProviderPage:
    """One page of statement data as the provider returned it."""

    csv_text: str
    next_cursor: Optional[str]
    http_status: int
    elapsed_ms: float
    retries: int = 0
    n_rows: int = 0


class StatementProvider(Protocol):
    """The whole contract a statement source must satisfy.

    Deliberately narrow. A provider returns a page of CSV and a cursor; it does
    not parse, normalize, or know what an `Inflow` is. That is what lets the
    local reference provider and a real HTTP bank sit behind the same sync
    orchestration without either one being a special case.
    """

    name: str
    kind: ProviderKind

    def fetch_page(
        self,
        *,
        account_reference: str,
        cursor: Optional[str],
        since: date,
        until: date,
        idempotency_key: str,
    ) -> ProviderPage:
        ...


# ---------------------------------------------------------------------------
# Rate cap
# ---------------------------------------------------------------------------

class RateCap:
    """Rolling-window call counter.

    The architecture plan commits to ~30 calls/user/month and notes that a
    commitment needs "an actual counter, not just a plan". This is the counter.

    The window is **rolling** rather than calendar-monthly: a calendar reset
    lets a caller spend the whole budget on the 31st and the whole next budget
    on the 1st, which is twice the intended rate through the moment that
    matters. Timestamps are kept per call so the window slides continuously.

    In-process and therefore per-worker. That is a real limitation and it is
    named rather than glossed: enforcing this across workers needs shared
    state, which is the same Postgres/Redis dependency the rest of the project
    has deliberately not taken on.
    """

    def __init__(
        self,
        *,
        calls_allowed: int = DEFAULT_MONTHLY_CALL_CAP,
        window_days: int = DEFAULT_WINDOW_DAYS,
    ) -> None:
        self.calls_allowed = calls_allowed
        self.window_days = window_days
        self._calls: dict[str, list[datetime]] = {}

    def _prune(self, key: str, now: datetime) -> list[datetime]:
        cutoff = now - timedelta(days=self.window_days)
        kept = [t for t in self._calls.get(key, []) if t > cutoff]
        self._calls[key] = kept
        return kept

    def status(self, key: str, *, now: Optional[datetime] = None) -> RateCapStatus:
        now = now or datetime.now(timezone.utc)
        used = self._prune(key, now)
        oldest = min(used) if used else now
        return RateCapStatus(
            window_days=self.window_days,
            calls_used=len(used),
            calls_allowed=self.calls_allowed,
            calls_remaining=max(0, self.calls_allowed - len(used)),
            window_resets_on=(oldest + timedelta(days=self.window_days)).date(),
            exhausted=len(used) >= self.calls_allowed,
        )

    def consume(self, key: str, *, now: Optional[datetime] = None) -> None:
        """Record one call, or refuse when the budget is spent."""
        now = now or datetime.now(timezone.utc)
        used = self._prune(key, now)
        if len(used) >= self.calls_allowed:
            raise RateCapExceeded(
                f"rate cap reached for {key!r}: {len(used)} calls in the last "
                f"{self.window_days} days, cap is {self.calls_allowed}"
            )
        self._calls.setdefault(key, []).append(now)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def make_idempotency_key(
    *, provider: str, account_reference: str, since: date, until: date, business_id: UUID
) -> str:
    """Derive a stable key from what the request actually asks for.

    Derived rather than random, deliberately. A random key makes every retry a
    new request, which is precisely the double-post the mechanism exists to
    prevent — a client that times out and retries would pull the same window
    twice under two keys and post it twice. Hashing the request parameters
    means a retry of the *same* window is recognised as the same request, and a
    genuinely different window gets a genuinely different key.
    """
    raw = f"{provider}|{business_id}|{account_reference}|{since.isoformat()}|{until.isoformat()}"
    return sha256(raw.encode()).hexdigest()[:32]


@dataclass
class _CacheEntry:
    result: StatementSyncResult
    inflows: list[Inflow]
    outflows: list[Outflow]


class IdempotencyCache:
    """Remembers what a key already returned, so a retry replays it.

    In-process, like the rate cap, and with the same caveat: this makes a retry
    safe within one worker's lifetime, not across a restart. The deterministic
    record ids in `statement_parser._record_id` are the second line of defence
    and the one that survives a restart — re-ingesting the same rows produces
    the same ids, so a duplicate post is detectable downstream even if the
    cache is gone.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Optional[_CacheEntry]:
        return self._entries.get(key)

    def put(
        self, key: str, result: StatementSyncResult,
        inflows: list[Inflow], outflows: list[Outflow],
    ) -> None:
        self._entries[key] = _CacheEntry(result, inflows, outflows)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class LocalReferenceProvider:
    """Serves generated statements in-process, paginated.

    Exists so the whole pipeline is runnable and testable with no credentials
    and no network. It is labelled `LOCAL_REFERENCE` everywhere it appears; it
    is not a bank and no response implies otherwise.
    """

    name = "local_reference"
    kind = ProviderKind.LOCAL_REFERENCE

    def __init__(
        self,
        *,
        seed: int = 11,
        dialect: Dialect = Dialect.SIMPLE_DEBIT_CREDIT,
        n_rows: int = 60,
        page_size: int = 25,
    ) -> None:
        self.seed = seed
        self.dialect = dialect
        self.n_rows = n_rows
        self.page_size = page_size

    def fetch_page(
        self, *, account_reference: str, cursor: Optional[str],
        since: date, until: date, idempotency_key: str,
    ) -> ProviderPage:
        t0 = time.perf_counter()
        offset = int(cursor) if cursor else 0
        truth, text = generate_statement(
            seed=self.seed, dialect=self.dialect, n_rows=self.n_rows
        )
        lines = text.splitlines()
        # The header block is whatever precedes the first data row; it is
        # repeated on every page so each page is independently parseable. A
        # page that only makes sense concatenated to its predecessors would
        # make a partial sync unusable.
        n_header = len(lines) - len(truth.rows)
        head, data = lines[:n_header], lines[n_header:]

        page = data[offset:offset + self.page_size]
        nxt = offset + self.page_size
        elapsed = (time.perf_counter() - t0) * 1000.0
        return ProviderPage(
            csv_text="\n".join(head + page) + "\n",
            next_cursor=str(nxt) if nxt < len(data) else None,
            http_status=200,
            elapsed_ms=elapsed,
            n_rows=len(page),
        )


class HttpStatementProvider:
    """Talks real HTTP to a statement endpoint.

    Expects a JSON response shaped `{"csv": "...", "next_cursor": "..." | null}`.
    A CSV payload rather than JSON rows is deliberate: it keeps the *one*
    statement parser on the critical path for both channels, so the column-
    inference and reconciliation work is exercised by API pulls too. Accepting
    pre-structured JSON rows would route around the reconciliation check and
    quietly make API-sourced data less validated than file-sourced data.

    RETRY POLICY
    ------------
    Exponential backoff with **full jitter** (`sleep ~ U(0, base * 2^n)`) on
    retryable statuses. Full jitter rather than fixed backoff because every
    client retrying on the same schedule reconverges on the server at the same
    instant — the thundering herd that turns one 503 into a sustained outage.
    A `Retry-After` header, when present, overrides the computed delay: the
    server knows better than the client's guess.
    """

    name = "http_open_banking"
    kind = ProviderKind.HTTP_OPEN_BANKING

    def __init__(
        self,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self._rng = rng or random.Random()

    def _build_request(
        self, account_reference: str, cursor: Optional[str],
        since: date, until: date, idempotency_key: str,
    ) -> urllib.request.Request:
        params = {
            "account": account_reference,
            "since": since.isoformat(),
            "until": until.isoformat(),
        }
        if cursor:
            params["cursor"] = cursor
        url = f"{self.base_url}/statements?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/json")
        # Sent on every page, not just the first: a retry can land on any page,
        # and a key that only covers page 1 leaves the rest unprotected.
        req.add_header("Idempotency-Key", idempotency_key)
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        return req

    def fetch_page(
        self, *, account_reference: str, cursor: Optional[str],
        since: date, until: date, idempotency_key: str,
    ) -> ProviderPage:
        req = self._build_request(account_reference, cursor, since, until, idempotency_key)
        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    payload = json.loads(body)
                    csv_text = payload.get("csv") or ""
                    if not csv_text.strip():
                        raise ProviderError(
                            "provider returned an empty CSV payload; treating as an "
                            "error rather than as an empty statement, because the two "
                            "are indistinguishable here and only one is safe to assume"
                        )
                    return ProviderPage(
                        csv_text=csv_text,
                        next_cursor=payload.get("next_cursor"),
                        http_status=resp.status,
                        elapsed_ms=elapsed,
                        retries=attempt,
                        n_rows=max(0, len(csv_text.strip().splitlines()) - 1),
                    )
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}"
                if exc.code not in RETRYABLE_STATUS or attempt >= self.max_retries:
                    raise ProviderError(
                        f"provider returned HTTP {exc.code} after {attempt} retries"
                    ) from exc
                self._sleep_for_retry(attempt, exc.headers.get("Retry-After"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_retries:
                    raise ProviderError(
                        f"provider unreachable after {attempt} retries: {last_error}"
                    ) from exc
                self._sleep_for_retry(attempt, None)

        raise ProviderError(f"exhausted retries: {last_error}")

    def _sleep_for_retry(self, attempt: int, retry_after: Optional[str]) -> None:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), 60.0))
                return
            except (TypeError, ValueError):
                pass
        ceiling = self.backoff_base_seconds * (2 ** attempt)
        time.sleep(self._rng.uniform(0.0, ceiling))


# ---------------------------------------------------------------------------
# Sync orchestration
# ---------------------------------------------------------------------------

@dataclass
class SyncEngine:
    """Owns the rate cap and the idempotency cache across syncs.

    These are per-engine rather than per-call because both are stateful by
    nature — a cap that resets every call is not a cap, and a replay cache with
    no lifetime never replays anything.
    """

    rate_cap: RateCap = field(default_factory=RateCap)
    cache: IdempotencyCache = field(default_factory=IdempotencyCache)

    def sync(
        self,
        provider: StatementProvider,
        *,
        business_id: UUID,
        account_reference: str,
        since: date,
        until: date,
        max_pages: int = 20,
        source_type: Optional[SourceType] = None,
    ) -> tuple[StatementSyncResult, list[Inflow], list[Outflow]]:
        """Pull a date window, parse it, and return domain records.

        Pages are fetched until the provider stops returning a cursor, then the
        accumulated CSV is parsed **as one statement**. Parsing per page would
        break the running-balance check across page boundaries and lose the
        strongest validation the statement carries.

        Args:
            provider: any `StatementProvider`.
            business_id: owning business.
            account_reference: the account to pull.
            since / until: the window. Also the idempotency key's input, so the
                same window is never double-posted.
            max_pages: hard stop against a provider that never clears its
                cursor. A pagination bug on the server should cost a bounded
                number of calls, not the whole monthly budget.
            source_type: provenance stamped on the records. Defaults to the
                one matching the provider kind.

        Returns:
            (result, inflows, outflows). On a rejected parse the records are
            empty and `result.parse.rejection_reason` says why.

        Raises:
            RateCapExceeded: budget spent. Raised before any call is made.
            ProviderError: the provider failed after retries.
        """
        key = make_idempotency_key(
            provider=provider.name, account_reference=account_reference,
            since=since, until=until, business_id=business_id,
        )

        cached = self.cache.get(key)
        if cached is not None:
            replayed = cached.result.model_copy(update={"replayed_from_cache": True})
            return replayed, cached.inflows, cached.outflows

        cap_key = f"{provider.name}:{business_id}"
        pages: list[SyncPage] = []
        chunks: list[str] = []
        cursor: Optional[str] = None
        warnings: list[str] = []

        for page_no in range(1, max_pages + 1):
            # Consumed BEFORE the call, so a failed call still costs budget.
            # Counting only successes would let a caller retry a failing
            # endpoint without limit, which is the behaviour a cap exists to
            # prevent.
            self.rate_cap.consume(cap_key)
            page = provider.fetch_page(
                account_reference=account_reference, cursor=cursor,
                since=since, until=until, idempotency_key=key,
            )
            pages.append(
                SyncPage(
                    page_number=page_no, cursor=cursor, n_rows=page.n_rows,
                    http_status=page.http_status, elapsed_ms=page.elapsed_ms,
                    retries=page.retries,
                )
            )
            chunks.append(page.csv_text)
            cursor = page.next_cursor
            if cursor is None:
                break
        else:
            warnings.append(
                f"stopped at the {max_pages}-page ceiling with a cursor still "
                "outstanding; the window is INCOMPLETE and should be re-pulled in "
                "smaller date ranges"
            )

        combined = _concatenate_pages(chunks)

        parse = None
        inflows: list[Inflow] = []
        outflows: list[Outflow] = []
        try:
            parse = parse_statement(combined)
            if parse.rejected:
                warnings.append(
                    "parse rejected; no records were created. See "
                    "parse.rejection_reason."
                )
            else:
                inflows, outflows = to_records(
                    parse, business_id=business_id,
                    account_reference=account_reference,
                    source_type=source_type or _default_source(provider.kind),
                )
        except StatementParseError as exc:
            warnings.append(f"statement could not be parsed at all: {exc}")

        result = StatementSyncResult(
            provider=provider.name,
            provider_kind=provider.kind,
            account_reference=account_reference,
            idempotency_key=key,
            replayed_from_cache=False,
            pages=pages,
            rate_cap=self.rate_cap.status(cap_key),
            parse=parse,
            n_inflows=len(inflows),
            n_outflows=len(outflows),
            warnings=warnings,
        )
        self.cache.put(key, result, inflows, outflows)
        return result, inflows, outflows


def _concatenate_pages(chunks: list[str]) -> str:
    """Join pages into one statement, keeping only the first page's header.

    Each page repeats the header so it is independently parseable; concatenated
    verbatim, those repeats become data rows in the middle of the file and every
    one of them breaks the reconciliation. Dropping them is what lets the
    running-balance identity run across the whole window, which is where it has
    the most to say.
    """
    if not chunks:
        return ""
    first = chunks[0].rstrip("\n").splitlines()
    out = list(first)
    for chunk in chunks[1:]:
        lines = chunk.rstrip("\n").splitlines()
        # Drop the leading lines this page shares verbatim with the first page.
        i = 0
        while i < len(lines) and i < len(first) and lines[i] == first[i]:
            i += 1
        out.extend(lines[i:])
    return "\n".join(out) + "\n"


def _default_source(kind: ProviderKind) -> SourceType:
    """Provenance that matches what actually served the data."""
    # LOCAL_REFERENCE maps to BANK_STATEMENT rather than to one of the vendor
    # source types: the data really is a bank statement, and stamping it
    # DECENTRO_API would put a claim in the ledger that no Decentro call backs.
    return SourceType.BANK_STATEMENT
