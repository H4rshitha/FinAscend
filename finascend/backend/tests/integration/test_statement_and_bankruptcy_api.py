"""Integration: the statement and insolvency endpoints, over the real app.

These go through the actual FastAPI stack — routing, JWT auth, validation,
serialization — rather than calling the services directly, because that is
where the wiring bugs live. A service can be perfectly correct and still be
unreachable, unserializable, or unauthenticated.

The database is a throwaway SQLite file created per session. The env var must
be set *before* `app.db.session` is imported, since the engine is built at
module import, which is why this module does its imports inside fixtures.
"""

from __future__ import annotations

import importlib
import os
import tempfile
import uuid

import pytest


@pytest.fixture(scope="module")
def client():
    tmp = tempfile.mkdtemp().replace("\\", "/")
    os.environ["FINASCEND_DATABASE_URL"] = f"sqlite+pysqlite:///{tmp}/test.db"

    importlib.import_module("app.models")
    from fastapi.testclient import TestClient

    from app.db.session import engine
    from app.main import app as fastapi_app
    from app.models.base import Base

    Base.metadata.create_all(engine)
    with TestClient(fastapi_app) as c:
        yield c


@pytest.fixture(scope="module")
def auth(client) -> dict[str, str]:
    resp = client.post(
        "/api/v1/auth/signup",
        json={
            "full_name": "Test Owner",
            "email": f"owner-{uuid.uuid4().hex[:8]}@example.com",
            "password": "Correct-Horse-Battery-9",
            "company_name": "Test Co",
            "company_size": "small",
        },
    )
    assert resp.status_code == 201, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


# ---------------------------------------------------------------------------
# Auth is actually enforced
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/api/v1/ingestion/providers"),
        ("GET", "/api/v1/ingestion/statements/dialects"),
        ("GET", "/api/v1/ingestion/statements/sample"),
        ("POST", "/api/v1/ingestion/statements/sync"),
        ("GET", "/api/v1/risk/bankruptcy"),
        ("GET", "/api/v1/risk/bankruptcy/calibration"),
    ],
)
def test_new_endpoints_require_authentication(client, method, path):
    assert client.request(method, path).status_code == 401


# ---------------------------------------------------------------------------
# Statement ingestion
# ---------------------------------------------------------------------------

def test_providers_endpoint_states_what_is_and_is_not_a_real_bank(client, auth):
    body = client.get("/api/v1/ingestion/providers", headers=auth).json()
    by_name = {p["name"]: p for p in body["providers"]}

    assert by_name["local_reference"]["kind"] == "local_reference"
    assert by_name["local_reference"]["available"] is True
    assert "NOT a bank" in by_name["local_reference"]["description"]
    # The absent integration must be listed as absent rather than omitted.
    assert by_name["decentro / plaid"]["available"] is False
    assert body["rate_cap_default"]["calls_allowed"] == 30


@pytest.mark.parametrize(
    "dialect",
    ["simple_debit_credit", "signed_amount", "amount_with_dr_cr_flag",
     "indian_bank_preamble", "us_no_balance", "ambiguous_mmdd"],
)
def test_sample_then_parse_round_trip_recovers_the_totals(client, auth, dialect):
    """Fetch a generated statement, post it back, and score against its truth."""
    sample = client.get(
        f"/api/v1/ingestion/statements/sample?dialect={dialect}", headers=auth
    ).json()

    resp = client.post(
        "/api/v1/ingestion/statements/parse?account_reference=ACC1",
        headers=auth,
        files={"file": ("statement.csv", sample["csv"].encode(), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["rejected"] is False
    assert body["totals"]["n_rows"] == sample["truth"]["n_rows"]
    assert body["totals"]["total_debits"] == pytest.approx(
        sample["truth"]["total_debits"]
    )
    assert body["totals"]["total_credits"] == pytest.approx(
        sample["truth"]["total_credits"]
    )
    assert body["mapping"]["date_format"] == sample["truth"]["date_format"]

    # The one dialect with no balance column must report unchecked, not passed.
    rec = body["reconciliation"]
    if dialect == "us_no_balance":
        assert rec["checkable"] is False and rec["passed"] is False
    else:
        assert rec["checkable"] is True and rec["passed"] is True


def test_an_unreconcilable_upload_is_refused_with_a_diagnosis(client, auth):
    import csv
    import io

    sample = client.get("/api/v1/ingestion/statements/sample", headers=auth).json()
    rows = list(csv.reader(io.StringIO(sample["csv"])))
    rows[7][5] = "1.00"                      # corrupt one balance cell
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)

    body = client.post(
        "/api/v1/ingestion/statements/parse",
        headers=auth,
        files={"file": ("bad.csv", buf.getvalue().encode(), "text/csv")},
    ).json()

    assert body["rejected"] is True
    assert body["rejection_reason"]
    assert body["reconciliation"]["diagnosis"]
    assert body["records"]["n_inflows"] == 0
    assert body["records"]["n_outflows"] == 0


def test_non_csv_upload_is_a_400_not_a_500(client, auth):
    resp = client.post(
        "/api/v1/ingestion/statements/parse",
        headers=auth,
        files={"file": ("notes.txt", b"just some prose with no columns\n", "text/plain")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error_code"] == "unparseable_statement"


def test_empty_upload_is_rejected(client, auth):
    resp = client.post(
        "/api/v1/ingestion/statements/parse",
        headers=auth,
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error_code"] == "empty_upload"


def test_api_sync_paginates_reconciles_and_labels_the_provider(client, auth):
    resp = client.post(
        "/api/v1/ingestion/statements/sync?n_rows=60&page_size=25",
        headers=auth,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["provider_kind"] == "local_reference", (
        "the reference provider must never be presented as a bank integration"
    )
    assert len(body["pages"]) == 3
    assert body["parse"]["reconciliation"]["passed"] is True
    assert body["records"]["n_inflows"] + body["records"]["n_outflows"] == 60
    assert body["rate_cap"]["calls_allowed"] == 30
    assert body["idempotency_key"]


def test_repeating_a_sync_replays_rather_than_re_pulling(client, auth):
    url = "/api/v1/ingestion/statements/sync?n_rows=40&page_size=20"
    first = client.post(url, headers=auth).json()
    second = client.post(url, headers=auth).json()

    assert second["idempotency_key"] == first["idempotency_key"]
    assert second["replayed_from_cache"] is True
    assert second["rate_cap"]["calls_used"] == first["rate_cap"]["calls_used"]


def test_unreachable_provider_url_surfaces_as_502(client, auth):
    resp = client.post(
        "/api/v1/ingestion/statements/sync?base_url=http://127.0.0.1:1",
        headers=auth,
    )
    assert resp.status_code == 502
    assert resp.json()["detail"]["error_code"] == "provider_unavailable"


# ---------------------------------------------------------------------------
# Bankruptcy risk
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bankruptcy(client, auth):
    resp = client.get(
        "/api/v1/risk/bankruptcy?include_calibration=true", headers=auth
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_ruin_curve_is_served_with_its_uncertainty(bankruptcy):
    curve = bankruptcy["ruin_curve"]
    assert len(curve) >= 4
    probs = [p["ruin_probability"] for p in curve]
    assert all(a <= b + 1e-12 for a, b in zip(probs, probs[1:]))
    for point in curve:
        assert 0.0 <= point["ci_lower"] <= point["ruin_probability"] <= point["ci_upper"] <= 1.0
    assert bankruptcy["headline_ruin_probability"] == probs[-1]


def test_hazard_is_served_beside_the_cumulative_curve(bankruptcy):
    hazard = bankruptcy["hazard"]
    assert hazard
    at_risk = [h["n_at_risk"] for h in hazard]
    assert all(a >= b for a, b in zip(at_risk, at_risk[1:]))


def test_z_score_is_absent_when_no_balance_sheet_was_supplied(bankruptcy):
    assert bankruptcy["altman"] is None, "absent, not defaulted"
    assert bankruptcy["agreement"] is None


def test_supplying_a_balance_sheet_adds_the_z_score_and_an_agreement_verdict(
    client, auth
):
    resp = client.get(
        "/api/v1/risk/bankruptcy",
        params={
            "total_assets": 5_000_000,
            "total_liabilities": 500_000,
            "current_assets": 900_000,
            "current_liabilities": 200_000,
            "retained_earnings": 1_200_000,
            "ebit": 700_000,
            "include_calibration": "false",
        },
        headers=auth,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["altman"] is not None
    assert body["altman"]["zone"] == "safe"
    assert body["agreement"]
    # Every input must carry where it came from.
    assert all(i["provenance"] for i in body["altman"]["inputs"])
    assert "NOT fitted or validated" in " ".join(body["caveats"])


def test_calibration_is_attached_and_reports_real_skill(bankruptcy):
    cal = bankruptcy["calibration"]
    assert cal is not None
    assert cal["brier_skill_score"] > 0.25
    assert cal["roc_auc"] > 0.85
    assert cal["n_businesses"] > 0


def test_calibration_endpoint_explains_how_to_read_itself(client, auth):
    body = client.get("/api/v1/risk/bankruptcy/calibration", headers=auth).json()
    assert body["brier_skill_score"] > 0.25
    assert body["roc_auc"] > 0.85
    assert "how_to_read" in body
    assert body["buckets"]


def test_bankruptcy_risk_is_not_confused_with_counterparty_default(bankruptcy):
    """The two are different quantities and the response must say so."""
    joined = " ".join(bankruptcy["caveats"])
    assert "counterparty default" in joined.lower()
