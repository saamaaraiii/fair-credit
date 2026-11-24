import pytest
from fastapi.testclient import TestClient

from faircredit.api import app

EXAMPLE = app.openapi()["components"]["schemas"]["Applicant"]["example"]


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("model", ["scorecard", "gbm", "scorecard_reweighed"])
def test_score_returns_decision_and_reasons(client, model):
    r = client.post(f"/v1/score?model={model}", json=EXAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 < body["probability_of_default"] < 1
    assert body["decision"] in {"approve", "decline"}
    assert (body["decision"] == "approve") == (body["probability_of_default"] < body["threshold"])
    if model == "gbm":
        assert body["credit_score"] is None
    else:
        assert 300 < body["credit_score"] < 900
    assert all(r["impact"] > 0 for r in body["reasons"])


def test_risky_applicant_is_declined_with_reasons(client):
    risky = dict(
        EXAMPLE,
        checking_status="< 0",
        duration_months=60,
        savings="< 100",
        credit_history="critical account",
        credit_amount=15000,
        installment_rate=4,
    )
    safe = dict(
        EXAMPLE,
        checking_status="no account",
        duration_months=12,
        savings=">= 1000",
        credit_history="critical account",
        credit_amount=1500,
    )
    r_risky = client.post("/v1/score", json=risky).json()
    r_safe = client.post("/v1/score", json=safe).json()
    assert r_risky["probability_of_default"] > r_safe["probability_of_default"]
    assert r_risky["decision"] == "decline"
    assert {x["feature"] for x in r_risky["reasons"]} & {"checking_status", "duration_months"}


def test_validation_rejects_bad_input(client):
    assert client.post("/v1/score", json=dict(EXAMPLE, purpose="yacht")).status_code == 422
    assert client.post("/v1/score", json=dict(EXAMPLE, duration_months=-3)).status_code == 422


def test_models_endpoint(client):
    body = client.get("/v1/models").json()
    assert set(body) == {"scorecard", "gbm", "scorecard_reweighed"}
    assert all(0.6 < v["auc"] < 1 for v in body.values())
