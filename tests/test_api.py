"""API tests: does the service behave correctly?

CONCEPT - testing the contract: these tests call the API exactly as the web app
does (through FastAPI's TestClient, no server needed) and check both the happy
path and that invalid input is REJECTED. Rejecting bad input is as important
as accepting good input.

The api_client fixture (tests/conftest.py) serves a small model trained on
synthetic data, so these tests never depend on a downloaded dataset.
"""

from src.config import APP_VERSION
from src.schema import EXAMPLE_INPUT


def test_health_reports_versions(api_client):
    body = api_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["app_version"] == APP_VERSION
    assert body["model_version"] == "test-1"


def test_predict_returns_probability(api_client):
    body = api_client.post("/predict", json=EXAMPLE_INPUT).json()
    assert 0 <= body["probability"] <= 1
    assert body["risk_level"] in {"low", "moderate", "high"}
    assert "not a medical" in body["disclaimer"]


def test_rejects_impossible_bmi(api_client):
    assert api_client.post("/predict", json={**EXAMPLE_INPUT, "BMI": 500}).status_code == 422


def test_rejects_missing_field(api_client):
    payload = {k: v for k, v in EXAMPLE_INPUT.items() if k != "Age"}
    assert api_client.post("/predict", json=payload).status_code == 422


def test_predictions_are_logged(api_client, tmp_path):
    from src.prediction_log import load_recent_inputs

    api_client.post("/predict", json=EXAMPLE_INPUT)
    assert len(load_recent_inputs(tmp_path / "predictions.db", limit=10)) == 1
