"""Smoke tests for the dashboard HTTP API (/api/...) and the UI-aware root route."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

READ_ENDPOINTS = [
    "/api/overview", "/api/data/summary", "/api/monitoring/status", "/api/retraining/status",
    "/api/pipeline/status", "/api/jobs", "/api/predictions/recent", "/api/predictions/stats",
    "/api/logs", "/api/system",
]


@pytest.mark.parametrize("path", READ_ENDPOINTS)
def test_read_endpoints_respond(path):
    response = client.get(path)
    assert response.status_code == 200, response.text


def test_model_endpoint_after_training():
    response = client.get("/api/model")
    assert response.status_code in (200, 404)  # 404 only when no model has been trained yet
    if response.status_code == 200:
        assert "test_metrics" in response.json() and "feature_importance" in response.json()


def test_root_returns_json_for_api_clients():
    response = client.get("/")
    assert response.status_code == 200 and response.json()["status"] == "running"


def test_unknown_job_is_404():
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_unknown_figure_is_404():
    assert client.get("/api/figures/not-a-real-figure.png").status_code == 404
    assert client.get("/api/figures/..%2F..%2Fconfig%2Fconfig.yaml").status_code in (404, 422)


def test_monitoring_rejects_unknown_scenario_and_bad_severity():
    assert client.post("/api/monitoring/run", json={"scenario": "nope"}).status_code == 400
    assert client.post("/api/monitoring/run", json={"scenario": "baseline", "severity": 7}).status_code == 422


def test_retraining_rejects_unknown_mode():
    assert client.post("/api/retraining/trigger", json={"mode": "sometimes"}).status_code == 422


def test_predict_rejects_empty_batch():
    assert client.post("/predict", json={"inputs": []}).status_code == 422
