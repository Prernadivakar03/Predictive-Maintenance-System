import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root_endpoint():
    """Verifies GET / returns running status and documentation links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "health" in data

def test_health_endpoint():
    """Verifies GET /health returns 200 OK and model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data

def test_predict_endpoint_normal_sample():
    """Verifies POST /predict correctly predicts normal equipment operation (prediction = 0)."""
    payload = {
        "inputs": [
            {
                "Type": "M",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551.0,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 0.0
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["predictions"]) == 1
    
    pred = data["predictions"][0]
    assert pred["prediction"] == 0
    assert pred["status"] == "Normal Operation"
    assert 0.0 <= pred["failure_probability"] <= 1.0

def test_predict_endpoint_high_strain_sample():
    """Verifies POST /predict correctly accepts high-strain telemetry sample."""
    payload = {
        "inputs": [
            {
                "Type": "L",
                "Air temperature [K]": 298.9,
                "Process temperature [K]": 309.1,
                "Rotational speed [rpm]": 2861.0,
                "Torque [Nm]": 4.6,
                "Tool wear [min]": 240.0
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["predictions"]) == 1

def test_predict_endpoint_invalid_input():
    """Verifies Pydantic schema validation returns HTTP 422 for out-of-bound numerical feature."""
    payload = {
        "inputs": [
            {
                "Type": "M",
                "Air temperature [K]": 500.0,  # Invalid: max allowed is 350.0
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551.0,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 0.0
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
