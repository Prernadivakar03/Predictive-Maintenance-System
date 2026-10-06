import pytest
import pandas as pd
from fastapi.testclient import TestClient
from src.data.ingest import download_dataset, load_raw_data
from src.data.validate import validate_dataset
from src.features.build_features import preprocess_and_save
from src.models.train import train_and_evaluate_all
from src.models.register import get_production_model_info
from src.monitoring.drift_detector import run_monitoring_suite
from src.retraining.trigger import evaluate_retraining_decision
from src.pipeline import run_pipeline
from src.api.main import app

client = TestClient(app)

def test_full_system_e2e_smoke():
    """
    End-to-End System Smoke Test:
    Executes and verifies the complete MLOps lifecycle:
    Ingestion -> Validation -> Preprocessing -> Model Training -> MLflow Registry ->
    FastAPI Serving -> Monitoring Audit -> Retraining Decision.
    """
    # 1. Master Pipeline Execution
    pipeline_summary = run_pipeline(force=False)
    assert pipeline_summary["pipeline_status"] == "SUCCESS"
    assert len(pipeline_summary["stages_executed"]) == 5
    
    # 2. MLflow Production Model Verification
    prod_info = get_production_model_info("PredictiveMaintenanceModel")
    assert prod_info is not None
    assert prod_info["name"] == "PredictiveMaintenanceModel"
    assert "version" in prod_info
    
    # 3. FastAPI Health Endpoint Verification
    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "healthy"
    
    # 4. FastAPI Real-time Inference Verification
    sample_payload = {
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
    predict_response = client.post("/predict", json=sample_payload)
    assert predict_response.status_code == 200
    pred_data = predict_response.json()
    assert pred_data["success"] is True
    assert len(pred_data["predictions"]) == 1
    assert pred_data["predictions"][0]["prediction"] in [0, 1]
    
    # 5. Monitoring & Drift Detection Audit Verification
    monitoring_report = run_monitoring_suite()
    assert "summary" in monitoring_report
    assert "data_drift_detected" in monitoring_report["summary"]
    
    # 6. Retraining Decision Engine Verification
    decision = evaluate_retraining_decision(force=False)
    assert "status" in decision
    assert decision["status"] in ["NO_RETRAIN_REQUIRED", "RETRAIN_REQUIRED"]
