import pytest
import pandas as pd
from unittest.mock import patch
from src.pipeline import run_pipeline

def test_run_pipeline_success():
    """Verifies end-to-end execution of the automated MLOps pipeline."""
    summary = run_pipeline(force=False)
    
    assert summary["pipeline_status"] == "SUCCESS"
    assert "Data Ingestion" in summary["stages_executed"]
    assert "Data Validation" in summary["stages_executed"]
    assert "Preprocessing & Feature Engineering" in summary["stages_executed"]
    assert "Model Training & Evaluation" in summary["stages_executed"]
    assert "MLflow Tracking & Registration" in summary["stages_executed"]
    assert "best_model_name" in summary["artifacts"]
    assert "registered_model_version" in summary["artifacts"]

def test_pipeline_validation_failure_handling():
    """Verifies that an upstream Data Validation failure immediately halts pipeline execution."""
    # Mock validate_dataset to return is_valid=False
    with patch("src.pipeline.validate_dataset") as mock_val:
        mock_val.return_value = (False, {"issues": ["Simulated schema corruption test"]})
        
        with pytest.raises(RuntimeError) as exc_info:
            run_pipeline(force=False)
            
        assert "Pipeline failed at Stage 2" in str(exc_info.value)
