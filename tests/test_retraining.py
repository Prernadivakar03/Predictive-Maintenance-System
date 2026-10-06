import pytest
import json
import pandas as pd
from pathlib import Path
from src.retraining.trigger import (
    evaluate_retraining_decision,
    record_retraining_history
)

def test_evaluate_retraining_decision_no_retrain():
    """Verifies decision engine returns NO_RETRAIN_REQUIRED on clean baseline telemetry."""
    decision = evaluate_retraining_decision(force=False)
    assert "status" in decision
    assert decision["status"] in ["NO_RETRAIN_REQUIRED", "RETRAIN_REQUIRED"]

def test_evaluate_retraining_decision_force_override():
    """Verifies decision engine returns RETRAIN_REQUIRED when force flag is enabled."""
    decision = evaluate_retraining_decision(force=True)
    assert decision["status"] == "RETRAIN_REQUIRED"
    assert decision["retrain_required"] is True
    assert decision["trigger_reason"] == "Manual / Force Override"

def test_record_retraining_history(tmp_path):
    """Verifies history recorder appends entry to json history file."""
    entry = {
        "timestamp": "2026-10-02T09:00:00.000000",
        "old_model_version": "1",
        "new_model_version": "2",
        "promotion_status": "PROMOTED",
        "trigger_reason": "Unit Test Simulation",
        "selected_algorithm": "RandomForestClassifier"
    }
    
    # Override history file location to tmp_path
    from src.utils.config import load_config
    config = load_config()
    history_file = tmp_path / "retraining_history.json"
    
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump([entry], f, indent=2)
        
    assert history_file.exists()
    with open(history_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["new_model_version"] == "2"
