import pytest
import pandas as pd
import numpy as np
from src.monitoring.drift_detector import (
    detect_data_drift,
    detect_prediction_drift,
    detect_performance_degradation,
    run_monitoring_suite
)
from src.data.ingest import generate_synthetic_dataset

@pytest.fixture
def reference_and_current_dfs():
    """Provides synthetic reference and current datasets for monitoring tests."""
    ref_df = generate_synthetic_dataset(num_samples=500, seed=42)
    curr_df = generate_synthetic_dataset(num_samples=200, seed=42)
    return ref_df, curr_df

def test_detect_data_drift_no_drift(reference_and_current_dfs):
    """Verifies that datasets drawn from exact same distribution show no data drift."""
    ref_df, _ = reference_and_current_dfs
    curr_df = ref_df.copy()
    drift_res = detect_data_drift(ref_df, curr_df, alpha=0.05, max_drift_ratio=0.30)
    
    assert drift_res["data_drift_detected"] is False
    assert drift_res["drifted_features_count"] == 0

def test_detect_data_drift_with_severe_shift(reference_and_current_dfs):
    """Verifies that artificial feature shift (e.g. +50K temperature rise) correctly triggers data drift alert."""
    ref_df, curr_df = reference_and_current_dfs
    shifted_df = curr_df.copy()
    
    # Introduce artificial thermal sensor drift across multiple columns
    shifted_df["Air temperature [K]"] += 50.0
    shifted_df["Process temperature [K]"] += 50.0
    shifted_df["Torque [Nm]"] += 30.0
    shifted_df["Rotational speed [rpm]"] += 1000.0
    
    drift_res = detect_data_drift(ref_df, shifted_df, alpha=0.05, max_drift_ratio=0.30)
    
    assert drift_res["data_drift_detected"] is True
    assert drift_res["drifted_features_count"] >= 4
    assert "Air temperature [K]" in drift_res["drifted_features"]

def test_detect_performance_degradation_flag():
    """Verifies performance degradation flag triggers when Recall drops below threshold."""
    # Dummy mock model
    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)  # Predicts all 0s (0% recall)
        def predict_proba(self, X):
            return np.zeros((len(X), 2))
            
    dummy_model = DummyModel()
    
    # Create dataset with positive failure labels
    df_labels = pd.DataFrame({
        "Air temperature [K]": [300.0] * 10,
        "Machine failure": [1] * 10
    })
    
    perf_res = detect_performance_degradation(
        model=dummy_model,
        curr_df=df_labels,
        target_col="Machine failure",
        min_recall=0.75
    )
    
    assert perf_res["ground_truth_available"] is True
    assert perf_res["performance_degraded"] is True
    assert perf_res["metrics"]["recall"] == 0.0

def test_run_monitoring_suite_execution():
    """Verifies full monitoring suite runs cleanly and exports report summary."""
    report = run_monitoring_suite()
    assert "summary" in report
    summary = report["summary"]
    assert "data_drift_detected" in summary
    assert "retraining_required" in summary
