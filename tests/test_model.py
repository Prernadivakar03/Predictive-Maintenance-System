import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.evaluate import calculate_classification_metrics
from src.models.train import get_candidate_models, setup_mlflow
from src.models.register import get_production_model_info
from src.data.ingest import generate_synthetic_dataset
from src.features.build_features import create_preprocessing_pipeline

@pytest.fixture
def synthetic_processed_data():
    """Provides synthetic processed features and binary target array for model testing."""
    raw_df = generate_synthetic_dataset(num_samples=200, seed=42)
    X_raw = raw_df.drop(columns=["Machine failure"])
    y = raw_df["Machine failure"].values
    
    pipeline = create_preprocessing_pipeline()
    X_proc = pipeline.fit_transform(X_raw)
    return X_proc, y

def test_calculate_classification_metrics():
    """Verifies calculate_classification_metrics correctly computes expected metric dictionary."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.8, 0.9, 0.95])
    
    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "confusion_matrix" in metrics
    
    # Specific metric verifications
    assert metrics["accuracy"] == 0.75
    assert metrics["recall"] == 0.75  # 3 TP out of 4 actual positives
    assert metrics["confusion_matrix"]["true_positives"] == 3
    assert metrics["confusion_matrix"]["false_negatives"] == 1

def test_get_candidate_models():
    """Verifies candidate model dictionary contains Logistic Regression, Random Forest, and XGBoost."""
    models = get_candidate_models()
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    assert "XGBoost" in models

def test_candidate_model_training_and_predict(synthetic_processed_data):
    """Verifies all candidate models fit on training data and generate prediction probabilities."""
    X, y = synthetic_processed_data
    models = get_candidate_models()
    
    for name, model in models.items():
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (200, 2)
        assert not np.isnan(probs).any()

def test_mlflow_setup_and_registry():
    """Verifies MLflow experiment setup and production model registry query."""
    exp_id = setup_mlflow()
    assert exp_id is not None
    
    info = get_production_model_info("PredictiveMaintenanceModel")
    if info:
        assert info["name"] == "PredictiveMaintenanceModel"
        assert "version" in info
