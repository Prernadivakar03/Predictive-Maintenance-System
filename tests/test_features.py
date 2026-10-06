import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.features.build_features import (
    PredictiveMaintenanceFeatureEngineer,
    create_preprocessing_pipeline,
    split_data,
    preprocess_and_save,
    load_preprocessor
)
from src.data.ingest import generate_synthetic_dataset

@pytest.fixture
def sample_raw_df():
    """Provides a synthetic raw telemetry DataFrame for feature engineering testing."""
    return generate_synthetic_dataset(num_samples=200, seed=42)

def test_feature_engineer_transformer(sample_raw_df):
    """Verifies that domain feature engineering transformer correctly calculates engineered columns."""
    fe = PredictiveMaintenanceFeatureEngineer()
    df_transformed = fe.transform(sample_raw_df)
    
    assert "temp_difference" in df_transformed.columns
    assert "power_kw" in df_transformed.columns
    assert "tool_wear_torque" in df_transformed.columns
    assert "speed_torque_ratio" in df_transformed.columns
    assert "temp_ratio" in df_transformed.columns
    
    # Check math correctness for row 0
    expected_temp_diff = sample_raw_df.loc[0, "Process temperature [K]"] - sample_raw_df.loc[0, "Air temperature [K]"]
    assert np.isclose(df_transformed.loc[0, "temp_difference"], expected_temp_diff)

def test_preprocessing_pipeline_fit_transform(sample_raw_df):
    """Verifies scikit-learn preprocessing pipeline outputs expected 13 features with no NaNs."""
    pipeline = create_preprocessing_pipeline()
    X_raw = sample_raw_df.drop(columns=["Machine failure"])
    
    X_processed = pipeline.fit_transform(X_raw)
    
    assert X_processed.shape[0] == 200
    assert X_processed.shape[1] == 13  # 10 numeric + 3 one-hot columns (Type_H, Type_L, Type_M)
    assert not np.isnan(X_processed).any()

def test_stratified_split(sample_raw_df):
    """Verifies train, val, test splits preserve row ratios and target balance."""
    df_train, df_val, df_test = split_data(sample_raw_df, target_col="Machine failure")
    
    assert len(df_train) == 140
    assert len(df_val) == 30
    assert len(df_test) == 30
    
    # Check failure rate balance across splits
    train_rate = df_train["Machine failure"].mean()
    val_rate = df_val["Machine failure"].mean()
    test_rate = df_test["Machine failure"].mean()
    
    assert np.isclose(train_rate, val_rate, atol=0.05)
    assert np.isclose(train_rate, test_rate, atol=0.05)

def test_load_preprocessor_inference(tmp_path, sample_raw_df):
    """Verifies fitted preprocessor can be saved, loaded, and used for single-sample inference."""
    import joblib
    pipeline = create_preprocessing_pipeline()
    X_raw = sample_raw_df.drop(columns=["Machine failure"])
    pipeline.fit(X_raw)
    
    save_file = tmp_path / "preprocessor.joblib"
    joblib.dump(pipeline, save_file)
    
    loaded_pipeline = joblib.load(save_file)
    single_sample = sample_raw_df.iloc[[0]].drop(columns=["Machine failure"])
    transformed_sample = loaded_pipeline.transform(single_sample)
    
    assert transformed_sample.shape == (1, 13)
    assert not np.isnan(transformed_sample).any()
