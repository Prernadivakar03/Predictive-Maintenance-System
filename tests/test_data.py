import os
import json
import pytest
import pandas as pd
from pathlib import Path
from src.data.ingest import download_dataset, load_raw_data, generate_synthetic_dataset
from src.data.validate import validate_dataset, save_quality_report

@pytest.fixture
def sample_valid_df():
    """Fixture providing a valid synthetic sample dataframe matching AI4I 2020 schema."""
    return generate_synthetic_dataset(num_samples=100, seed=42)

def test_ingest_generate_synthetic(sample_valid_df):
    """Verifies synthetic dataset generator creates expected columns and rows."""
    assert isinstance(sample_valid_df, pd.DataFrame)
    assert len(sample_valid_df) == 100
    assert "Machine failure" in sample_valid_df.columns
    assert "Air temperature [K]" in sample_valid_df.columns

def test_validate_dataset_success(sample_valid_df):
    """Verifies validation succeeds on clean sample dataset."""
    is_valid, report = validate_dataset(sample_valid_df)
    assert is_valid is True
    assert report["is_valid"] is True
    assert len(report["issues"]) == 0
    assert report["total_rows"] == 100

def test_validate_dataset_missing_target(sample_valid_df):
    """Verifies validation fails when target column is missing."""
    df_missing_target = sample_valid_df.drop(columns=["Machine failure"])
    is_valid, report = validate_dataset(df_missing_target)
    assert is_valid is False
    assert any("Target column 'Machine failure' missing" in issue for issue in report["issues"])

def test_validate_dataset_missing_columns(sample_valid_df):
    """Verifies validation fails when expected columns are missing."""
    df_missing_col = sample_valid_df.drop(columns=["Torque [Nm]"])
    is_valid, report = validate_dataset(df_missing_col)
    assert is_valid is False
    assert any("Torque [Nm]" in issue for issue in report["issues"])

def test_validate_dataset_invalid_numerical_range(sample_valid_df):
    """Verifies validation flags out-of-bound numerical values."""
    df_invalid = sample_valid_df.copy()
    df_invalid.loc[0, "Air temperature [K]"] = 500.0  # Max allowed is 350.0
    is_valid, report = validate_dataset(df_invalid)
    assert is_valid is False
    assert "Air temperature [K]" in report["invalid_value_counts"]

def test_save_quality_report(sample_valid_df, tmp_path):
    """Verifies data quality report is properly written to JSON file."""
    _, report = validate_dataset(sample_valid_df)
    report_file = tmp_path / "test_report.json"
    saved_path = save_quality_report(report, output_path=report_file)
    
    assert saved_path.exists()
    with open(saved_path, "r", encoding="utf-8") as f:
        loaded_report = json.load(f)
    assert loaded_report["total_rows"] == 100
    assert loaded_report["is_valid"] is True
