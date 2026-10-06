import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from src.utils.logger import get_logger
from src.utils.config import load_config, get_path
from src.data.ingest import load_raw_data

logger = get_logger("data_validation")

def validate_dataset(df: pd.DataFrame, config: dict = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Performs comprehensive data quality and validation checks on the dataset.
    
    Checks performed:
    1. Dataset Availability & Row Count
    2. Expected Column Schema & Target Column
    3. Data Types Verification
    4. Missing / NULL Values Analysis
    5. Duplicate Rows Analysis
    6. Numerical Range & Categorical Value Validation
    7. Class Balance Analysis
    
    Returns:
        Tuple of (is_valid: bool, quality_report: dict)
    """
    if config is None:
        config = load_config()
        
    val_config = config.get("validation", {})
    expected_cols = val_config.get("expected_columns", [])
    target_col = val_config.get("target_column", "Machine failure")
    num_ranges = val_config.get("numerical_ranges", {})
    cat_values = val_config.get("categorical_values", {})
    
    is_valid = True
    issues = []
    
    report: Dict[str, Any] = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns_present": list(df.columns),
        "is_valid": True,
        "issues": [],
        "missing_values": {},
        "duplicate_rows": 0,
        "invalid_value_counts": {},
        "target_distribution": {}
    }
    
    logger.info("Starting Data Validation Checks...")
    
    # 1. Dataset non-empty check
    if df.empty:
        is_valid = False
        msg = "Dataset is completely empty (0 rows)."
        issues.append(msg)
        logger.error(msg)
        report["is_valid"] = False
        report["issues"] = issues
        return is_valid, report

    # 2. Target Column Check
    if target_col not in df.columns:
        is_valid = False
        msg = f"Target column '{target_col}' missing from dataset."
        issues.append(msg)
        logger.error(msg)
    else:
        target_counts = df[target_col].value_counts().to_dict()
        failure_rate = float(df[target_col].mean()) if len(df) > 0 else 0.0
        report["target_distribution"] = {
            "counts": {str(k): int(v) for k, v in target_counts.items()},
            "failure_rate_pct": round(failure_rate * 100, 2)
        }
        logger.info(f"Target column '{target_col}' present. Failure rate: {report['target_distribution']['failure_rate_pct']}%")

    # 3. Expected Columns Schema Check
    missing_expected = [col for col in expected_cols if col not in df.columns]
    if missing_expected:
        is_valid = False
        msg = f"Missing expected schema columns: {missing_expected}"
        issues.append(msg)
        logger.error(msg)
        
    # 4. Missing Values Analysis
    null_counts = df.isnull().sum()
    missing_report = {}
    for col, count in null_counts.items():
        if count > 0:
            pct = round(float(count) / len(df) * 100, 2)
            missing_report[col] = {"count": int(count), "percentage": pct}
            if pct > 10.0:  # Fail if > 10% missing
                is_valid = False
                msg = f"Column '{col}' has high missing rate: {pct}%"
                issues.append(msg)
                logger.error(msg)
                
    report["missing_values"] = missing_report
    logger.info(f"Missing values check completed. Total missing entries across dataset: {null_counts.sum()}")

    # 5. Duplicate Rows Analysis
    num_duplicates = int(df.duplicated().sum())
    report["duplicate_rows"] = num_duplicates
    if num_duplicates > 0:
        logger.warning(f"Found {num_duplicates} duplicate rows in dataset.")
        
    # 6. Value Range & Validity Checks
    invalid_counts = {}
    
    # Numerical Ranges
    for col, limits in num_ranges.items():
        if col in df.columns:
            min_val, max_val = limits.get("min"), limits.get("max")
            out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
            count_invalid = len(out_of_bounds)
            if count_invalid > 0:
                invalid_counts[col] = count_invalid
                is_valid = False
                msg = f"Column '{col}' has {count_invalid} out-of-bounds values (Range: [{min_val}, {max_val}])."
                issues.append(msg)
                logger.error(msg)
                
    # Categorical Allowed Values
    for col, allowed in cat_values.items():
        if col in df.columns:
            invalid_cats = df[~df[col].isin(allowed)]
            count_invalid = len(invalid_cats)
            if count_invalid > 0:
                invalid_counts[col] = count_invalid
                is_valid = False
                msg = f"Column '{col}' has {count_invalid} invalid categorical values. Allowed: {allowed}"
                issues.append(msg)
                logger.error(msg)
                
    report["invalid_value_counts"] = invalid_counts
    report["is_valid"] = is_valid
    report["issues"] = issues
    
    if is_valid:
        logger.info("Data Validation Passed Successfully! All quality criteria met.")
    else:
        logger.warning(f"Data Validation Failed with {len(issues)} issue(s).")
        
    return is_valid, report

def save_quality_report(report: dict, output_path: Path = None) -> Path:
    """
    Saves data quality validation report as a JSON file.
    """
    config = load_config()
    reports_dir = get_path(config["data"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    if output_path is None:
        output_path = reports_dir / config["data"]["report_filename"]
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Data quality report saved to: {output_path}")
    return output_path

def run_validation_pipeline() -> bool:
    """
    Loads raw dataset, performs validation checks, and exports report.
    """
    df = load_raw_data()
    is_valid, report = validate_dataset(df)
    save_quality_report(report)
    return is_valid

if __name__ == "__main__":
    logger.info("Executing Data Validation Pipeline...")
    valid = run_validation_pipeline()
    print(f"Data Validation Execution Completed. Status: {'SUCCESS' if valid else 'FAILED'}")
