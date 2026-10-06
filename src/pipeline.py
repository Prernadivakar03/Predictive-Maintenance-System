import os
import sys
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config, get_path
from src.data.ingest import download_dataset, load_raw_data
from src.data.validate import validate_dataset, save_quality_report
from src.features.build_features import preprocess_and_save
from src.models.train import train_and_evaluate_all
from src.models.register import get_production_model_info

logger = get_logger("ml_pipeline")

def compute_file_hash(filepath: Path) -> str:
    """Computes SHA256 checksum of a file to check for upstream data changes."""
    if not filepath.exists():
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def run_pipeline(force: bool = False) -> Dict[str, Any]:
    """
    Executes the end-to-end automated machine learning pipeline:
    1. Data Ingestion
    2. Data Validation
    3. Data Preprocessing & Feature Engineering
    4. Model Training & Evaluation
    5. MLflow Tracking & Model Registration
    
    Args:
        force: If True, forces execution of all stages regardless of cached checksums.
        
    Returns:
        Dict containing full pipeline execution metrics and registered model details.
    """
    logger.info("=================================================================")
    logger.info("  STARTING AUTOMATED PREDICTIVE MAINTENANCE MLOPS PIPELINE      ")
    logger.info("=================================================================")
    
    config = load_config()
    summary: Dict[str, Any] = {
        "pipeline_status": "SUCCESS",
        "stages_executed": [],
        "artifacts": {}
    }
    
    # -------------------------------------------------------------------------
    # STAGE 1: DATA INGESTION
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 1: Data Ingestion")
    try:
        raw_csv_path = download_dataset(force=force)
        raw_df = load_raw_data(raw_csv_path)
        
        # Inter-stage validation
        if raw_df is None or raw_df.empty:
            raise ValueError("Data Ingestion failed: Raw dataset is empty or invalid.")
            
        summary["stages_executed"].append("Data Ingestion")
        summary["artifacts"]["raw_data_path"] = str(raw_csv_path)
        summary["artifacts"]["raw_data_rows"] = len(raw_df)
        logger.info(f"[OK] Stage 1 Complete. Raw data shape: {raw_df.shape}")
    except Exception as e:
        logger.error(f"❌ PIPELINE FAILURE at Stage 1 (Data Ingestion): {str(e)}")
        summary["pipeline_status"] = "FAILED"
        summary["failed_stage"] = "Data Ingestion"
        summary["error"] = str(e)
        raise RuntimeError(f"Pipeline failed at Stage 1: {str(e)}") from e

    # -------------------------------------------------------------------------
    # STAGE 2: DATA VALIDATION
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 2: Data Validation")
    try:
        is_valid, quality_report = validate_dataset(raw_df, config=config)
        report_path = save_quality_report(quality_report)
        
        # Inter-stage validation: Halt pipeline if data quality checks fail
        if not is_valid:
            issues_str = "; ".join(quality_report.get("issues", []))
            raise ValueError(f"Data Quality Validation Failed! Issues: {issues_str}")
            
        summary["stages_executed"].append("Data Validation")
        summary["artifacts"]["quality_report_path"] = str(report_path)
        logger.info("[OK] Stage 2 Complete. Data Quality Validation Passed.")
    except Exception as e:
        logger.error(f"❌ PIPELINE FAILURE at Stage 2 (Data Validation): {str(e)}")
        summary["pipeline_status"] = "FAILED"
        summary["failed_stage"] = "Data Validation"
        summary["error"] = str(e)
        raise RuntimeError(f"Pipeline failed at Stage 2: {str(e)}") from e

    # -------------------------------------------------------------------------
    # STAGE 3: PREPROCESSING & FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGE 3: Preprocessing & Feature Engineering")
    try:
        processed_dir = get_path(config["data"]["processed_dir"])
        train_path = processed_dir / "train.joblib"
        preprocessor_path = processed_dir / "preprocessor.joblib"
        
        # Check if cache is valid to avoid redundant reprocessing
        current_hash = compute_file_hash(raw_csv_path)
        hash_file = processed_dir / ".raw_hash"
        cached_hash = ""
        if hash_file.exists():
            cached_hash = hash_file.read_text(encoding="utf-8").strip()
            
        if not force and train_path.exists() and preprocessor_path.exists() and current_hash == cached_hash:
            logger.info("Upstream raw data unchanged and preprocessor exists. Using cached preprocessed artifacts.")
        else:
            logger.info("Executing Preprocessing and Domain Feature Engineering...")
            proc_artifacts = preprocess_and_save(raw_df)
            hash_file.write_text(current_hash, encoding="utf-8")
            
        # Inter-stage validation
        if not (train_path.exists() and preprocessor_path.exists()):
            raise FileNotFoundError("Preprocessing failed: Processed datasets or preprocessor.joblib missing.")
            
        summary["stages_executed"].append("Preprocessing & Feature Engineering")
        summary["artifacts"]["processed_dir"] = str(processed_dir)
        logger.info("[OK] Stage 3 Complete. Feature Engineering & Preprocessing artifacts ready.")
    except Exception as e:
        logger.error(f"❌ PIPELINE FAILURE at Stage 3 (Preprocessing): {str(e)}")
        summary["pipeline_status"] = "FAILED"
        summary["failed_stage"] = "Preprocessing & Feature Engineering"
        summary["error"] = str(e)
        raise RuntimeError(f"Pipeline failed at Stage 3: {str(e)}") from e

    # -------------------------------------------------------------------------
    # STAGES 4 & 5: MODEL TRAINING, EVALUATION & MLFLOW REGISTRATION
    # -------------------------------------------------------------------------
    logger.info("\n>>> STAGES 4 & 5: Model Training, MLflow Tracking & Model Registration")
    try:
        eval_summary = train_and_evaluate_all(target_col=config["validation"]["target_column"])
        
        best_model_path = get_path("models/best_model.joblib")
        if not best_model_path.exists():
            raise FileNotFoundError("Model Training failed: Best model binary artifact missing from models/.")
            
        summary["stages_executed"].append("Model Training & Evaluation")
        summary["stages_executed"].append("MLflow Tracking & Registration")
        summary["artifacts"]["best_model_name"] = eval_summary["selected_model"]
        summary["artifacts"]["mlflow_run_id"] = eval_summary["selected_run_id"]
        summary["artifacts"]["registered_model_version"] = eval_summary["registry_info"]["version"]
        summary["artifacts"]["test_recall"] = eval_summary["selected_model_test_metrics"]["recall"]
        summary["artifacts"]["test_f1"] = eval_summary["selected_model_test_metrics"]["f1_score"]
        
        logger.info(
            f"[OK] Stages 4 & 5 Complete. Best Model '{eval_summary['selected_model']}' "
            f"Registered in MLflow as '{eval_summary['registry_info']['model_name']}' Version {eval_summary['registry_info']['version']} @Production"
        )
    except Exception as e:
        logger.error(f"❌ PIPELINE FAILURE at Stages 4 & 5 (Model Training/Registration): {str(e)}")
        summary["pipeline_status"] = "FAILED"
        summary["failed_stage"] = "Model Training & Registration"
        summary["error"] = str(e)
        raise RuntimeError(f"Pipeline failed at Stages 4 & 5: {str(e)}") from e

    logger.info("\n=================================================================")
    logger.info("  AUTOMATED MLOPS PIPELINE EXECUTED SUCCESSFULLY                ")
    logger.info("=================================================================")
    return summary

def main():
    """CLI Entrypoint for the ML Pipeline with argument parsing and exit code handling."""
    parser = argparse.ArgumentParser(description="Predictive Maintenance Automated MLOps Pipeline")
    parser.add_argument("--force", action="store_true", help="Force re-execution of all pipeline stages")
    args = parser.parse_args()
    
    try:
        summary = run_pipeline(force=args.force)
        print("\n" + json.dumps(summary, indent=2))
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Pipeline Execution Terminated with Error: {str(e)}")
        print(f"\n[PIPELINE ERROR] {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
