import os
import sys
import json
import shutil
import tempfile
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import load_config, get_path
from src.monitoring.drift_detector import run_monitoring_suite
from src.models.train import train_and_evaluate_all
from src.models.register import get_production_model_info, get_mlflow_client
from src.retraining.promotion import evaluate_promotion

logger = get_logger("retraining_engine")

MODEL_NAME = "PredictiveMaintenanceModel"

# Files that together define what the API actually serves. They are overwritten by every training run
# and therefore must be restored if the candidate model is rejected by the promotion gate.
_SERVING_ARTIFACTS = ("best_model.joblib", "evaluation_metrics.json", "model_metadata.json")


def decide_from_summary(summary: Dict[str, Any], new_samples: int = 0, min_new_samples: int = 500) -> Dict[str, Any]:
    """
    Pure decision logic: turns a monitoring summary (+ count of new labeled samples) into a retraining decision.
    Used by both the retraining engine and the read-only dashboard preview.
    """
    data_drift = summary.get("data_drift_detected", False)
    pred_drift = summary.get("prediction_drift_detected", False)
    perf_degraded = summary.get("performance_degraded", False)
    new_data_threshold_met = bool(new_samples >= min_new_samples)

    trigger_reasons = []
    if data_drift:
        trigger_reasons.append(f"Significant Feature Data Drift ({summary.get('drifted_features_count')} features)")
    if pred_drift:
        trigger_reasons.append(f"Prediction Probability Shift (Wasserstein: {summary.get('wasserstein_distance')})")
    if perf_degraded:
        trigger_reasons.append(f"Model Performance Degraded (Recall: {summary.get('current_recall')})")
    if new_data_threshold_met:
        trigger_reasons.append(f"Accumulated {new_samples} New Labeled Samples (Threshold: {min_new_samples})")

    retrain_required = bool(trigger_reasons)
    return {
        "status": "RETRAIN_REQUIRED" if retrain_required else "NO_RETRAIN_REQUIRED",
        "retrain_required": retrain_required,
        "trigger_reason": "; ".join(trigger_reasons) if trigger_reasons else "All monitoring metrics within normal bounds",
        "monitoring_summary": summary,
    }


def evaluate_retraining_decision(curr_df: Optional[pd.DataFrame] = None, force: bool = False) -> Dict[str, Any]:
    """
    Evaluates aggregated monitoring indicators and dataset growth to determine if model retraining is required.
    
    Trigger Conditions Checked:
    1. Data Drift Detected (KS-Test feature ratio > max_drifted_feature_ratio)
    2. Prediction Probability Shift (Wasserstein distance > threshold)
    3. Performance Degradation (Recall < min_acceptable_recall)
    4. New Labeled Data Volume Threshold Met (>= min_new_samples_threshold)
    5. Manual / Force Override Flag
    
    Returns:
        Dict containing decision status ('RETRAIN_REQUIRED' or 'NO_RETRAIN_REQUIRED') and trigger_reason.
    """
    config = load_config()
    min_new_samples = config.get("retraining", {}).get("min_new_samples_threshold", 500)
    
    logger.info("Evaluating Automated Retraining Decision Criteria...")
    
    if force:
        logger.info("Retraining decision: RETRAIN_REQUIRED (Manual / Force Override).")
        return {
            "status": "RETRAIN_REQUIRED",
            "retrain_required": True,
            "trigger_reason": "Manual / Force Override",
            "monitoring_summary": {}
        }
        
    monitoring_report = run_monitoring_suite(curr_df=curr_df)
    curr_samples_count = len(curr_df) if curr_df is not None else 0
    decision = decide_from_summary(monitoring_report["summary"], curr_samples_count, min_new_samples)
    
    logger.info(f"Retraining Decision Evaluated -> Status: '{decision['status']}', Reason: '{decision['trigger_reason']}'")
    return decision


def record_retraining_history(entry: Dict[str, Any]) -> Path:
    """Appends retraining execution entry to models/retraining_history.json."""
    config = load_config()
    history_filename = config.get("retraining", {}).get("history_filename", "retraining_history.json")
    history_path = get_path("models") / history_filename
    history_path.parent.mkdir(parents=True, exist_ok=True)
    
    history = []
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
            
    history.append(entry)
    
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    logger.info(f"Recorded retraining history entry to: {history_path}")
    return history_path


def _snapshot_serving_artifacts() -> Path:
    """Copies the currently served model files into a temp directory so they can be restored on rejection."""
    snapshot_dir = Path(tempfile.mkdtemp(prefix="model_snapshot_"))
    models_dir = get_path("models")
    for name in _SERVING_ARTIFACTS:
        src = models_dir / name
        if src.exists():
            shutil.copy2(src, snapshot_dir / name)
    return snapshot_dir


def _restore_serving_artifacts(snapshot_dir: Path) -> None:
    """Puts the previously served model files back (undoes a rejected/failed candidate)."""
    models_dir = get_path("models")
    for name in _SERVING_ARTIFACTS:
        backup = snapshot_dir / name
        if backup.exists():
            shutil.copy2(backup, models_dir / name)
    logger.warning("Restored previous production model artifacts in models/.")


def _load_previous_test_metrics() -> Optional[Dict[str, Any]]:
    """Reads the test metrics of the model that is currently being served (before retraining overwrites them)."""
    path = get_path("models") / "evaluation_metrics.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("selected_model_test_metrics")
    except Exception:
        return None


def _revert_production_alias(old_version: str) -> None:
    """Points the @Production alias back to the previous version in the MLflow registry."""
    if old_version in ("None", "", None):
        return
    try:
        client = get_mlflow_client()
        client.set_registered_model_alias(name=MODEL_NAME, alias="Production", version=str(old_version))
        logger.warning(f"Reverted '@Production' alias back to Old Version {old_version}.")
    except Exception as exc:
        logger.error(f"Failed to revert '@Production' alias to version {old_version}: {exc}")


def execute_automated_retraining(
    force: bool = False,
    curr_df: Optional[pd.DataFrame] = None,
    decision: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes automated model retraining workflow:
    1. Evaluates retraining decision criteria (or uses a precomputed ``decision``, e.g. derived from the
       latest monitoring report by the dashboard).
    2. If NO_RETRAIN_REQUIRED, exits cleanly.
    3. If RETRAIN_REQUIRED:
       a. Captures current Production model version, metrics and served artifacts (Old Version).
       b. Executes retraining & evaluation pipeline.
       c. Compares the candidate against the old Production model using the promotion gate.
       d. Keeps the candidate as @Production ONLY IF the gate passes; otherwise restores the
          previous alias AND the previous model files that the API serves.
       e. Records full retraining history.
    """
    if decision is None:
        decision = evaluate_retraining_decision(curr_df=curr_df, force=force)
    
    if decision["status"] == "NO_RETRAIN_REQUIRED":
        logger.info("No retraining required at this time. Skipping pipeline execution.")
        return {
            "status": "SKIPPED",
            "reason": decision["trigger_reason"],
            "decision": decision
        }
        
    logger.info(f"Triggering Automated Retraining Pipeline! Reason: {decision['trigger_reason']}")
    
    # 1. Capture old production state
    old_prod_info = get_production_model_info(MODEL_NAME)
    old_version = str(old_prod_info.get("version", "None")) if old_prod_info else "None"
    old_test_metrics = _load_previous_test_metrics()
    snapshot_dir = _snapshot_serving_artifacts()
    
    try:
        # 2. Execute Training & MLflow Logging Pipeline
        try:
            eval_summary = train_and_evaluate_all()
        except Exception as exc:
            logger.error(f"Retraining failed during training: {exc}")
            _restore_serving_artifacts(snapshot_dir)
            _revert_production_alias(old_version)
            record_retraining_history({
                "timestamp": pd.Timestamp.now().isoformat(),
                "old_model_version": old_version,
                "new_model_version": None,
                "promotion_status": "FAILED",
                "promotion_reason": f"Training failed: {exc}",
                "trigger_reason": decision["trigger_reason"],
                "selected_algorithm": None,
                "test_metrics": None,
            })
            raise

        new_version = str(eval_summary["registry_info"]["version"])
        new_test_metrics = eval_summary["selected_model_test_metrics"]
        
        # 3. Promotion safety gate (candidate vs. current production)
        criteria = load_config().get("retraining", {}).get("promotion_criteria", {})
        gate = evaluate_promotion(old_test_metrics, new_test_metrics, criteria)
        promotion_status = gate["status"]
        promotion_reason = f"New Model Version {new_version}: {gate['reason']}"
        
        if not gate["promote"]:
            _restore_serving_artifacts(snapshot_dir)
            _revert_production_alias(old_version)
                
        # 4. Record Retraining History Entry
        history_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "old_model_version": old_version,
            "new_model_version": new_version,
            "promotion_status": promotion_status,
            "promotion_reason": promotion_reason,
            "trigger_reason": decision["trigger_reason"],
            "selected_algorithm": eval_summary["selected_model"],
            "test_metrics": new_test_metrics,
            "previous_test_metrics": old_test_metrics,
            "gate_checks": gate["checks"],
        }
        record_retraining_history(history_entry)
        
        logger.info(f"Automated Retraining Complete -> Status: {promotion_status}. {promotion_reason}")
        return history_entry
    finally:
        shutil.rmtree(snapshot_dir, ignore_errors=True)

if __name__ == "__main__":
    logger.info("Executing Automated Retraining Engine...")
    result = execute_automated_retraining(force=True)
    print("\n================ RETRAIN ENGINE RESULT ================")
    print(f"Trigger Reason    : {result.get('trigger_reason')}")
    print(f"Old Model Version : {result.get('old_model_version')}")
    print(f"New Model Version : {result.get('new_model_version')}")
    print(f"Promotion Status  : {result.get('promotion_status')}")
    print("=======================================================\n")
