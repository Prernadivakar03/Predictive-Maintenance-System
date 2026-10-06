import os
import sys
import threading
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import load_config, get_path
from src.models.register import get_production_model_info

logger = get_logger("api_model_loader")

# In-memory Singleton Cache for Artifacts
_MODEL_CACHE: Dict[str, Any] = {
    "model": None,
    "preprocessor": None,
    "model_name": "PredictiveMaintenanceModel",
    "model_version": "1",
    "signature": None,
}
_LOAD_LOCK = threading.Lock()


def _artifact_paths() -> Tuple[Path, Path]:
    config = load_config()
    preprocessor_path = get_path(config["data"]["processed_dir"]) / "preprocessor.joblib"
    model_path = get_path("models/best_model.joblib")
    return preprocessor_path, model_path


def _signature(preprocessor_path: Path, model_path: Path) -> Tuple:
    """Modification times of the served artifacts; a change means a new model was trained or restored."""
    def mtime(p: Path):
        try:
            return p.stat().st_mtime_ns
        except OSError:
            return None
    return (mtime(preprocessor_path), mtime(model_path))


def load_artifacts(force_reload: bool = False) -> Tuple[Any, Any, str, str]:
    """
    Loads preprocessor and trained model artifacts into memory cache.
    The cache is refreshed automatically when the artifact files change on disk, so a retraining run
    (CLI, Airflow or dashboard) is picked up without restarting the API.
    
    Returns:
        Tuple of (model, preprocessor, model_name, model_version)
    """
    global _MODEL_CACHE
    
    preprocessor_path, model_path = _artifact_paths()
    signature = _signature(preprocessor_path, model_path)

    def cached():
        return (
            _MODEL_CACHE["model"],
            _MODEL_CACHE["preprocessor"],
            _MODEL_CACHE["model_name"],
            _MODEL_CACHE["model_version"]
        )

    if (not force_reload and _MODEL_CACHE["model"] is not None and _MODEL_CACHE["preprocessor"] is not None
            and _MODEL_CACHE["signature"] == signature):
        return cached()

    with _LOAD_LOCK:
        # Another thread may have finished loading while we waited for the lock
        if (not force_reload and _MODEL_CACHE["model"] is not None and _MODEL_CACHE["preprocessor"] is not None
                and _MODEL_CACHE["signature"] == signature):
            return cached()

        # 1. Load Preprocessor
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact missing at '{preprocessor_path}'. Execute src/pipeline.py first.")
            
        logger.info(f"Loading Preprocessor from: {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
        
        # 2. Load Model Binary
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model artifact missing at '{model_path}'. Execute src/pipeline.py first.")
            
        logger.info(f"Loading Best Model binary from: {model_path}")
        model = joblib.load(model_path)
        
        # 3. Retrieve Model Version Info (a registry outage must not take inference down)
        model_name = "PredictiveMaintenanceModel"
        model_version = "1"
        try:
            prod_info = get_production_model_info(model_name)
            if prod_info:
                model_version = str(prod_info.get("version", "1"))
        except Exception as exc:
            logger.warning(f"Could not read model version from MLflow registry: {exc}")
            
        _MODEL_CACHE.update({
            "model": model,
            "preprocessor": preprocessor,
            "model_name": model_name,
            "model_version": model_version,
            "signature": signature,
        })
        
        logger.info(f"Model Artifacts Loaded Successfully! [{model_name} Version {model_version}]")
        return model, preprocessor, model_name, model_version

def run_inference(telemetry_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transforms raw input dictionary telemetry records and returns failure predictions.
    
    Args:
        telemetry_list: List of dictionaries matching TelemetryInput schema.
        
    Returns:
        List of prediction result dictionaries.
    """
    model, preprocessor, model_name, model_version = load_artifacts()
    
    # Convert input dicts to DataFrame
    raw_df = pd.DataFrame(telemetry_list)
    
    # Standardize Column Names if aliases were used
    column_mapping = {
        "air_temperature": "Air temperature [K]",
        "process_temperature": "Process temperature [K]",
        "rotational_speed": "Rotational speed [rpm]",
        "torque": "Torque [Nm]",
        "tool_wear": "Tool wear [min]"
    }
    raw_df = raw_df.rename(columns=column_mapping)
    
    # Transform using preprocessor pipeline
    X_proc = preprocessor.transform(raw_df)
    
    # Predict Class and Failure Probabilities
    predictions = model.predict(X_proc)
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_proc)[:, 1]
    else:
        probabilities = np.where(predictions == 1, 1.0, 0.0)
        
    timestamp = pd.Timestamp.now().isoformat()
    results = []
    
    for pred, prob in zip(predictions, probabilities):
        pred_int = int(pred)
        prob_float = float(round(prob, 4))
        status = "Equipment Failure Warning" if pred_int == 1 else "Normal Operation"
        
        results.append({
            "prediction": pred_int,
            "status": status,
            "failure_probability": prob_float,
            "model_name": model_name,
            "model_version": model_version,
            "timestamp": timestamp
        })
        
    return results
