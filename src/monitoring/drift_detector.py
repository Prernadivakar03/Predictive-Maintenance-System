import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from scipy.stats import ks_2samp, chisquare, wasserstein_distance
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config, get_path
from src.models.evaluate import calculate_classification_metrics
from src.models.train import get_candidate_models

logger = get_logger("drift_detector")

def load_reference_data(config: dict = None) -> pd.DataFrame:
    """Loads baseline reference dataset (training dataset)."""
    if config is None:
        config = load_config()
    ref_path = get_path(config.get("monitoring", {}).get("reference_data_path", "data/processed/train.csv"))
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference dataset not found at '{ref_path}'. Run src/pipeline.py first.")
    logger.info(f"Loaded reference dataset from: {ref_path} (Shape: {pd.read_csv(ref_path).shape})")
    return pd.read_csv(ref_path)

def load_production_model_and_preprocessor() -> Tuple[Any, Any]:
    """Loads current production model and preprocessor artifacts."""
    config = load_config()
    processed_dir = get_path(config["data"]["processed_dir"])
    model_path = get_path("models/best_model.joblib")
    preprocessor_path = processed_dir / "preprocessor.joblib"
    
    if not (model_path.exists() and preprocessor_path.exists()):
        raise FileNotFoundError("Model or preprocessor artifact missing from models/ or data/processed/.")
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def detect_data_drift(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    target_col: str = "Machine failure",
    alpha: float = 0.05,
    max_drift_ratio: float = 0.30
) -> Dict[str, Any]:
    """
    Detects feature distribution shifts using 2-sample Kolmogorov-Smirnov (KS) test.
    
    Args:
        ref_df: Reference baseline DataFrame.
        curr_df: Current production/inference DataFrame.
        target_col: Name of target column to exclude from feature drift calculation.
        alpha: Statistical significance p-value threshold (p < alpha indicates drift).
        max_drift_ratio: Ratio of drifted features required to flag overall data drift.
        
    Returns:
        Dict containing per-feature drift p-values, drifted feature counts, and boolean flag.
    """
    logger.info("Evaluating Feature Data Drift (KS-Test)...")
    feature_cols = [c for c in ref_df.columns if c != target_col]
    
    feature_drift_details = {}
    drifted_features = []
    
    for col in feature_cols:
        if col in curr_df.columns:
            # Drop NaN for clean KS comparison
            ref_vals = ref_df[col].dropna().values
            curr_vals = curr_df[col].dropna().values
            
            ks_stat, p_val = ks_2samp(ref_vals, curr_vals)
            p_val_float = float(round(p_val, 5))
            ks_stat_float = float(round(ks_stat, 5))
            
            is_drifted = bool(p_val < alpha)
            if is_drifted:
                drifted_features.append(col)
                
            feature_drift_details[col] = {
                "ks_statistic": ks_stat_float,
                "p_value": p_val_float,
                "is_drifted": is_drifted
            }
            
    drifted_ratio = float(len(drifted_features)) / len(feature_cols) if feature_cols else 0.0
    data_drift_detected = bool(drifted_ratio > max_drift_ratio)
    
    logger.info(
        f"Data Drift Audit Complete -> {len(drifted_features)} / {len(feature_cols)} features drifted "
        f"({drifted_ratio:.1%}). Overall Data Drift Detected: {data_drift_detected}"
    )
    
    return {
        "data_drift_detected": data_drift_detected,
        "drifted_features_count": len(drifted_features),
        "total_features_count": len(feature_cols),
        "drifted_feature_ratio": round(drifted_ratio, 4),
        "drifted_features": drifted_features,
        "feature_details": feature_drift_details
    }

def detect_prediction_drift(
    model: Any,
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    target_col: str = "Machine failure",
    limit: float = 0.10
) -> Dict[str, Any]:
    """
    Detects shift in predicted failure probability distributions using Wasserstein Distance.
    """
    logger.info("Evaluating Prediction Probability Shift (Wasserstein Distance)...")
    
    X_ref = ref_df.drop(columns=[target_col], errors="ignore").values
    X_curr = curr_df.drop(columns=[target_col], errors="ignore").values
    
    ref_probs = model.predict_proba(X_ref)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_ref)
    curr_probs = model.predict_proba(X_curr)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_curr)
    
    w_dist = float(wasserstein_distance(ref_probs, curr_probs))
    pred_drift_detected = bool(w_dist > limit)
    
    logger.info(f"Prediction Probability Wasserstein Distance: {w_dist:.4f} (Threshold: {limit}). Shift Detected: {pred_drift_detected}")
    
    return {
        "prediction_drift_detected": pred_drift_detected,
        "wasserstein_distance": round(w_dist, 4),
        "ref_mean_probability": round(float(ref_probs.mean()), 4),
        "curr_mean_probability": round(float(curr_probs.mean()), 4)
    }

def detect_performance_degradation(
    model: Any,
    curr_df: pd.DataFrame,
    target_col: str = "Machine failure",
    min_recall: float = 0.75,
    min_f1: float = 0.65
) -> Dict[str, Any]:
    """
    Calculates actual model metrics when ground-truth labels arrive and flags degradation.
    """
    if target_col not in curr_df.columns:
        logger.warning(f"Target column '{target_col}' not in current dataset. Ground-truth evaluation skipped.")
        return {
            "performance_degraded": False,
            "ground_truth_available": False,
            "metrics": {}
        }
        
    X_curr = curr_df.drop(columns=[target_col]).values
    y_curr = curr_df[target_col].values
    
    y_pred = model.predict(X_curr)
    y_prob = model.predict_proba(X_curr)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = calculate_classification_metrics(y_curr, y_pred, y_prob=y_prob)
    
    curr_recall = metrics["recall"]
    curr_f1 = metrics["f1_score"]
    
    degraded = bool(curr_recall < min_recall or curr_f1 < min_f1)
    
    logger.info(
        f"Ground-Truth Performance -> Recall: {curr_recall} (Min: {min_recall}), "
        f"F1: {curr_f1} (Min: {min_f1}). Performance Degraded: {degraded}"
    )
    
    return {
        "performance_degraded": degraded,
        "ground_truth_available": True,
        "metrics": metrics
    }

def run_monitoring_suite(curr_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Master monitoring function:
    1. Loads reference data & production model.
    2. Runs Data Drift Detection (KS-Test).
    3. Runs Prediction Probability Shift Detection (Wasserstein Distance).
    4. Runs Ground-Truth Degradation Audit if labels are present.
    5. Saves full JSON report and summary JSON to data/reports/.
    """
    config = load_config()
    mon_config = config.get("monitoring", {})
    thresholds = mon_config.get("thresholds", {})
    
    ref_df = load_reference_data(config=config)
    
    if curr_df is None:
        # Default to loading validation/test processed set as current inference batch
        val_path = get_path(config["data"]["processed_dir"]) / "val.csv"
        if val_path.exists():
            curr_df = pd.read_csv(val_path)
        else:
            curr_df = ref_df.copy()
            
    model, _ = load_production_model_and_preprocessor()
    target_col = config["validation"]["target_column"]
    
    # 1. Data Drift Audit
    data_drift_res = detect_data_drift(
        ref_df=ref_df,
        curr_df=curr_df,
        target_col=target_col,
        alpha=thresholds.get("p_value_alpha", 0.05),
        max_drift_ratio=thresholds.get("max_drifted_feature_ratio", 0.30)
    )
    
    # 2. Prediction Shift Audit
    pred_drift_res = detect_prediction_drift(
        model=model,
        ref_df=ref_df,
        curr_df=curr_df,
        target_col=target_col,
        limit=thresholds.get("wasserstein_prob_drift_limit", 0.10)
    )
    
    # 3. Model Performance Degradation Audit
    perf_res = detect_performance_degradation(
        model=model,
        curr_df=curr_df,
        target_col=target_col,
        min_recall=thresholds.get("min_acceptable_recall", 0.75),
        min_f1=thresholds.get("min_acceptable_f1", 0.65)
    )
    
    # Master Decision Logic
    retraining_required = bool(
        data_drift_res["data_drift_detected"] or
        pred_drift_res["prediction_drift_detected"] or
        perf_res["performance_degraded"]
    )
    
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "data_drift_detected": data_drift_res["data_drift_detected"],
        "prediction_drift_detected": pred_drift_res["prediction_drift_detected"],
        "performance_degraded": perf_res["performance_degraded"],
        "retraining_required": retraining_required,
        "drifted_features_count": data_drift_res["drifted_features_count"],
        "wasserstein_distance": pred_drift_res["wasserstein_distance"],
        "current_recall": perf_res["metrics"].get("recall", None) if perf_res["ground_truth_available"] else None
    }
    
    full_report = {
        "summary": summary,
        "data_drift": data_drift_res,
        "prediction_drift": pred_drift_res,
        "performance_degradation": perf_res
    }
    
    # Save Reports
    reports_dir = get_path(mon_config.get("reports_dir", "data/reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = reports_dir / mon_config.get("summary_filename", "monitoring_summary.json")
    report_path = reports_dir / mon_config.get("json_report_filename", "drift_report.json")
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        
    logger.info(f"Saved Monitoring Summary to: {summary_path}")
    logger.info(f"Saved Monitoring Full Report to: {report_path}")
    
    return full_report

if __name__ == "__main__":
    logger.info("Executing Monitoring and Drift Detection Suite...")
    report = run_monitoring_suite()
    summary = report["summary"]
    print("\n================ MONITORING SUMMARY ================")
    print(f"Data Drift Detected       : {summary['data_drift_detected']}")
    print(f"Prediction Shift Detected : {summary['prediction_drift_detected']}")
    print(f"Performance Degraded     : {summary['performance_degraded']}")
    print(f"Retraining Required       : {summary['retraining_required']}")
    print("===================================================\n")
