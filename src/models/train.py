import os
import sys
import json
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config, get_path, resolve_tracking_uri, PROJECT_ROOT
from src.models.evaluate import (
    calculate_classification_metrics,
    plot_confusion_matrix,
    plot_model_comparison_curves
)
from src.models.register import register_model_from_run

logger = get_logger("model_training")

def _artifact_location_usable(artifact_location: str) -> bool:
    """
    Checks whether an experiment's stored artifact root can be written on THIS machine.

    MLflow bakes the absolute artifact path into the tracking DB when an experiment is created
    (e.g. ``file:///D:/MLOOPS/mlruns/1``). After moving the project, or running inside Docker/Linux,
    that path no longer exists and every training run fails while logging artifacts.
    """
    from urllib.parse import urlparse, unquote
    from urllib.request import url2pathname

    parsed = urlparse(artifact_location or "")
    if parsed.scheme not in ("", "file"):
        return True  # remote artifact stores (s3://, mlflow-artifacts:/ ...) are managed elsewhere
    try:
        path = Path(url2pathname(unquote(parsed.path))) if parsed.scheme == "file" else Path(artifact_location)
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def setup_mlflow() -> str:
    """Configures MLflow tracking URI and initializes experiment."""
    config = load_config()
    tracking_uri = resolve_tracking_uri()
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        config.get("mlflow", {}).get("experiment_name", "predictive-maintenance")
    )
    
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Self-heal: an experiment pointing at an artifact folder that doesn't exist on this machine
    # is archived (history preserved) and recreated with a project-local artifact root.
    if experiment is not None and not _artifact_location_usable(experiment.artifact_location):
        try:
            from datetime import datetime
            archived_name = f"{experiment_name} (archived {datetime.now():%Y%m%d-%H%M%S})"
            logger.warning(
                f"Experiment '{experiment_name}' points to an unusable artifact path "
                f"'{experiment.artifact_location}'. Archiving it as '{archived_name}' and recreating it locally."
            )
            mlflow.tracking.MlflowClient(tracking_uri=tracking_uri).rename_experiment(experiment.experiment_id, archived_name)
            experiment = None
        except Exception as exc:
            logger.error(f"Could not repair experiment artifact location automatically: {exc}")

    if experiment is None:
        artifact_root = (PROJECT_ROOT / config.get("mlflow", {}).get("artifact_location", "mlruns")).resolve()
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())
        logger.info(f"Created new MLflow experiment '{experiment_name}' with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment '{experiment_name}' with ID: {experiment_id}")
        
    mlflow.set_experiment(experiment_name)
    return experiment_id

def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads processed train, validation, and test datasets from data/processed/."""
    config = load_config()
    processed_dir = get_path(config["data"]["processed_dir"])
    
    train_path = processed_dir / "train.joblib"
    val_path = processed_dir / "val.joblib"
    test_path = processed_dir / "test.joblib"
    
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"Processed data files not found in '{processed_dir}'. Run src/features/build_features.py first."
        )
        
    df_train = joblib.load(train_path)
    df_val = joblib.load(val_path)
    df_test = joblib.load(test_path)
    
    logger.info(f"Loaded processed datasets -> Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")
    return df_train, df_val, df_test

def get_candidate_models(model_config: dict = None) -> Dict[str, Any]:
    """Initializes candidate classifiers with reproducible seeds and parameters."""
    if model_config is None:
        model_config_path = get_path("config/model_config.json")
        if model_config_path.exists():
            with open(model_config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)
        else:
            model_config = {}
            
    xgb_params = model_config.get("model_parameters", {}).get("xgboost", {
        "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "scale_pos_weight": 28.0, "random_state": 42
    })
    
    rf_params = model_config.get("model_parameters", {}).get("random_forest", {
        "n_estimators": 100, "max_depth": 10, "class_weight": "balanced", "random_state": 42
    })
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(**rf_params),
        "XGBoost": XGBClassifier(**xgb_params)
    }
    
    return models

def train_and_evaluate_all(target_col: str = "Machine failure") -> Dict[str, Any]:
    """
    Executes model training, evaluation, MLflow tracking, and model registration:
    1. Sets up MLflow experiment.
    2. Trains candidate models and logs parameters, metrics, tags, and artifacts to MLflow.
    3. Selects Best Model (Validation Recall & F1).
    4. Registers Best Model in MLflow Model Registry as 'PredictiveMaintenanceModel' @Production.
    5. Saves local joblib backups to models/.
    """
    setup_mlflow()
    df_train, df_val, df_test = load_processed_data()
    
    X_train = df_train.drop(columns=[target_col]).values
    y_train = df_train[target_col].values
    
    X_val = df_val.drop(columns=[target_col]).values
    y_val = df_val[target_col].values
    
    X_test = df_test.drop(columns=[target_col]).values
    y_test = df_test[target_col].values
    
    feature_names = [col for col in df_train.columns if col != target_col]
    candidate_models = get_candidate_models()
    
    val_results = {}
    fitted_models = {}
    mlflow_run_ids = {}
    
    logger.info("Starting MLflow Tracked Candidate Model Training...")
    
    for name, model in candidate_models.items():
        run_name = f"{name.replace(' ', '')}_Run"
        
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow_run_ids[name] = run_id
            
            # 1. Log MLflow Tags
            mlflow.set_tags({
                "project": "predictive-maintenance",
                "model_type": name,
                "dataset_version": "1.0.0",
                "environment": "development",
                "task": "classification"
            })
            
            # 2. Log Parameters
            params = model.get_params() if hasattr(model, "get_params") else {}
            mlflow.log_params({
                "model_type": name,
                "random_state": params.get("random_state", 42),
                "num_features": len(feature_names),
                "scaling": "StandardScaler"
            })
            # Log hyperparameters (sanitize dict values for MLflow)
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(f"hp_{k}", v)
                    
            # 3. Train Model
            logger.info(f"Training {name} on {len(X_train)} samples [Run ID: {run_id}]...")
            model.fit(X_train, y_train)
            fitted_models[name] = model
            
            # 4. Evaluate Validation Metrics
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
            
            metrics = calculate_classification_metrics(y_val, y_val_pred, y_prob=y_val_prob)
            val_results[name] = metrics
            
            # Log Metrics to MLflow
            mlflow.log_metrics({
                "val_accuracy": metrics["accuracy"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_f1_score": metrics["f1_score"],
                "val_roc_auc": metrics["roc_auc"],
                "val_pr_auc": metrics["pr_auc"]
            })
            
            # 5. Log Confusion Matrix Artifact
            cm_plot_path = plot_confusion_matrix(y_val, y_val_pred, model_name=name)
            mlflow.log_artifact(str(cm_plot_path), artifact_path="figures")
            
            # 6. Log Model Binary Artifact
            if "XGBoost" in name:
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
                
            logger.info(f"Logged run '{name}' to MLflow with Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")

    # Plot Comparison Curves
    comp_plot_path = plot_model_comparison_curves(fitted_models, X_val, y_val)
    
    # 7. Select Best Model (Maximizing Validation Recall & F1)
    best_model_name = max(
        val_results.keys(),
        key=lambda k: (val_results[k]["recall"], val_results[k]["f1_score"], val_results[k]["pr_auc"])
    )
    
    best_model = fitted_models[best_model_name]
    best_val_metrics = val_results[best_model_name]
    best_run_id = mlflow_run_ids[best_model_name]
    
    logger.info(f"Selected Best Model: '{best_model_name}' (Run ID: {best_run_id})")
    
    # 8. Register Best Model in MLflow Registry
    reg_info = register_model_from_run(
        run_id=best_run_id,
        registered_model_name="PredictiveMaintenanceModel",
        alias="Production",
        description=f"Best model '{best_model_name}' selected based on Validation Recall={best_val_metrics['recall']}."
    )
    
    # 9. Evaluate Best Model on Unseen Test Set
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    test_metrics = calculate_classification_metrics(y_test, y_test_pred, y_prob=y_test_prob)
    
    # Log Test Metrics to Best Run
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metrics({
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1_score": test_metrics["f1_score"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_pr_auc": test_metrics["pr_auc"]
        })
        mlflow.set_tag("stage", "Production")

    # 10. Local Joblib Backup Savings
    models_dir = get_path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / "best_model.joblib")

    # Model metadata (documented in README but previously never written -> went stale after retraining)
    try:
        metadata = {
            "model_name": best_model_name,
            "algorithm": type(best_model).__name__,
            "model_params": {k: v for k, v in best_model.get_params().items() if isinstance(v, (int, float, str, bool, type(None)))},
            "feature_names": feature_names,
            "num_features": len(feature_names),
            "target_column": target_col,
            "registered_model_version": str(reg_info["version"]),
            "mlflow_run_id": best_run_id,
            "trained_at": pd.Timestamp.now().isoformat(),
        }
        with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as exc:
        logger.warning(f"Could not write model_metadata.json: {exc}")
    
    eval_summary = {
        "selected_model": best_model_name,
        "selected_run_id": best_run_id,
        "registry_info": reg_info,
        "candidate_val_metrics": val_results,
        "selected_model_test_metrics": test_metrics
    }
    
    with open(models_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
        
    return eval_summary

if __name__ == "__main__":
    logger.info("Executing MLflow Tracked Training & Registration Pipeline...")
    summary = train_and_evaluate_all()
    print("\n================ MLFLOW TRAINING & REGISTRATION SUMMARY ================")
    print(f"Selected Best Model : {summary['selected_model']}")
    print(f"MLflow Run ID       : {summary['selected_run_id']}")
    print(f"Registered Model    : {summary['registry_info']['model_name']} (Version {summary['registry_info']['version']} @{summary['registry_info']['alias']})")
    print(f"Test Recall         : {summary['selected_model_test_metrics']['recall']}")
    print(f"Test F1-Score       : {summary['selected_model_test_metrics']['f1_score']}")
    print("========================================================================\n")
