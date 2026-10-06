import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config, resolve_tracking_uri

logger = get_logger("model_registry")

def get_mlflow_client() -> MlflowClient:
    """Returns initialized MLflowClient with configured tracking URI."""
    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)

def register_model_from_run(
    run_id: str,
    artifact_path: str = "model",
    registered_model_name: Optional[str] = None,
    alias: str = "Production",
    description: str = "Best predictive maintenance failure prediction model"
) -> Dict[str, Any]:
    """
    Registers a model from a specific MLflow run ID into the MLflow Model Registry.
    Applies semantic versioning and assigns alias tag ('Production', 'Staging', 'Candidate').
    
    Args:
        run_id: MLflow run identifier.
        artifact_path: Logged model directory path within the run artifacts.
        registered_model_name: Target registered model name.
        alias: Stage alias tag ('Production', 'Staging', 'Candidate').
        description: Description metadata.
        
    Returns:
        Dict containing registered model version details.
    """
    config = load_config()
    if registered_model_name is None:
        registered_model_name = os.getenv(
            "MLFLOW_REGISTERED_MODEL_NAME",
            config.get("mlflow", {}).get("registered_model_name", "PredictiveMaintenanceModel")
        )
        
    client = get_mlflow_client()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    
    logger.info(f"Registering model from run '{run_id}' as '{registered_model_name}'...")
    
    # 1. Create or register model version
    mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    version_str = str(mv.version)
    logger.info(f"Successfully registered '{registered_model_name}' Version {version_str}.")
    
    # 2. Update model description & tags
    client.update_model_version(
        name=registered_model_name,
        version=version_str,
        description=description
    )
    
    # 3. Assign Alias (e.g. 'Production')
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=version_str
    )
    logger.info(f"Assigned alias '@{alias}' to '{registered_model_name}' Version {version_str}.")
    
    info = {
        "model_name": registered_model_name,
        "version": version_str,
        "alias": alias,
        "run_id": run_id,
        "model_uri": model_uri
    }
    
    return info

def get_production_model_info(registered_model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves metadata for the model version tagged with '@Production' alias.
    """
    config = load_config()
    if registered_model_name is None:
        registered_model_name = os.getenv(
            "MLFLOW_REGISTERED_MODEL_NAME",
            config.get("mlflow", {}).get("registered_model_name", "PredictiveMaintenanceModel")
        )
        
    client = get_mlflow_client()
    
    try:
        model_version = client.get_model_version_by_alias(name=registered_model_name, alias="Production")
        info = {
            "name": model_version.name,
            "version": model_version.version,
            "run_id": model_version.run_id,
            "source": model_version.source,
            "status": model_version.status,
            "creation_timestamp": model_version.creation_timestamp
        }
        logger.info(f"Active Production Model: '{model_version.name}' Version {model_version.version}")
        return info
    except Exception as e:
        logger.warning(f"No model found under '@Production' alias for '{registered_model_name}': {str(e)}")
        return None

if __name__ == "__main__":
    logger.info("Executing Model Registry Inspection...")
    info = get_production_model_info()
    if info:
        print(f"Production Model Info: {info}")
    else:
        print("No Production model currently registered.")
