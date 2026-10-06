"""
Read-only views over the MLflow tracking store and model registry for the dashboard.

Entity -> dict conversion is kept in small pure functions (duck-typed on MLflow's Run / ModelVersion
objects) so it is independent of the MLflow version in use.
"""
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.api.errors import ServiceError
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("api_mlflow_reader")

METRIC_KEYS = ["val_accuracy", "val_precision", "val_recall", "val_f1_score", "val_roc_auc", "val_pr_auc",
               "test_accuracy", "test_precision", "test_recall", "test_f1_score", "test_roc_auc", "test_pr_auc"]


def _iso(ms: Optional[int]) -> Optional[str]:
    if not ms:
        return None
    return pd.Timestamp(ms, unit="ms", tz="UTC").isoformat()


def run_to_dict(run: Any) -> Dict[str, Any]:
    info, data = run.info, run.data
    start, end = getattr(info, "start_time", None), getattr(info, "end_time", None)
    params = dict(data.params or {})
    tags = dict(data.tags or {})
    return {
        "run_id": info.run_id,
        "name": getattr(info, "run_name", None) or tags.get("mlflow.runName") or info.run_id[:8],
        "algorithm": params.get("model_type") or tags.get("model_type"),
        "status": info.status,
        "started_at": _iso(start),
        "duration_s": round((end - start) / 1000, 1) if start and end else None,
        "metrics": {k: round(float(v), 4) for k, v in (data.metrics or {}).items() if k in METRIC_KEYS},
        "stage": tags.get("stage"),
    }


def version_to_dict(mv: Any, metrics_by_run: Dict[str, Dict[str, float]], algorithm_by_run: Dict[str, Optional[str]]) -> Dict[str, Any]:
    metrics = metrics_by_run.get(mv.run_id, {})
    return {
        "version": str(mv.version),
        "run_id": mv.run_id,
        "status": str(mv.status),
        "created_at": _iso(getattr(mv, "creation_timestamp", None)),
        "description": getattr(mv, "description", None) or "",
        "aliases": list(getattr(mv, "aliases", None) or []),
        "algorithm": algorithm_by_run.get(mv.run_id),
        "metrics": metrics,
    }


def _client():
    from src.models.register import get_mlflow_client
    try:
        return get_mlflow_client()
    except Exception as exc:
        raise ServiceError(503, f"MLflow tracking store is unavailable: {exc}")


def _experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", load_config().get("mlflow", {}).get("experiment_name", "predictive-maintenance"))


def _model_name() -> str:
    return os.getenv("MLFLOW_REGISTERED_MODEL_NAME", load_config().get("mlflow", {}).get("registered_model_name", "PredictiveMaintenanceModel"))


def list_runs(limit: int = 200) -> Dict[str, Any]:
    client = _client()
    try:
        experiment = client.get_experiment_by_name(_experiment_name())
        if experiment is None:
            return {"experiment": None, "total": 0, "runs": []}
        runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=max(1, min(limit, 500)))
        return {
            "experiment": {
                "id": experiment.experiment_id,
                "name": experiment.name,
                "artifact_location": experiment.artifact_location,
            },
            "total": len(runs),
            "runs": [run_to_dict(r) for r in runs],
        }
    except ServiceError:
        raise
    except Exception as exc:
        logger.error(f"Listing MLflow runs failed: {exc}")
        raise ServiceError(503, f"Could not read MLflow runs: {exc}")


def list_versions(limit: int = 50) -> Dict[str, Any]:
    client = _client()
    name = _model_name()
    try:
        versions = list(client.search_model_versions(f"name='{name}'"))
        versions.sort(key=lambda v: int(v.version), reverse=True)
        total = len(versions)
        versions = versions[: max(1, min(limit, 200))]

        metrics_by_run: Dict[str, Dict[str, float]] = {}
        algorithm_by_run: Dict[str, Optional[str]] = {}
        experiment = client.get_experiment_by_name(_experiment_name())
        if experiment is not None:
            for run in client.search_runs([experiment.experiment_id], max_results=500):
                d = run_to_dict(run)
                metrics_by_run[d["run_id"]] = d["metrics"]
                algorithm_by_run[d["run_id"]] = d["algorithm"]

        production = next((str(v.version) for v in versions if "Production" in (getattr(v, "aliases", None) or [])), None)
        return {
            "model_name": name,
            "production_version": production,
            "total": total,
            "versions": [version_to_dict(v, metrics_by_run, algorithm_by_run) for v in versions],
        }
    except ServiceError:
        raise
    except Exception as exc:
        logger.error(f"Listing registry versions failed: {exc}")
        raise ServiceError(503, f"Could not read the model registry: {exc}")


def production_info() -> Optional[Dict[str, Any]]:
    """Registry details of the @Production version (None if unavailable)."""
    try:
        from src.models.register import get_production_model_info
        info = get_production_model_info(_model_name())
        if not info:
            return None
        info = dict(info)
        info["created_at"] = _iso(info.pop("creation_timestamp", None))
        info["version"] = str(info.get("version"))
        return info
    except Exception as exc:
        logger.warning(f"Production model lookup failed: {exc}")
        return None
