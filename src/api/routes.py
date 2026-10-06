"""
Dashboard API (mounted under /api). Thin FastAPI wrappers around the framework-independent service layer in
``insights.py`` (read) and ``operations.py`` (write), so all logic stays unit-testable without FastAPI.
"""
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from src.api import insights, jobs, mlflow_reader, operations, prediction_log
from src.api.errors import ServiceError
from src.api.schemas import MonitoringRunRequest, PipelineRunRequest, RetrainRequest
from src.api.timeutil import localize_timestamps

router = APIRouter(prefix="/api")


def _respond(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Runs a service function; converts ServiceError to HTTP errors and localizes timestamps."""
    try:
        return localize_timestamps(fn(*args, **kwargs))
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)


# ---- read ---------------------------------------------------------------------------------------------
@router.get("/overview", tags=["Dashboard"])
def get_overview():
    """Aggregated state for the dashboard landing page."""
    return _respond(insights.overview)


@router.get("/model", tags=["Model"])
def get_model():
    """Production model details: metrics, candidates, feature importance, parameters."""
    return _respond(insights.model_overview)


@router.get("/registry/versions", tags=["Model"])
def get_registry_versions(limit: int = Query(50, ge=1, le=200)):
    """MLflow model registry versions with aliases and metrics."""
    return _respond(mlflow_reader.list_versions, limit)


@router.get("/experiments/runs", tags=["Experiments"])
def get_experiment_runs(limit: int = Query(200, ge=1, le=500)):
    """MLflow runs of the project experiment, newest first."""
    return _respond(mlflow_reader.list_runs, limit)


@router.get("/data/summary", tags=["Data"])
def get_data_summary():
    """Dataset statistics, distributions and the latest data-quality report."""
    return _respond(insights.data_summary)


@router.get("/figures/{name}", tags=["Data"])
def get_figure(name: str):
    """Serves a generated report figure (PNG)."""
    try:
        return FileResponse(insights.figure_path(name), media_type="image/png")
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)


@router.get("/monitoring/status", tags=["Monitoring"])
def get_monitoring_status():
    """Latest drift report, history, thresholds and available scenarios."""
    return _respond(insights.monitoring_status)


@router.get("/retraining/status", tags=["Retraining"])
def get_retraining_status():
    """Retraining history, promotion criteria and the current retraining decision."""
    return _respond(insights.retraining_status)


@router.get("/pipeline/status", tags=["Pipeline"])
def get_pipeline_status():
    """DAG definition, per-stage artifact state and recent jobs."""
    return _respond(insights.pipeline_status)


@router.get("/jobs", tags=["Pipeline"])
def get_jobs(limit: int = Query(20, ge=1, le=50)):
    return _respond(jobs.list_jobs, limit)


@router.get("/jobs/{job_id}", tags=["Pipeline"])
def get_job(job_id: str):
    """A single job including its captured log lines."""
    return _respond(jobs.get_job, job_id)


@router.get("/predictions/recent", tags=["Predictions"])
def get_recent_predictions(limit: int = Query(50, ge=1, le=500)):
    return _respond(prediction_log.recent, limit)


@router.get("/predictions/stats", tags=["Predictions"])
def get_prediction_stats():
    return _respond(prediction_log.stats)


@router.get("/logs", tags=["System"])
def get_logs(lines: int = Query(200, ge=10, le=1000)):
    return _respond(insights.logs_tail, lines)


@router.get("/system", tags=["System"])
def get_system():
    return _respond(insights.system_info)


# ---- write --------------------------------------------------------------------------------------------
@router.post("/monitoring/run", tags=["Monitoring"])
def post_monitoring_run(request: MonitoringRunRequest):
    """Runs the monitoring suite on a scenario / uploaded batch and returns the new status."""
    return _respond(operations.run_monitoring, request.scenario, request.severity, request.rows)


@router.post("/pipeline/run", status_code=202, tags=["Pipeline"])
def post_pipeline_run(request: PipelineRunRequest):
    """Starts the ML pipeline as a background job (HTTP 409 if a job is already running)."""
    return _respond(operations.start_pipeline, request.force)


@router.post("/retraining/trigger", status_code=202, tags=["Retraining"])
def post_retraining_trigger(request: RetrainRequest):
    """Starts automated retraining as a background job (HTTP 409 if a job is already running)."""
    return _respond(operations.start_retraining, request.mode)


@router.post("/model/reload", tags=["Model"])
def post_model_reload():
    """Forces the API to reload the model artifacts from disk."""
    return _respond(operations.reload_model)
