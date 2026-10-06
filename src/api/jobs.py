"""
Background job runner for long-running operations triggered from the dashboard
(pipeline runs, retraining). One job at a time; state is persisted so history survives restarts.
"""
import json
import logging
import threading
import traceback
import uuid
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.api.errors import ServiceError
from src.utils.config import get_path
from src.utils.logger import get_logger

logger = get_logger("api_jobs")

MAX_JOBS = 50
MAX_LOG_LINES = 400

_LOCK = threading.RLock()
_JOBS: List[Dict[str, Any]] = []
_LOADED = False


def _jobs_file() -> Path:
    return get_path("data/reports/jobs.json")


def _persist() -> None:
    try:
        path = _jobs_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_JOBS, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception as exc:  # persistence is best-effort
        logger.warning(f"Could not persist job history: {exc}")


def _ensure_loaded() -> None:
    """Loads persisted history once; jobs that were 'running' when the process died are marked interrupted."""
    global _LOADED
    with _LOCK:
        if _LOADED:
            return
        _LOADED = True
        path = _jobs_file()
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                for job in loaded:
                    if job.get("status") == "running":
                        job["status"] = "interrupted"
                        job["error"] = "The API process stopped while this job was running."
                        job["finished_at"] = job.get("finished_at") or pd.Timestamp.now().isoformat()
                _JOBS.extend(loaded[:MAX_JOBS])
            except Exception as exc:
                logger.warning(f"Ignoring unreadable job history: {exc}")


class _JobLogHandler(logging.Handler):
    """Captures log records emitted by the job's own thread into the job record."""

    def __init__(self, job_id: str, thread_id: int):
        super().__init__(level=logging.INFO)
        self.job_id = job_id
        self.thread_id = thread_id

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.thread_id:
            return
        try:
            message = record.getMessage()
        except Exception:
            return
        with _LOCK:
            job = _find(self.job_id)
            if job is None:
                return
            for line in message.splitlines():
                line = line.strip()
                if not line:
                    continue
                if ">>> STAGE" in line:
                    job["stage"] = line.split(">>>", 1)[1].strip()
                job["log"].append({
                    "ts": pd.Timestamp.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": line,
                })
            if len(job["log"]) > MAX_LOG_LINES:
                job["log"] = job["log"][-MAX_LOG_LINES:]


def _find(job_id: str) -> Optional[Dict[str, Any]]:
    for job in _JOBS:
        if job["id"] == job_id:
            return job
    return None


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


def _reload_serving_model() -> None:
    """After a job the model files may have changed (or been restored): make the API serve what is on disk."""
    try:
        from src.api.model_loader import load_artifacts
        load_artifacts(force_reload=True)
    except Exception as exc:
        logger.warning(f"Model reload after job failed: {exc}")


def _run(job_id: str, target: Callable[[], Any]) -> None:
    handler = _JobLogHandler(job_id, threading.get_ident())
    root = logging.getLogger()
    root.addHandler(handler)
    started = pd.Timestamp.now()
    status, result, error = "succeeded", None, None
    try:
        result = target()
    except BaseException as exc:  # noqa: BLE001 - a job must never crash the worker thread silently
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        logger.error(f"Job {job_id} failed: {error}\n{traceback.format_exc(limit=3)}")
    finally:
        root.removeHandler(handler)

    _reload_serving_model()

    with _LOCK:
        job = _find(job_id)
        if job is not None:
            finished = pd.Timestamp.now()
            job.update({
                "status": status,
                "result": _jsonable(result),
                "error": error,
                "finished_at": finished.isoformat(),
                "duration_s": round((finished - started).total_seconds(), 1),
                "stage": "Finished" if status == "succeeded" else "Failed",
            })
            _persist()


def start_job(kind: str, target: Callable[[], Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Starts ``target`` in a background thread. Raises ServiceError(409) if another job is running."""
    _ensure_loaded()
    with _LOCK:
        running = next((j for j in _JOBS if j["status"] == "running"), None)
        if running:
            raise ServiceError(409, f"A {running['kind']} job is already running (id {running['id']}). Wait for it to finish.")
        job = {
            "id": uuid.uuid4().hex[:8],
            "kind": kind,
            "status": "running",
            "params": params or {},
            "started_at": pd.Timestamp.now().isoformat(),
            "finished_at": None,
            "duration_s": None,
            "stage": "Starting",
            "result": None,
            "error": None,
            "log": [],
        }
        _JOBS.insert(0, job)
        del _JOBS[MAX_JOBS:]
        _persist()
        job_id = job["id"]

    threading.Thread(target=_run, args=(job_id, target), daemon=True, name=f"job-{job_id}").start()
    return get_job(job_id)


def get_job(job_id: str) -> Dict[str, Any]:
    _ensure_loaded()
    with _LOCK:
        job = _find(job_id)
        if job is None:
            raise ServiceError(404, f"Job '{job_id}' not found.")
        return json.loads(json.dumps(job, default=str))


def list_jobs(limit: int = 20) -> List[Dict[str, Any]]:
    """Newest first, without the (large) log payload."""
    _ensure_loaded()
    with _LOCK:
        out = []
        for job in _JOBS[: max(1, min(limit, MAX_JOBS))]:
            slim = {k: v for k, v in job.items() if k != "log"}
            slim["log_lines"] = len(job.get("log", []))
            out.append(json.loads(json.dumps(slim, default=str)))
        return out


def running_job() -> Optional[Dict[str, Any]]:
    _ensure_loaded()
    with _LOCK:
        job = next((j for j in _JOBS if j["status"] == "running"), None)
        if not job:
            return None
        slim = {k: v for k, v in job.items() if k != "log"}
        return json.loads(json.dumps(slim, default=str))
