"""
Read-only analytics for the dashboard: everything is derived from artifacts the pipeline already produces
(JSON reports, joblib models, CSVs, logs) plus the MLflow store. No function here mutates state.
"""
import json
import os
import platform
import re
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.api import jobs, mlflow_reader, prediction_log
from src.api.errors import ServiceError
from src.monitoring.scenarios import describe_scenarios
from src.retraining.trigger import decide_from_summary
from src.utils.config import load_config, get_path, resolve_tracking_uri
from src.utils.logger import get_logger

logger = get_logger("api_insights")

API_VERSION = "1.0.0"
_STARTED_AT = time.time()

SENSOR_META = {
    "Air temperature [K]": ("Air temperature", "K"),
    "Process temperature [K]": ("Process temperature", "K"),
    "Rotational speed [rpm]": ("Rotational speed", "rpm"),
    "Torque [Nm]": ("Torque", "Nm"),
    "Tool wear [min]": ("Tool wear", "min"),
}
FAILURE_MODES = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "RNF": "Random failure",
}
FIGURE_TITLES = {
    "target_distribution.png": "Target distribution",
    "failure_modes_breakdown.png": "Failure modes breakdown",
    "sensor_distributions.png": "Sensor distributions",
    "feature_vs_failure.png": "Features vs. failure",
    "correlation_matrix.png": "Correlation matrix",
    "model_comparison_curves.png": "ROC and precision-recall curves",
    "cm_logistic_regression.png": "Confusion matrix: Logistic Regression",
    "cm_random_forest.png": "Confusion matrix: Random Forest",
    "cm_xgboost.png": "Confusion matrix: XGBoost",
}
DAG_TASKS = [
    {"id": "data_ingestion", "label": "Data ingestion",
     "description": "Downloads or loads the raw AI4I 2020 sensor dataset.", "upstream": []},
    {"id": "data_validation", "label": "Data validation",
     "description": "Checks schema, value ranges, nulls and duplicates. Halts the run on failure.", "upstream": ["data_ingestion"]},
    {"id": "preprocessing", "label": "Preprocessing",
     "description": "Stratified split, domain feature engineering, scaling and encoding.", "upstream": ["data_validation"]},
    {"id": "model_training", "label": "Model training",
     "description": "Trains Logistic Regression, Random Forest and XGBoost; logs every run to MLflow.", "upstream": ["preprocessing"]},
    {"id": "mlflow_registration", "label": "Model registration",
     "description": "Registers the winning model and assigns the Production alias.", "upstream": ["model_training"]},
]


# --------------------------------------------------------------------------- helpers
def _read_json(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not read {path}: {exc}")
    return None


def _mtime_iso(path: Path) -> Optional[str]:
    try:
        return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()
    except OSError:
        return None


def _reports_dir() -> Path:
    return get_path(load_config()["data"]["reports_dir"])


def _models_dir() -> Path:
    return get_path("models")


def _links() -> Dict[str, str]:
    return {
        "mlflow": os.getenv("MLFLOW_UI_URL", "http://localhost:5000"),
        "airflow": os.getenv("AIRFLOW_UI_URL", "http://localhost:8080"),
        "api_docs": "/docs",
    }


# --------------------------------------------------------------------------- data
_DATA_CACHE: Dict[str, Any] = {"key": None, "value": None}


def data_summary() -> Dict[str, Any]:
    config = load_config()
    raw_path = get_path(config["data"]["raw_dir"]) / config["data"]["dataset_filename"]
    quality = _read_json(_reports_dir() / config["data"]["report_filename"])
    figures_dir = get_path("reports/figures")
    figures = [{"name": p.name, "title": FIGURE_TITLES.get(p.name, p.stem.replace("_", " ").capitalize())}
               for p in sorted(figures_dir.glob("*.png"))] if figures_dir.exists() else []

    if not raw_path.exists():
        return {"available": False, "path": str(raw_path), "quality": quality, "figures": figures}

    key = (str(raw_path), raw_path.stat().st_mtime_ns)
    if _DATA_CACHE["key"] == key:
        base = _DATA_CACHE["value"]
    else:
        target = config["validation"]["target_column"]
        df = pd.read_csv(raw_path)
        failed = df[target] == 1

        sensors = []
        for col, (label, unit) in SENSOR_META.items():
            if col not in df.columns:
                continue
            values = df[col].astype(float)
            counts_all, edges = np.histogram(values, bins=24)
            counts_fail, _ = np.histogram(values[failed], bins=edges)
            sensors.append({
                "column": col, "label": label, "unit": unit,
                "min": round(float(values.min()), 2), "max": round(float(values.max()), 2),
                "mean": round(float(values.mean()), 2), "std": round(float(values.std()), 2),
                "hist": {
                    "edges": [round(float(e), 3) for e in edges],
                    "normal": [int(a - b) for a, b in zip(counts_all, counts_fail)],
                    "failure": [int(c) for c in counts_fail],
                },
            })

        by_type = []
        if "Type" in df.columns:
            for t in ["L", "M", "H"]:
                sub = df[df["Type"] == t]
                if len(sub):
                    by_type.append({"type": t, "count": int(len(sub)), "failures": int(sub[target].sum()),
                                    "failure_rate_pct": round(float(sub[target].mean()) * 100, 2)})

        base = {
            "available": True,
            "path": str(raw_path),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "failure_rate_pct": round(float(df[target].mean()) * 100, 2),
            "class_counts": {"normal": int((~failed).sum()), "failure": int(failed.sum())},
            "failure_modes": [{"mode": m, "name": n, "count": int(df[m].sum())} for m, n in FAILURE_MODES.items() if m in df.columns],
            "by_type": by_type,
            "sensors": sensors,
        }
        _DATA_CACHE.update({"key": key, "value": base})
    return {**base, "quality": quality, "figures": figures}


def figure_path(name: str) -> Path:
    """Resolves a report figure by file name; only files that exist in reports/figures are served."""
    figures_dir = get_path("reports/figures")
    candidates = {p.name: p for p in figures_dir.glob("*.png")} if figures_dir.exists() else {}
    if name not in candidates:
        raise ServiceError(404, f"Figure '{name}' not found.")
    return candidates[name]


# --------------------------------------------------------------------------- model
def _feature_importance(model: Any, feature_names: List[str]) -> List[Dict[str, Any]]:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
        total = values.sum()
        values = values / total if total else values
    else:
        return []
    if len(values) != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(len(values))]
    pairs = sorted(zip(feature_names, values), key=lambda kv: kv[1], reverse=True)
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]


def _serving_state() -> Dict[str, Any]:
    """Loads the served model (cached) and returns its identity; never raises."""
    try:
        from src.api.model_loader import load_artifacts
        model, preprocessor, name, version = load_artifacts()
        return {"ok": True, "model": model, "preprocessor": preprocessor, "name": name, "version": str(version)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "model": None, "preprocessor": None, "name": None, "version": None}


def model_overview() -> Dict[str, Any]:
    evaluation = _read_json(_models_dir() / "evaluation_metrics.json") or {}
    metadata = _read_json(_models_dir() / "model_metadata.json") or {}
    state = _serving_state()
    if not state["ok"] and not evaluation:
        raise ServiceError(404, f"No trained model found. {state.get('error', '')}".strip())

    feature_names = metadata.get("feature_names") or []
    if not feature_names and state["preprocessor"] is not None:
        try:
            from src.features.build_features import get_processed_feature_names
            feature_names = get_processed_feature_names(state["preprocessor"])
        except Exception:
            feature_names = []

    thresholds = load_config().get("monitoring", {}).get("thresholds", {})
    model_path = _models_dir() / "best_model.joblib"
    return {
        "production": mlflow_reader.production_info(),
        "serving_version": state.get("version"),
        "algorithm": evaluation.get("selected_model") or metadata.get("model_name"),
        "estimator": metadata.get("algorithm") or (type(state["model"]).__name__ if state["model"] is not None else None),
        "selected_run_id": evaluation.get("selected_run_id") or metadata.get("mlflow_run_id"),
        "trained_at": metadata.get("trained_at") or _mtime_iso(model_path),
        "params": metadata.get("model_params") or {},
        "feature_names": feature_names,
        "test_metrics": evaluation.get("selected_model_test_metrics"),
        "candidates": evaluation.get("candidate_val_metrics") or {},
        "feature_importance": _feature_importance(state["model"], feature_names) if state["model"] is not None else [],
        "targets": {"min_recall": thresholds.get("min_acceptable_recall"), "min_f1": thresholds.get("min_acceptable_f1")},
    }


# --------------------------------------------------------------------------- monitoring
def _monitoring_files():
    config = load_config()
    mon = config.get("monitoring", {})
    reports = get_path(mon.get("reports_dir", "data/reports"))
    return (reports / mon.get("summary_filename", "monitoring_summary.json"),
            reports / mon.get("json_report_filename", "drift_report.json"),
            reports / "monitoring_context.json",
            reports / "monitoring_history.jsonl")


def read_monitoring_history(limit: int = 60) -> List[Dict[str, Any]]:
    path = _monitoring_files()[3]
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]


def monitoring_status() -> Dict[str, Any]:
    summary_path, report_path, context_path, _ = _monitoring_files()
    summary = _read_json(summary_path)
    report = _read_json(report_path)
    thresholds = load_config().get("monitoring", {}).get("thresholds", {})
    return {
        "available": summary is not None and report is not None,
        "summary": summary,
        "report": report,
        "context": _read_json(context_path),
        "history": read_monitoring_history(),
        "thresholds": thresholds,
        "scenarios": describe_scenarios(),
        "live_samples": len(prediction_log.as_raw_rows()),
    }


# --------------------------------------------------------------------------- retraining
def retraining_status() -> Dict[str, Any]:
    config = load_config()
    retrain_cfg = config.get("retraining", {})
    history = _read_json(_models_dir() / retrain_cfg.get("history_filename", "retraining_history.json")) or []
    summary = _read_json(_monitoring_files()[0])
    min_new = retrain_cfg.get("min_new_samples_threshold", 500)
    decision = decide_from_summary(summary, 0, min_new) if summary else None
    running = jobs.running_job()
    return {
        "history": list(reversed(history)),
        "criteria": retrain_cfg.get("promotion_criteria", {}),
        "thresholds": config.get("monitoring", {}).get("thresholds", {}),
        "min_new_samples": min_new,
        "decision": decision,
        "monitoring_summary": summary,
        "running": running if running and running.get("kind") == "retraining" else None,
    }


# --------------------------------------------------------------------------- pipeline
def _stage_states() -> List[Dict[str, Any]]:
    config = load_config()
    raw_path = get_path(config["data"]["raw_dir"]) / config["data"]["dataset_filename"]
    processed = get_path(config["data"]["processed_dir"])
    quality = _read_json(_reports_dir() / config["data"]["report_filename"])
    metadata = _read_json(_models_dir() / "model_metadata.json") or {}
    evaluation = _read_json(_models_dir() / "evaluation_metrics.json") or {}
    model_path = _models_dir() / "best_model.joblib"

    stages = []
    stages.append({
        "id": "data_ingestion", "status": "ok" if raw_path.exists() else "missing",
        "updated_at": _mtime_iso(raw_path),
        "detail": f"{quality['total_rows']:,} rows" if quality and quality.get("total_rows") else ("Dataset present" if raw_path.exists() else "Dataset not found"),
    })
    if quality:
        issues = len(quality.get("issues", []))
        stages.append({"id": "data_validation", "status": "ok" if quality.get("is_valid") else "failed",
                       "updated_at": quality.get("timestamp"),
                       "detail": "All checks passed" if quality.get("is_valid") else f"{issues} issue(s) found"})
    else:
        stages.append({"id": "data_validation", "status": "missing", "updated_at": None, "detail": "No validation report yet"})

    pre = processed / "preprocessor.joblib"
    stages.append({"id": "preprocessing", "status": "ok" if pre.exists() and (processed / "train.joblib").exists() else "missing",
                   "updated_at": _mtime_iso(pre), "detail": f"{metadata.get('num_features', '13')} features" if pre.exists() else "No processed data yet"})

    stages.append({"id": "model_training", "status": "ok" if model_path.exists() else "missing",
                   "updated_at": metadata.get("trained_at") or _mtime_iso(model_path),
                   "detail": evaluation.get("selected_model") or ("Model present" if model_path.exists() else "No trained model yet")})

    production = mlflow_reader.production_info() if model_path.exists() else None
    version = (production or {}).get("version") or metadata.get("registered_model_version") or (evaluation.get("registry_info") or {}).get("version")
    stages.append({"id": "mlflow_registration", "status": "ok" if version else "missing",
                   "updated_at": (production or {}).get("created_at"),
                   "detail": f"Version {version} @Production" if version else "Nothing registered yet"})
    return stages


def pipeline_status() -> Dict[str, Any]:
    all_jobs = jobs.list_jobs(15)
    pipeline_jobs = [j for j in all_jobs if j["kind"] == "pipeline"]
    running = jobs.running_job()
    return {
        "dag": {
            "dag_id": "predictive_maintenance_mlops_pipeline",
            "schedule_cron": "0 0 * * 0",
            "schedule_text": "Weekly, Sundays at 00:00",
            "retries": 2,
            "retry_delay_minutes": 1,
            "tasks": DAG_TASKS,
        },
        "stages": _stage_states(),
        "jobs": all_jobs,
        "last_pipeline_job": pipeline_jobs[0] if pipeline_jobs else None,
        "running": running,
        "links": _links(),
    }


# --------------------------------------------------------------------------- logs & system
_LOG_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+\[(?P<level>[A-Z]+)\]\s+\[(?P<logger>[^\]]+)\]\s+-\s+(?P<message>.*)$")


def logs_tail(lines: int = 200) -> Dict[str, Any]:
    lines = max(10, min(int(lines), 1000))
    path = get_path("logs/app.log")
    if not path.exists():
        return {"path": str(path), "lines": []}
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - 400_000))
        chunk = f.read().decode("utf-8", errors="replace")
    parsed: List[Dict[str, str]] = []
    for raw in chunk.splitlines()[-lines:]:
        raw = raw.rstrip()
        if not raw:
            continue
        m = _LOG_RE.match(raw)
        if m:
            parsed.append(m.groupdict())
        elif parsed:
            parsed[-1]["message"] += "\n" + raw
    return {"path": str(path), "lines": parsed}


def system_info() -> Dict[str, Any]:
    packages = {}
    for pkg in ["fastapi", "uvicorn", "pydantic", "mlflow", "scikit-learn", "xgboost", "pandas", "numpy", "scipy", "apache-airflow"]:
        try:
            packages[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            packages[pkg] = None
    config = load_config()
    paths = {
        "Raw dataset": get_path(config["data"]["raw_dir"]) / config["data"]["dataset_filename"],
        "Preprocessor": get_path(config["data"]["processed_dir"]) / "preprocessor.joblib",
        "Production model": _models_dir() / "best_model.joblib",
        "MLflow store": Path(re.sub(r"^sqlite:///", "", resolve_tracking_uri())),
        "Application log": get_path("logs/app.log"),
    }
    return {
        "api_version": API_VERSION,
        "project": config.get("project", {}).get("name"),
        "environment": os.getenv("ENVIRONMENT", config.get("project", {}).get("environment", "development")),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "packages": packages,
        "uptime_s": int(time.time() - _STARTED_AT),
        "started_at": pd.Timestamp(_STARTED_AT, unit="s").isoformat(),
        "paths": [{"label": k, "path": str(v), "exists": v.exists()} for k, v in paths.items()],
        "links": _links(),
        "thresholds": config.get("monitoring", {}).get("thresholds", {}),
        "promotion_criteria": config.get("retraining", {}).get("promotion_criteria", {}),
    }


# --------------------------------------------------------------------------- overview
def overview() -> Dict[str, Any]:
    state = _serving_state()
    evaluation = _read_json(_models_dir() / "evaluation_metrics.json") or {}
    metadata = _read_json(_models_dir() / "model_metadata.json") or {}
    config = load_config()
    quality = _read_json(_reports_dir() / config["data"]["report_filename"])
    summary = _read_json(_monitoring_files()[0])
    context = _read_json(_monitoring_files()[2])
    retrain = retraining_status()
    all_jobs = jobs.list_jobs(5)
    selected = evaluation.get("selected_model")

    model = None
    if state["ok"] or evaluation:
        model = {
            "name": state.get("name") or "PredictiveMaintenanceModel",
            "version": state.get("version"),
            "algorithm": selected or metadata.get("model_name"),
            "run_id": evaluation.get("selected_run_id"),
            "trained_at": metadata.get("trained_at") or _mtime_iso(_models_dir() / "best_model.joblib"),
            "test_metrics": evaluation.get("selected_model_test_metrics"),
            "val_metrics": (evaluation.get("candidate_val_metrics") or {}).get(selected),
        }

    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "service": {
            "status": "healthy" if state["ok"] else "down",
            "model_loaded": state["ok"],
            "error": None if state["ok"] else state.get("error"),
            "environment": os.getenv("ENVIRONMENT", config.get("project", {}).get("environment", "development")),
            "api_version": API_VERSION,
        },
        "model": model,
        "data": None if not quality else {
            "rows": quality.get("total_rows"),
            "failure_rate_pct": (quality.get("target_distribution") or {}).get("failure_rate_pct"),
            "quality_valid": quality.get("is_valid"),
            "quality_checked_at": quality.get("timestamp"),
            "issues": quality.get("issues", []),
        },
        "monitoring": None if not summary else {**summary, "scenario": context},
        "retraining": {
            "count": len(retrain["history"]),
            "last": retrain["history"][0] if retrain["history"] else None,
            "decision": retrain["decision"],
        },
        "pipeline": {
            "last_job": next((j for j in all_jobs if j["kind"] == "pipeline"), None),
            "running": jobs.running_job(),
            "stages": _stage_states(),
        },
        "jobs": all_jobs,
        "predictions": prediction_log.stats(),
        "thresholds": config.get("monitoring", {}).get("thresholds", {}),
    }
