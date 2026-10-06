"""
State-changing operations exposed to the dashboard: run monitoring, start pipeline / retraining jobs,
reload the served model.
"""
import json
from typing import Any, Dict, List, Optional

import pandas as pd

from src.api import insights, jobs, prediction_log
from src.api.errors import ServiceError
from src.monitoring import scenarios
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("api_operations")

SIMULATED = ("sensor_drift", "thermal_drift", "wear_shift", "combined")
MIN_LIVE_SAMPLES = 30


def run_monitoring(scenario: str = "baseline", severity: float = 0.5, rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Runs the drift / performance monitoring suite on the chosen batch and stores the result."""
    from src.monitoring.drift_detector import run_monitoring_suite

    if scenario not in scenarios.SCENARIOS:
        raise ServiceError(400, f"Unknown scenario '{scenario}'. Choose one of: {', '.join(scenarios.SCENARIOS)}.")
    if jobs.running_job():
        raise ServiceError(409, "A pipeline or retraining job is running. Run monitoring after it finishes.")

    try:
        curr_df: Optional[pd.DataFrame] = None
        n_rows: Optional[int] = None
        if scenario == "upload":
            raw = scenarios.rows_to_raw_frame(rows or [])
            curr_df, n_rows = scenarios.transform_raw(raw), len(raw)
        elif scenario == "live":
            live_rows = prediction_log.as_raw_rows()
            if len(live_rows) < MIN_LIVE_SAMPLES:
                raise ServiceError(400, f"Only {len(live_rows)} live prediction(s) logged. At least {MIN_LIVE_SAMPLES} are needed for a meaningful drift test; send more requests to /predict or use the batch tool on the Predict page.")
            raw = scenarios.rows_to_raw_frame(live_rows)
            curr_df, n_rows = scenarios.transform_raw(raw), len(raw)
        elif scenario in SIMULATED:
            severity = max(0.0, min(float(severity), 1.0))
            curr_df = scenarios.build_scenario_batch(scenario, severity)
            n_rows = len(curr_df) if curr_df is not None else None
        report = run_monitoring_suite(curr_df=curr_df)
    except ServiceError:
        raise
    except ValueError as exc:
        raise ServiceError(400, str(exc))
    except FileNotFoundError as exc:
        raise ServiceError(409, f"{exc} Run the pipeline first.")
    except Exception as exc:
        logger.error(f"Monitoring run failed: {exc}")
        raise ServiceError(500, f"Monitoring run failed: {exc}")

    summary_path, _, context_path, history_path = insights._monitoring_files()
    context = {
        "scenario": scenario,
        "label": scenarios.SCENARIOS[scenario]["label"],
        "severity": severity if scenario in SIMULATED else None,
        "rows": n_rows,
        "ran_at": pd.Timestamp.now().isoformat(),
    }
    try:
        context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**report["summary"], "scenario": scenario, "label": context["label"], "severity": context["severity"]}) + "\n")
    except OSError as exc:
        logger.warning(f"Could not persist monitoring context/history: {exc}")
    return insights.monitoring_status()


def start_pipeline(force: bool = False) -> Dict[str, Any]:
    from src.pipeline import run_pipeline

    def target():
        result = run_pipeline(force=force)
        if result.get("pipeline_status") != "SUCCESS":  # run_pipeline raises on failure; belt and braces
            raise RuntimeError(result.get("error") or "Pipeline did not complete successfully.")
        return result

    return jobs.start_job("pipeline", target, {"force": force})


def start_retraining(mode: str = "auto") -> Dict[str, Any]:
    """
    mode='force': retrain immediately (manual override).
    mode='auto' : evaluate the latest monitoring report first and only retrain if a trigger fired.
    """
    from src.retraining.trigger import execute_automated_retraining, decide_from_summary

    if mode not in ("auto", "force"):
        raise ServiceError(400, "mode must be 'auto' or 'force'.")

    min_new = load_config().get("retraining", {}).get("min_new_samples_threshold", 500)

    def target():
        if mode == "force":
            return execute_automated_retraining(force=True)
        summary = insights._read_json(insights._monitoring_files()[0])
        if not summary:
            return {"status": "SKIPPED", "reason": "No monitoring report exists yet. Run monitoring first."}
        decision = decide_from_summary(summary, 0, min_new)
        return execute_automated_retraining(decision=decision)

    return jobs.start_job("retraining", target, {"mode": mode})


def reload_model() -> Dict[str, Any]:
    from src.api.model_loader import load_artifacts
    try:
        _, _, name, version = load_artifacts(force_reload=True)
    except FileNotFoundError as exc:
        raise ServiceError(404, str(exc))
    return {"reloaded": True, "model_name": name, "model_version": version}
