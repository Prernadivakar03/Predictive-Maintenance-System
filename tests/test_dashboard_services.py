"""Tests for the dashboard service layer (promotion gate, scenarios, prediction log, jobs, MLflow reader)."""
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from src.api import jobs, mlflow_reader, prediction_log
from src.api.errors import ServiceError
from src.api.timeutil import localize_timestamps
from src.monitoring import scenarios
from src.retraining.promotion import evaluate_promotion
from src.retraining.trigger import decide_from_summary
from src.utils.config import resolve_tracking_uri, PROJECT_ROOT

CRITERIA = {"min_recall_threshold": 0.75, "min_recall_improvement_pct": 0.0, "max_precision_drop_pct": 5.0}
OLD = {"recall": 0.8431, "precision": 0.6515}


# ---------------------------------------------------------------- promotion gate
def test_gate_promotes_identical_candidate():
    result = evaluate_promotion(OLD, dict(OLD), CRITERIA)
    assert result["promote"] and result["status"] == "PROMOTED"


def test_gate_rejects_recall_regression():
    result = evaluate_promotion(OLD, {"recall": 0.80, "precision": 0.70}, CRITERIA)
    assert not result["promote"] and result["status"] == "REJECTED_INFERIOR"
    assert any(c["name"] == "Recall vs production" and not c["passed"] for c in result["checks"])


def test_gate_rejects_precision_collapse():
    result = evaluate_promotion(OLD, {"recall": 0.90, "precision": 0.55}, CRITERIA)
    assert not result["promote"]
    assert any(c["name"] == "Precision regression" and not c["passed"] for c in result["checks"])


def test_gate_applies_floor_without_previous_model():
    assert not evaluate_promotion(None, {"recall": 0.70, "precision": 0.9}, CRITERIA)["promote"]
    assert evaluate_promotion(None, {"recall": 0.80, "precision": 0.9}, CRITERIA)["promote"]


# ---------------------------------------------------------------- retraining decision
def test_decide_from_summary():
    quiet = {"data_drift_detected": False, "prediction_drift_detected": False, "performance_degraded": False}
    assert decide_from_summary(quiet)["status"] == "NO_RETRAIN_REQUIRED"
    drifted = {**quiet, "data_drift_detected": True, "drifted_features_count": 4}
    decision = decide_from_summary(drifted)
    assert decision["retrain_required"] and "Drift" in decision["trigger_reason"]
    assert decide_from_summary(quiet, new_samples=600, min_new_samples=500)["retrain_required"]


# ---------------------------------------------------------------- helpers
def test_localize_timestamps_adds_offset_only_to_naive_strings():
    out = localize_timestamps({"a": "2026-10-02T09:01:14.100547", "b": ["2026-10-02T09:01:14+00:00", "text"], "c": 1})
    assert out["a"][-6] in "+-" or out["a"].endswith("Z")
    assert out["b"] == ["2026-10-02T09:01:14+00:00", "text"] and out["c"] == 1


def test_resolve_tracking_uri_anchors_relative_sqlite_paths():
    uri = resolve_tracking_uri("sqlite:///mlflow.db")
    assert uri.startswith("sqlite:///") and uri.endswith("mlflow.db")
    assert PROJECT_ROOT.as_posix() in uri
    assert resolve_tracking_uri("sqlite:////abs/path/mlflow.db") == "sqlite:////abs/path/mlflow.db"
    assert resolve_tracking_uri("http://mlflow:5000") == "http://mlflow:5000"


# ---------------------------------------------------------------- scenarios
def _row(**over):
    base = {"Type": "M", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551, "Torque [Nm]": 42.8, "Tool wear [min]": 0}
    base.update(over)
    return base


def test_rows_to_raw_frame_accepts_snake_case_and_csv_strings():
    df = scenarios.rows_to_raw_frame([{"type": "l", "air_temperature": "298.1", "process_temperature": "308.6",
                                       "rotational_speed": "1500", "torque": "40", "tool_wear": "10", "machine_failure": "1"}])
    assert df.loc[0, "Type"] == "L" and df.loc[0, "Machine failure"] == 1


def test_rows_to_raw_frame_rejects_bad_input():
    for rows in ([], [{"Type": "M"}], [_row(Type="Z")], [_row(**{"Torque [Nm]": "abc"})], [_row(**{"Air temperature [K]": 999})]):
        with pytest.raises(ValueError):
            scenarios.rows_to_raw_frame(rows)


def test_apply_shift_moves_expected_columns_within_valid_ranges():
    raw = pd.DataFrame([_row(), _row(**{"Torque [Nm]": 70})])
    shifted = scenarios.apply_shift(raw, "sensor_drift", 1.0)
    assert (shifted["Torque [Nm]"] > raw["Torque [Nm]"]).all()
    assert (shifted["Rotational speed [rpm]"] < raw["Rotational speed [rpm]"]).all()
    assert (shifted["Tool wear [min]"] == raw["Tool wear [min]"]).all()
    assert (scenarios.apply_shift(raw, "wear_shift", 0.0)["Tool wear [min]"] == raw["Tool wear [min]"]).all()
    with pytest.raises(ValueError):
        scenarios.apply_shift(raw, "baseline", 0.5)


# ---------------------------------------------------------------- prediction log
@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_log, "get_path", lambda rel: tmp_path / rel)
    return tmp_path


def test_prediction_log_roundtrip(isolated_log):
    assert prediction_log.stats()["total"] == 0 and prediction_log.recent() == []
    results = [{"prediction": 1, "failure_probability": 0.93, "model_version": "7"},
               {"prediction": 0, "failure_probability": 0.02, "model_version": "7"}]
    prediction_log.log_predictions([_row(), _row(Type="L")], results)
    stats = prediction_log.stats()
    assert stats["total"] == 2 and stats["failures"] == 1 and stats["failure_rate_pct"] == 50.0
    assert sum(stats["probability_histogram"]) == 2 and len(stats["hourly"]) == 24
    assert prediction_log.recent(1)[0]["Type"] == "L"  # newest first
    assert len(prediction_log.as_raw_rows()) == 2


def test_prediction_log_never_raises(isolated_log):
    prediction_log.log_predictions([{"bad": 1}], [{"nope": 1}])  # malformed input is swallowed
    assert prediction_log.stats()["total"] == 0


# ---------------------------------------------------------------- job runner
@pytest.fixture
def clean_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "get_path", lambda rel: tmp_path / rel)
    monkeypatch.setattr(jobs, "_JOBS", [])
    monkeypatch.setattr(jobs, "_LOADED", False)
    monkeypatch.setattr(jobs, "_reload_serving_model", lambda: None)


def _wait(job_id, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        job = jobs.get_job(job_id)
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_job_success_captures_stage_and_logs(clean_jobs):
    from src.utils.logger import get_logger

    def work():
        get_logger("test_job_logger").info("\n>>> STAGE 2: Data Validation")
        return {"ok": True}

    job = _wait(jobs.start_job("pipeline", work, {})["id"])
    assert job["status"] == "succeeded" and job["result"] == {"ok": True}
    assert any("STAGE 2" in line["message"] for line in job["log"])
    assert job["duration_s"] is not None


def test_job_failure_is_recorded(clean_jobs):
    def boom():
        raise RuntimeError("boom")

    job = _wait(jobs.start_job("retraining", boom, {})["id"])
    assert job["status"] == "failed" and "boom" in job["error"]


def test_only_one_job_runs_at_a_time(clean_jobs):
    first = jobs.start_job("pipeline", lambda: time.sleep(0.4), {})
    with pytest.raises(ServiceError) as exc:
        jobs.start_job("pipeline", lambda: None, {})
    assert exc.value.status_code == 409
    _wait(first["id"])
    assert jobs.running_job() is None
    with pytest.raises(ServiceError) as missing:
        jobs.get_job("does-not-exist")
    assert missing.value.status_code == 404


# ---------------------------------------------------------------- MLflow reader (pure conversions)
def test_run_and_version_conversion():
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="abc123def456", run_name="RF_Run", status="FINISHED", start_time=1_760_000_000_000, end_time=1_760_000_005_400),
        data=SimpleNamespace(metrics={"val_recall": 0.82351, "unrelated": 1.0}, params={"model_type": "Random Forest"}, tags={"stage": "Production"}),
    )
    d = mlflow_reader.run_to_dict(run)
    assert d["algorithm"] == "Random Forest" and d["duration_s"] == 5.4
    assert d["metrics"] == {"val_recall": 0.8235} and d["started_at"].endswith("+00:00")

    mv = SimpleNamespace(version=3, run_id="abc123def456", status="READY", creation_timestamp=1_760_000_000_000, description="x", aliases=["Production"])
    v = mlflow_reader.version_to_dict(mv, {"abc123def456": d["metrics"]}, {"abc123def456": "Random Forest"})
    assert v["version"] == "3" and v["aliases"] == ["Production"] and v["metrics"]["val_recall"] == 0.8235
