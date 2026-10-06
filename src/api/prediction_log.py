"""
Append-only log of predictions served by the API (JSON Lines).

It powers the dashboard's live-prediction feed and lets the monitoring suite analyse *real* production
traffic (the 'live' scenario). The file is size-bounded so it cannot grow without limit.
"""
import json
import threading
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

from src.utils.config import get_path

_LOCK = threading.Lock()
MAX_BYTES = 2_000_000
KEEP_LINES = 5000

SENSOR_KEYS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
_SNAKE = {
    "air_temperature": "Air temperature [K]",
    "process_temperature": "Process temperature [K]",
    "rotational_speed": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
}


def _log_path() -> Path:
    return get_path("data/predictions/predictions.jsonl")


def log_predictions(inputs: List[Dict[str, Any]], results: List[Dict[str, Any]], source: str = "api") -> None:
    """Appends one JSON line per prediction. Never raises into the inference path."""
    try:
        lines = []
        for inp, res in zip(inputs, results):
            norm = {_SNAKE.get(k, k): v for k, v in inp.items()}
            record = {
                "timestamp": res.get("timestamp") or pd.Timestamp.now().isoformat(),
                "Type": norm.get("Type"),
                **{k: norm.get(k) for k in SENSOR_KEYS},
                "prediction": int(res["prediction"]),
                "failure_probability": float(res["failure_probability"]),
                "model_version": str(res.get("model_version", "")),
                "source": source,
            }
            lines.append(json.dumps(record))
        if not lines:
            return
        path = _log_path()
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            if path.stat().st_size > MAX_BYTES:
                kept = path.read_text(encoding="utf-8").splitlines()[-KEEP_LINES:]
                path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    except Exception:
        pass


def _read_all() -> List[Dict[str, Any]]:
    path = _log_path()
    if not path.exists():
        return []
    records = []
    with _LOCK:
        text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def recent(limit: int = 50) -> List[Dict[str, Any]]:
    """Newest predictions first."""
    limit = max(1, min(int(limit), 500))
    return list(reversed(_read_all()[-limit:]))


def stats() -> Dict[str, Any]:
    records = _read_all()
    total = len(records)
    failures = sum(1 for r in records if r.get("prediction") == 1)
    probs = [r["failure_probability"] for r in records if "failure_probability" in r]

    bins = [0] * 10
    for p in probs:
        bins[min(int(p * 10), 9)] += 1

    # Hourly volume for the last 24 hours (naive local timestamps, as written by the API)
    now = pd.Timestamp.now().floor("h")
    buckets = {now - pd.Timedelta(hours=h): {"total": 0, "failures": 0} for h in range(23, -1, -1)}
    for r in records:
        try:
            hour = pd.Timestamp(r["timestamp"]).floor("h")
        except Exception:
            continue
        if hour in buckets:
            buckets[hour]["total"] += 1
            buckets[hour]["failures"] += 1 if r.get("prediction") == 1 else 0

    return {
        "total": total,
        "failures": failures,
        "failure_rate_pct": round(failures / total * 100, 2) if total else 0.0,
        "avg_probability": round(sum(probs) / len(probs), 4) if probs else None,
        "last_prediction_at": records[-1]["timestamp"] if records else None,
        "probability_histogram": bins,
        "hourly": [{"hour": h.isoformat(), **v} for h, v in buckets.items()],
    }


def as_raw_rows(limit: int = 5000) -> List[Dict[str, Any]]:
    """Logged inputs in raw-sensor schema (no labels) for the 'live' monitoring scenario."""
    rows = []
    for r in _read_all()[-limit:]:
        if r.get("Type") is not None and all(r.get(k) is not None for k in SENSOR_KEYS):
            rows.append({"Type": r["Type"], **{k: r[k] for k in SENSOR_KEYS}})
    return rows
