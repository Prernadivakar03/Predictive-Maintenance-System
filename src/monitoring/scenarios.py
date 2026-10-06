"""
Builds the "current production batch" used by the monitoring suite from raw sensor records.

The drift detector compares engineered/scaled features, so every batch (simulated shift, uploaded CSV rows,
or logged live predictions) is pushed through the *fitted production preprocessor* first.
"""
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, get_path
from src.features.build_features import split_data, get_processed_feature_names

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
REQUIRED_COLUMNS = ["Type"] + SENSOR_COLUMNS

SNAKE_TO_RAW = {
    "type": "Type",
    "air_temperature": "Air temperature [K]",
    "process_temperature": "Process temperature [K]",
    "rotational_speed": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
    "machine_failure": "Machine failure",
}

MAX_BATCH_ROWS = 20000

SCENARIOS: Dict[str, Dict[str, str]] = {
    "baseline": {
        "label": "Held-out validation set",
        "description": "Unchanged validation data. Expected result: no drift.",
    },
    "live": {
        "label": "Live API traffic",
        "description": "Sensor readings received by the /predict endpoint (no ground-truth labels).",
    },
    "sensor_drift": {
        "label": "Mechanical drift",
        "description": "Torque rises and spindle speed falls, as with a harder material batch.",
    },
    "thermal_drift": {
        "label": "Thermal drift",
        "description": "Ambient and process temperatures creep upward.",
    },
    "wear_shift": {
        "label": "Tool wear shift",
        "description": "Tools are run much longer before replacement.",
    },
    "combined": {
        "label": "Combined drift",
        "description": "Mechanical, thermal and tool wear shifts together.",
    },
    "upload": {
        "label": "Uploaded batch",
        "description": "Raw sensor rows supplied by you (optionally with a 'Machine failure' label column).",
    },
}


def describe_scenarios() -> List[Dict[str, str]]:
    return [{"id": key, **value} for key, value in SCENARIOS.items()]


def apply_shift(raw: pd.DataFrame, scenario: str, severity: float) -> pd.DataFrame:
    """Applies a synthetic distribution shift (severity 0..1) to raw sensor columns."""
    if scenario not in ("sensor_drift", "thermal_drift", "wear_shift", "combined"):
        raise ValueError(f"Scenario '{scenario}' is not a simulated shift.")

    df = raw.copy()
    sev = float(np.clip(severity, 0.0, 1.0))
    if scenario == "combined":
        sev *= 0.75

    if scenario in ("sensor_drift", "combined"):
        df["Torque [Nm]"] = df["Torque [Nm]"] * (1 + 0.45 * sev)
        df["Rotational speed [rpm]"] = df["Rotational speed [rpm]"] * (1 - 0.20 * sev)
    if scenario in ("thermal_drift", "combined"):
        df["Air temperature [K]"] = df["Air temperature [K]"] + 6.0 * sev
        df["Process temperature [K]"] = df["Process temperature [K]"] + 9.0 * sev
    if scenario in ("wear_shift", "combined"):
        df["Tool wear [min]"] = df["Tool wear [min]"] + 90.0 * sev

    ranges = load_config().get("validation", {}).get("numerical_ranges", {})
    for col, limits in ranges.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=limits.get("min"), upper=limits.get("max"))
    return df


def load_raw_validation_frame() -> pd.DataFrame:
    """Re-creates the raw rows of the held-out validation split (same seed + stratification as the pipeline)."""
    config = load_config()
    raw_path = get_path(config["data"]["raw_dir"]) / config["data"]["dataset_filename"]
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at '{raw_path}'. Run the pipeline first.")
    raw = pd.read_csv(raw_path)
    _, df_val, _ = split_data(raw, target_col=config["validation"]["target_column"])
    return df_val.reset_index(drop=True)


def rows_to_raw_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Validates and normalizes user-supplied rows (JSON/CSV) into the raw sensor schema."""
    config = load_config()
    target = config["validation"]["target_column"]

    if not rows:
        raise ValueError("No rows supplied.")
    if len(rows) > MAX_BATCH_ROWS:
        raise ValueError(f"Too many rows ({len(rows)}). Maximum is {MAX_BATCH_ROWS}.")

    df = pd.DataFrame(rows)
    df = df.rename(columns={c: SNAKE_TO_RAW.get(str(c).strip().lower(), str(c).strip()) for c in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}.")

    for col in SENSOR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_numeric = int(df[SENSOR_COLUMNS].isna().any(axis=1).sum())
    if bad_numeric:
        raise ValueError(f"{bad_numeric} row(s) contain empty or non-numeric sensor values.")

    df["Type"] = df["Type"].astype(str).str.strip().str.upper()
    allowed = config["validation"].get("categorical_values", {}).get("Type", ["L", "M", "H"])
    bad_type = int((~df["Type"].isin(allowed)).sum())
    if bad_type:
        raise ValueError(f"{bad_type} row(s) have an invalid 'Type' (allowed: {', '.join(allowed)}).")

    for col, limits in config["validation"].get("numerical_ranges", {}).items():
        if col in df.columns:
            outside = int(((df[col] < limits["min"]) | (df[col] > limits["max"])).sum())
            if outside:
                raise ValueError(f"{outside} row(s) have '{col}' outside [{limits['min']}, {limits['max']}].")

    columns = REQUIRED_COLUMNS.copy()
    if target in df.columns:
        labels = pd.to_numeric(df[target], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError(f"'{target}' must contain only 0 or 1.")
        df[target] = labels.astype(int)
        columns.append(target)
    return df[columns].reset_index(drop=True)


def transform_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Pushes raw rows through the fitted production preprocessor; keeps the target column when present."""
    config = load_config()
    target = config["validation"]["target_column"]
    preprocessor_path = get_path(config["data"]["processed_dir"]) / "preprocessor.joblib"
    if not preprocessor_path.exists():
        raise FileNotFoundError("Preprocessor artifact missing. Run the pipeline first.")

    preprocessor = joblib.load(preprocessor_path)
    features = preprocessor.transform(raw_df.drop(columns=[target], errors="ignore"))
    out = pd.DataFrame(features, columns=get_processed_feature_names(preprocessor))
    if target in raw_df.columns:
        out[target] = raw_df[target].astype(int).values
    return out


def build_scenario_batch(scenario: str, severity: float = 0.5) -> Optional[pd.DataFrame]:
    """Returns the processed batch for a simulated scenario, or None for 'baseline' (suite default)."""
    if scenario == "baseline":
        return None
    raw_val = load_raw_validation_frame()
    return transform_raw(apply_shift(raw_val, scenario, severity))
