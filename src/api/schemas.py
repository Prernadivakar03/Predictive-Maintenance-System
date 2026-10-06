from pydantic import BaseModel, Field, ConfigDict
from typing import List, Literal, Optional, Dict, Any

class TelemetryInput(BaseModel):
    """
    Schema for raw machine sensor telemetry input.
    Supports both standard column names and snake_case aliases.
    """
    Type: Literal["L", "M", "H"] = Field(
        ...,
        description="Equipment quality type: 'L' (Low), 'M' (Medium), or 'H' (High)"
    )
    air_temperature: float = Field(
        ...,
        alias="Air temperature [K]",
        ge=250.0,
        le=350.0,
        description="Ambient air temperature in Kelvin (250K - 350K)"
    )
    process_temperature: float = Field(
        ...,
        alias="Process temperature [K]",
        ge=250.0,
        le=370.0,
        description="Process operating temperature in Kelvin (250K - 370K)"
    )
    rotational_speed: float = Field(
        ...,
        alias="Rotational speed [rpm]",
        ge=0.0,
        le=5000.0,
        description="Motor shaft rotational speed in rpm (0 - 5000 rpm)"
    )
    torque: float = Field(
        ...,
        alias="Torque [Nm]",
        ge=0.0,
        le=200.0,
        description="Motor torque in Newton-meters (0 - 200 Nm)"
    )
    tool_wear: float = Field(
        ...,
        alias="Tool wear [min]",
        ge=0.0,
        le=500.0,
        description="Cumulative tool wear in minutes (0 - 500 min)"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Type": "M",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551.0,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 0.0
            }
        }
    )

class PredictRequest(BaseModel):
    """Container for batch or single telemetry prediction requests (1 to 1000 samples)."""
    inputs: List[TelemetryInput] = Field(..., min_length=1, max_length=1000)

class SinglePredictionResult(BaseModel):
    """Single prediction output model."""
    prediction: int = Field(..., description="Binary failure target: 0 = Normal, 1 = Failure")
    status: str = Field(..., description="Human-readable status ('Normal Operation' or 'Equipment Failure Warning')")
    failure_probability: float = Field(..., description="Predicted failure probability (0.0 to 1.0)")
    model_name: str = Field("PredictiveMaintenanceModel", description="Registered model name")
    model_version: str = Field("1", description="Registered model version")
    timestamp: str = Field(..., description="ISO 8601 prediction timestamp")

class PredictResponse(BaseModel):
    """Prediction API response schema."""
    success: bool = True
    predictions: List[SinglePredictionResult]

class HealthResponse(BaseModel):
    """Service health check schema."""
    status: str = "healthy"
    model_loaded: bool
    model_name: str
    model_version: str
    timestamp: str


class MonitoringRunRequest(BaseModel):
    """Request to run the drift / performance monitoring suite on a chosen batch."""
    scenario: str = Field("baseline", description="baseline | live | sensor_drift | thermal_drift | wear_shift | combined | upload")
    severity: float = Field(0.5, ge=0.0, le=1.0, description="Strength of the simulated shift (simulated scenarios only)")
    rows: Optional[List[Dict[str, Any]]] = Field(None, description="Raw sensor rows for scenario='upload' (max 20,000)")


class PipelineRunRequest(BaseModel):
    """Request to start the end-to-end ML pipeline as a background job."""
    force: bool = Field(False, description="Re-run every stage even if cached artifacts are up to date")


class RetrainRequest(BaseModel):
    """Request to start automated retraining as a background job."""
    mode: Literal["auto", "force"] = Field(
        "auto",
        description="'auto' retrains only if the latest monitoring report fired a trigger; 'force' always retrains"
    )
