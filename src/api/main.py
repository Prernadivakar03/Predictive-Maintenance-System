import os
import sys
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.api.schemas import PredictRequest, PredictResponse, HealthResponse
from src.api.model_loader import load_artifacts, run_inference
from src.api.routes import router as dashboard_router
from src.api import prediction_log

logger = get_logger("api_main")

# Built React dashboard (npm run build -> frontend/dist). Served by this same process when present.
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", str(PROJECT_ROOT / "frontend" / "dist")))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan context manager for pre-loading model artifacts on startup."""
    logger.info("Initializing FastAPI Application & Pre-loading Model Artifacts...")
    try:
        _, _, model_name, model_version = load_artifacts()
        logger.info(f"FastAPI Startup Complete. Serving '{model_name}' Version {model_version}.")
    except Exception as e:
        logger.error(f"Failed to load model artifacts on startup: {str(e)}")
    if (FRONTEND_DIST / "index.html").is_file():
        logger.info(f"Serving dashboard UI from {FRONTEND_DIST}")
    else:
        logger.info("Dashboard UI build not found (run 'npm run build' in frontend/). API-only mode.")
    yield
    logger.info("Shutting down FastAPI Application...")

# Initialize FastAPI App
app = FastAPI(
    title="Predictive Maintenance MLOps API",
    description="Production REST API for real-time equipment failure prediction and sensor telemetry monitoring.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware. Set CORS_ORIGINS="http://localhost:5173,https://ui.example.com" to restrict origins in production.
_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)

if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="ui-assets")

@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    icon = FRONTEND_DIST / "favicon.svg"
    if icon.is_file():
        return FileResponse(icon, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Not found")

@app.get("/", tags=["General"])
def read_root(request: Request):
    """Browsers get the dashboard UI (when built); API clients get the JSON service index."""
    index = FRONTEND_DIST / "index.html"
    if index.is_file() and "text/html" in request.headers.get("accept", ""):
        return FileResponse(index)
    return {
        "service": "Predictive Maintenance MLOps API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint.
    Returns service status and active production model version details.
    """
    try:
        _, _, model_name, model_version = load_artifacts()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=model_name,
            model_version=model_version,
            timestamp=pd.Timestamp.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_equipment_failure(request: PredictRequest):
    """
    Real-time Inference Endpoint.
    Accepts list of machine telemetry features and returns failure predictions with probabilities.
    """
    logger.info(f"Received prediction request for {len(request.inputs)} telemetry sample(s).")
    try:
        telemetry_dicts = [sample.model_dump(by_alias=True) for sample in request.inputs]
        predictions = run_inference(telemetry_dicts)
        logger.info(f"Successfully generated {len(predictions)} prediction(s).")
        # Persist for the dashboard's live feed and for 'live traffic' drift monitoring (never raises)
        prediction_log.log_predictions(telemetry_dicts, predictions)
        return PredictResponse(success=True, predictions=predictions)
    except Exception as e:
        logger.error(f"Error during prediction execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", 8000))
    logger.info(f"Starting FastAPI Uvicorn Server at http://{host}:{port}...")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True)
