import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Force a headless matplotlib backend BEFORE anything imports pyplot.
# Without this, training can crash with "Can't find a usable init.tcl" on Windows
# (Tk backend) and is unsafe when training runs inside an API/background thread.
os.environ.setdefault("MPLBACKEND", "Agg")

# Base Directory Resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env if present
load_dotenv(PROJECT_ROOT / ".env")

def load_config(config_path: Path = None) -> dict:
    """
    Loads project YAML configuration file.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
        
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config

def get_path(relative_path: str) -> Path:
    """
    Resolves relative path to absolute Path object from project root.
    """
    return PROJECT_ROOT / relative_path

def resolve_tracking_uri(uri: str = None) -> str:
    """
    Returns the MLflow tracking URI with relative SQLite paths anchored to the project root.

    A bare ``sqlite:///mlflow.db`` is resolved against the *current working directory* by
    SQLAlchemy, which silently creates a second, empty database whenever the API or a
    scheduler is started from another folder. Anchoring it to PROJECT_ROOT makes the
    CLI, the API, Airflow and the tests all talk to the same tracking store.
    """
    if uri is None:
        uri = os.getenv("MLFLOW_TRACKING_URI") or load_config().get("mlflow", {}).get(
            "tracking_uri", "sqlite:///mlflow.db"
        )
    match = re.match(r"^sqlite:///(?!/)(.+)$", uri)
    if match:
        raw_path = match.group(1)
        is_absolute = raw_path.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", raw_path)
        if not is_absolute:
            return "sqlite:///" + (PROJECT_ROOT / raw_path).as_posix()
    return uri
