#!/usr/bin/env bash
# One-command launcher for macOS / Linux: sets up Python, trains a model if needed, starts API + dashboard.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "Python 3.10-3.12 not found."; exit 1; }

if [ ! -x .venv/bin/python ]; then
  echo "[1/4] Creating virtual environment..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/4] Installing Python dependencies (first run takes a few minutes)..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

if [ ! -f models/best_model.joblib ]; then
  echo "[3/4] No trained model found - running the ML pipeline once..."
  python -m src.pipeline
else
  echo "[3/4] Trained model found - skipping training."
fi

echo "[4/4] Starting server..."
echo
echo "  Dashboard : http://127.0.0.1:8000"
echo "  API docs  : http://127.0.0.1:8000/docs"
echo "  Stop with Ctrl+C"
echo
exec python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
