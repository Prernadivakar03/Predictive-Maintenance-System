@echo off
REM One-click launcher for Windows: sets up Python, trains a model if needed, starts API + dashboard.
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10-3.12 from https://www.python.org and tick "Add python.exe to PATH".
  goto :fail
)

if not exist .venv\Scripts\python.exe (
  echo [1/4] Creating virtual environment...
  python -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat

echo [2/4] Installing Python dependencies (first run takes a few minutes)...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt || goto :fail

if not exist models\best_model.joblib (
  echo [3/4] No trained model found - running the ML pipeline once...
  python -m src.pipeline || goto :fail
) else (
  echo [3/4] Trained model found - skipping training.
)

echo [4/4] Starting server...
echo.
echo   Dashboard : http://127.0.0.1:8000
echo   API docs  : http://127.0.0.1:8000/docs
echo   Stop with Ctrl+C
echo.
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
goto :eof

:fail
echo.
echo Setup failed - read the message above. See RUN_GUIDE.md ^> Troubleshooting.
pause
exit /b 1
