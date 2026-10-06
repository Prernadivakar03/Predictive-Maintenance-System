# What changed in this version

## New: React dashboard
`frontend/` (React 19 + Vite) with nine pages: Overview, Predict, Model, Experiments, Pipeline, Monitoring, Retraining, Data, System. Light and dark themes, responsive down to phone width, no UI-framework dependencies (hand-built SVG charts). A prebuilt copy is in `frontend/dist`, so Node.js is only needed to modify it. FastAPI serves it at `/`.

## New: dashboard API (`/api/...`)
Read endpoints for model, registry, experiments, data, monitoring, retraining, pipeline, logs and system info; write endpoints to run monitoring (simulated drift, uploaded CSV or live traffic), start the pipeline, and start retraining as background jobs with captured logs (`src/api/*`). Every prediction served by `/predict` is now logged for the live feed and live-traffic drift monitoring.

## Bugs fixed
| # | Problem | Fix |
|---|---|---|
| 1 | **A rejected retrain still shipped.** `train_and_evaluate_all()` overwrote `models/best_model.joblib` *before* the promotion gate ran; rejection only moved the MLflow alias back, so the API kept serving the rejected model. | Serving files are snapshotted before training and restored on rejection or failure (`src/retraining/trigger.py`). |
| 2 | **The gate never compared against production.** It only checked an absolute recall floor (the code comment admitted this). | New `src/retraining/promotion.py` compares recall and precision with the production model's test metrics using the thresholds already in `config.yaml`. README claimed a different rule (`recall >= production - 0.02`); the docs now match the code. |
| 3 | **API served a stale model after retraining** until restarted. | `model_loader` reloads when the artifact files change; jobs also force a reload. |
| 4 | **`model_metadata.json` was documented but never written**, so it went stale. | `train.py` writes it after every training run. |
| 5 | **Airflow `mlflow_registration` task re-registered the model** after training had already done so, creating a duplicate version on every DAG run. | Task is now idempotent (verifies instead of re-registering). Also `schedule` instead of the deprecated `schedule_interval`, and Airflow 3 import path. |
| 6 | **`mlflow>=2.9.0` was wrong**: `log_model(..., name=...)` is MLflow 3 API. | `requirements.txt` now `mlflow>=3.0.0`; `scipy` added (was imported but undeclared). |
| 7 | **`Can't find a usable init.tcl`** (seen in your `logs/app.log`) from matplotlib's Tk backend; also unsafe in server threads. | `MPLBACKEND=Agg` is forced in `src/utils/config.py` and the Dockerfile. |
| 8 | **Relative MLflow URI** (`sqlite:///mlflow.db`) resolved against the *current directory*, silently creating an empty second database when started elsewhere. Logs also depended on the working directory. | `resolve_tracking_uri()` anchors it to the project root; logger paths too. |
| 9 | **Moved project folder broke training**: the experiment stores an absolute artifact path (`file:///D:/MLOOPS/mlruns/1`). | `setup_mlflow()` detects an unusable artifact location, archives that experiment (history kept) and recreates it locally. Guarded, so a failure only logs a warning. |
| 10 | Windows console crashed log lines containing symbols (cp1252). | Console stream set to `errors="replace"`. |
| 11 | `/predict` accepted an empty list and failed with a 500. | `inputs` must contain 1-1000 items (422 otherwise). |
| 12 | Docker: XGBoost needs `libgomp1` (missing on `python:3.11-slim`); API container did not mount `mlflow.db`, so it reported version "1"; `.dockerignore` sent ~1 GB (`mlruns`, `node_modules`) to the daemon; compose `version:` is obsolete; MLflow container ran `pip install mlflow` on every start. | All fixed in `docker/Dockerfile.api`, `docker-compose.yml`, `.dockerignore`. |
| 13 | `.gitignore` patterns `lib/` and `dist/` matched *any* folder of that name (would drop `frontend/src/lib` from git). | Anchored to the repository root. |
| 14 | README mismatches: DAG id, retry delay, `Target` column name, API status label, monitoring file names, retraining rule, rollback procedure (an alias change alone does not change the served model). | README sections corrected. |
| 15 | CORS used `*` together with `allow_credentials=True` (invalid combination). | Credentials off; origins configurable via `CORS_ORIGINS`. |

## Known limitations
* **Retraining reuses the same processed training data.** Drift and sample counts decide *when* to retrain, but newly labeled batches are not merged into the training set yet. This is the best next improvement.
* The registry holds many identical versions because the test suite runs the real pipeline each time (see RUN_GUIDE section 7).
* The Airflow service in `docker-compose.yml` (profile `airflow`) is experimental.
