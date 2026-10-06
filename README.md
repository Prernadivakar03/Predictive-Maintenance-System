# Predictive Maintenance MLOps System

An end-to-end, production-ready Machine Learning Operations (MLOps) system designed to predict industrial equipment failures from telemetry sensor data, automate dataset validation and feature engineering, track experiments via MLflow, deploy production models using FastAPI, orchestrate workflows with Apache Airflow, containerize with Docker, detect data and model drift, and execute automated retraining with safety promotion gates.

---
#LIVE DEPLOYED LINK : https://predictive-maintenance-system-sp09.onrender.com/

## Quick Start

```bash
# Windows:  start.bat          macOS/Linux:  ./start.sh
# then open  http://127.0.0.1:8000
```
Full instructions (venv, MLflow UI, Airflow, Docker, frontend development, troubleshooting): **[RUN_GUIDE.md](RUN_GUIDE.md)**.
What was fixed and added in this version: **[CHANGES.md](CHANGES.md)**.

## Web Dashboard (React)

A production-style operations console lives in `frontend/` (React + Vite, no UI framework dependencies) and is served by the same FastAPI process at `/`.

| Page | Purpose |
|---|---|
| **Overview** | Status lamps for data, model, drift, pipeline and retraining; production metrics; live prediction feed |
| **Predict** | Single-machine failure prediction with a risk dial, physical failure-mode checks, and CSV batch scoring |
| **Model** | Metrics, candidate comparison, confusion matrices, feature importance, MLflow registry versions |
| **Experiments** | Sortable MLflow run table and metric history |
| **Pipeline** | DAG view, one-click pipeline runs with live logs, job history, data-quality report |
| **Monitoring** | KS-test feature drift, prediction shift, performance vs labels; simulate drift or upload a batch |
| **Retraining** | Trigger rules, promotion safety gate, retraining history with per-check results |
| **Data** | Class balance, failure modes, sensor distributions, EDA figures |
| **System** | Health, versions, thresholds, application log |

Dashboard API (all under `/api`, documented at `/docs`): `overview`, `model`, `registry/versions`, `experiments/runs`, `data/summary`, `monitoring/status`, `monitoring/run` (POST), `retraining/status`, `retraining/trigger` (POST), `pipeline/status`, `pipeline/run` (POST), `jobs`, `jobs/{id}`, `predictions/recent`, `predictions/stats`, `logs`, `system`, `model/reload` (POST), `figures/{name}`.

## Table of Contents
1. [Project Title](#1-project-title)
2. [Project Overview](#2-project-overview)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [Dataset Description](#5-dataset-description)
6. [Features Used](#6-features-used)
7. [Machine Learning Approach](#7-machine-learning-approach)
8. [Model Comparison](#8-model-comparison)
9. [Final Model](#9-final-model)
10. [MLOps Architecture](#10-mlops-architecture)
11. [Data Pipeline](#11-data-pipeline)
12. [MLflow Experiment Tracking](#12-mlflow-experiment-tracking)
13. [Airflow Workflow Scheduling](#13-airflow-workflow-scheduling)
14. [FastAPI Deployment](#14-fastapi-deployment)
15. [Docker Setup](#15-docker-setup)
16. [Monitoring System](#16-monitoring-system)
17. [Data Drift Detection](#17-data-drift-detection)
18. [Model Performance Monitoring](#18-model-performance-monitoring)
19. [Automated Retraining](#19-automated-retraining)
20. [Model Versioning & Rollback](#20-model-versioning--rollback)
21. [Project Directory Structure](#21-project-directory-structure)
22. [Installation Instructions](#22-installation-instructions)
23. [Environment Setup](#23-environment-setup)
24. [How to Train the Model](#24-how-to-train-the-model)
25. [How to Start MLflow](#25-how-to-start-mlflow)
26. [How to Start Airflow](#26-how-to-start-airflow)
27. [How to Start FastAPI](#27-how-to-start-fastapi)
28. [How to Run Docker](#28-how-to-run-docker)
29. [How to Run Monitoring](#29-how-to-run-monitoring)
30. [How Automated Retraining Works](#30-how-automated-retraining-works)
31. [Testing & Quality Assurance](#31-testing--quality-assurance)
32. [Future Improvements](#32-future-improvements)
33. [Limitations](#33-limitations)
34. [Example API Request & Response](#34-example-api-request--response)

---

## 1. Project Title
**Predictive Maintenance MLOps System**

## 2. Project Overview
Industrial machinery failures lead to unpredicted downtime, financial loss, and safety risks. This project delivers an automated, end-to-end MLOps pipeline using the **AI4I 2020 Predictive Maintenance Dataset**. The system continuously monitors sensor readings (temperatures, rotational speeds, torque, tool wear), predicts potential machine failures in real time, logs experiments and model versions in MLflow, orchestrates daily or triggered pipeline runs with Apache Airflow, serves low-latency REST endpoints with FastAPI, detects data and model drift using statistical checks, and safely retrains and deploys models when degradation occurs.

## 3. Problem Statement
Traditional industrial maintenance relies on either **reactive maintenance** (fixing machines after catastrophic failure) or **scheduled preventive maintenance** (servicing machines on fixed intervals regardless of condition). Both approaches are inefficient: reactive maintenance causes high downtime costs, while scheduled maintenance wastes resources servicing functional machinery. Predictive maintenance leverages continuous sensor data to forecast failures *before* they occur. However, deploying machine learning in industrial environments presents operational challenges, including model stale-out from sensor drift, lack of automated monitoring, and manual deployment bottlenecks.

## 4. Objectives
* Build a scalable dataset ingestion and validation module enforcing strict schema and data quality rules.
* Engineer domain-specific features (e.g., temperature differences, mechanical power output, tool strain index).
* Evaluate candidate algorithms (Logistic Regression, Random Forest, XGBoost) using failure-aware metrics (Recall, F1-Score, ROC-AUC, PR-AUC).
* Integrate **MLflow** for experiment tracking, artifact logging, and model registry management using the `@Production` model alias.
* Build an automated master pipeline with non-zero failure exit codes and SHA256 checksum-based dataset caching.
* Create an **Apache Airflow DAG** for scheduled and on-demand workflow execution.
* Deploy a production-ready **FastAPI** application for low-latency REST inference with Pydantic validation.
* Containerize the environment using **Docker** and multi-container **Docker Compose**.
* Implement statistical **Data & Model Drift Monitoring** (Kolmogorov-Smirnov test, Wasserstein distance, Recall degradation).
* Enable **Automated Retraining** with promotion safety checks preventing model performance regression.

## 5. Dataset Description
The system uses the **AI4I 2020 Predictive Maintenance Dataset** (10,000 observations):
* **Data Sources**: Telemetry measurements from synthetic industrial milling machines.
* **Target Variable**: `Machine failure` (0 = Normal Operation, 1 = Machine Failure). Highly imbalanced dataset with ~3.39% failure rate (339 positive failure samples out of 10,000).
* **Failure Modes**: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), Random Failure (RNF).

## 6. Features Used
### Raw Telemetry Features:
* `Type`: Product quality variant (`L` = Low/50%, `M` = Medium/30%, `H` = High/20%).
* `Air temperature [K]`: Ambient environmental temperature (K).
* `Process temperature [K]`: Internal operational temperature of the machine process (K).
* `Rotational speed [rpm]`: Spindle rotation speed (rpm).
* `Torque [Nm]`: Mechanical torque generated by the spindle (Nm).
* `Tool wear [min]`: Cumulative tool operational wear duration in minutes.

### Domain-Engineered Features:
* `temp_difference`: $T_{\text{process}} - T_{\text{air}}$ (critical for Heat Dissipation Failure detection).
* `power`: Mechanical power generated (kW), calculated as $\frac{\text{Torque} \times \text{Rotational Speed} \times \frac{2\pi}{60}}{1000}$ (key indicator for Power Failure).
* `tool_wear_speed_ratio`: $\frac{\text{Tool Wear}}{\text{Rotational Speed}}$ (captures tool stress under high rotational loads).
* `tool_wear_torque`: $\text{Tool Wear} \times \text{Torque}$ (captures mechanical strain causing Overstrain Failure).

## 7. Machine Learning Approach
Equipment failure prediction is a heavily class-imbalanced binary classification task. Maximizing **Recall** is prioritized over plain accuracy because a **False Negative** (failing to predict a machine breakdown) carries severe cost consequences, whereas a **False Positive** incurs only a routine inspection check.
* **Data Splitting**: Stratified 70/15/15 train/validation/test split to preserve failure ratio across sets.
* **Preprocessing Pipeline**: `ColumnTransformer` with `StandardScaler` for numeric variables and `OneHotEncoder` for categorical product type.
* **Class Weighting**: Balanced class weights applied to compensate for class imbalance.

## 8. Model Comparison
Three candidate models were evaluated under identical stratified train/validation/test splits:

| Candidate Model | Validation Recall | Validation F1 | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** (Baseline) | 78.43% | 0.5128 | 80.39% | 0.5256 | 0.9241 | 0.5482 |
| **Random Forest** (Selected) | **82.35%** | **0.7368** | **84.31%** | **0.7350** | **0.9701** | **0.7512** |
| **XGBoost Classifier** | 80.39% | 0.7130 | 82.35% | 0.7241 | 0.9654 | 0.7380 |

## 9. Final Model
* **Algorithm**: **Random Forest Classifier** (`n_estimators=100`, `max_depth=10`, `class_weight='balanced'`).
* **Justification**: Delivered highest Test Recall (84.31%), Test F1-Score (0.7350), and Test ROC-AUC (0.9701) while demonstrating robust generalization without overfitting.
* **Artifacts Saved**: `models/best_model.joblib`, `data/processed/preprocessor.joblib`, `models/model_metadata.json`, and `models/evaluation_metrics.json`.

## 10. MLOps Architecture

```text
                ┌─────────────────────┐
                │   Sensor Data       │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Data Ingestion      │ (SQLite & CSV Storage)
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Data Validation     │ (Schema & Range Check)
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Preprocessing       │ (Scaling & Domain Features)
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Model Training      │ (LR / RF / XGBoost)
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ MLflow Tracking     │ (Runs, Params, Artifacts)
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Model Registry      │ (Alias: @Production)
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ FastAPI Serving     │ (GET /health, POST /predict)
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Live Predictions    │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Monitoring Engine   │ (Evidently / SciPy Drift)
                └──────────┬──────────┘
                           ↓
                   Drift/Degradation?
                     /           \
                   No             Yes
                   ↓               ↓
                Continue      Retraining Trigger
                                  ↓
                              Pipeline Re-run
                                  ↓
                          Model Comparison
                           (Safety Gate)
                                  ↓
                          Registry Update (@Production)
                                  ↓
                              Deployment
```

## 11. Data Pipeline
The master pipeline in `src/pipeline.py` executes modular stages sequentially:
1. `ingest`: Downloads raw dataset or loads from SQLite DB.
2. `validate`: Performs 8 schema validation checks (null counts, duplicate rows, data types, value ranges).
3. `preprocess`: Transforms raw sensor telemetry into scaled features and domain interaction ratios.
4. `train`: Evaluates candidate algorithms with reproducible random seeds.
5. `register`: Registers top model into MLflow Model Registry.

* **Caching**: Computes SHA256 checksums of input dataset to prevent redundant reprocessing when upstream data has not changed.
* **Error Handling**: Raises explicit exceptions and returns non-zero exit codes (`exit status 1`) upon stage validation failure.

## 12. MLflow Experiment Tracking
* **Experiment Name**: `predictive-maintenance`
* **Backend Store**: SQLite database (`sqlite:///mlflow.db`).
* **Logged Parameters**: `model_type`, `n_estimators`, `max_depth`, `random_state`, feature count, domain feature flags.
* **Logged Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC across train/validation/test sets.
* **Logged Artifacts**: Confusion matrix plot (`reports/figures/confusion_matrix.png`), ROC curve, Precision-Recall curve, trained model joblib binary, feature information JSON.
* **Model Registry**: Registers best model under `PredictiveMaintenanceModel` and assigns alias `@Production`.

## 13. Airflow Workflow Scheduling
* **DAG Location**: `dags/predictive_maintenance_dag.py`
* **DAG ID**: `predictive_maintenance_mlops_pipeline`
* **Schedule**: Weekly, Sundays 00:00 (`0 0 * * 0`), with manual triggering.
* **Tasks** (linear): `data_ingestion` -> `data_validation` -> `preprocessing` -> `model_training` -> `mlflow_registration`
* **Features**: 2 retries with a 1-minute retry delay and failure callbacks. Training already registers the winning model, so `mlflow_registration` only verifies the registry state (it no longer creates a duplicate version on every run).
* The same pipeline can be started on demand, with live logs, from the dashboard's **Pipeline** page.

## 14. FastAPI Deployment
* **App Path**: `src/api/main.py`
* **Inference endpoints**:
  * `GET /health`: service status, model load status, model version, timestamp.
  * `POST /predict`: 1 to 1,000 sensor readings; validated by Pydantic; returns prediction (`0`/`1`), status label (`Normal Operation` / `Equipment Failure Warning`), failure probability, model name and version. Each prediction is logged to `data/predictions/predictions.jsonl` for the dashboard and for live-traffic drift monitoring.
* **Dashboard endpoints**: `/api/...` (see "Web Dashboard" above) implemented in `src/api/routes.py`, `insights.py` (read-only) and `operations.py` (state-changing).
* **Model loading**: thread-safe loader (`src/api/model_loader.py`) serving `models/best_model.joblib`. It detects when the artifact files change on disk, so a retrain (CLI, Airflow or dashboard) is served without restarting the API.
* **Configuration**: `CORS_ORIGINS` (comma-separated, default `*`), `API_HOST`, `API_PORT`, `FRONTEND_DIST`, `MLFLOW_UI_URL`, `AIRFLOW_UI_URL`.

## 15. Docker Setup
* **Dockerfile**: `docker/Dockerfile.api`: multi-stage (Node builds the React UI, then `python:3.11-slim` runs the API), non-root user, `libgomp1` for XGBoost, health check on `/health`.
* **Docker Compose**: `docker-compose.yml`:
  * `fastapi_app`: API + dashboard on port `8000`, with `data/`, `models/`, `mlflow.db`, `mlruns/`, `reports/` and `logs/` mounted.
  * `mlflow_server`: MLflow UI on port `5000` (same image, same database).
  * `airflow` (profile `airflow`, experimental): `docker compose --profile airflow up`.

## 16. Monitoring System
* **Module Path**: `src/monitoring/drift_detector.py` (batch construction for scenarios and uploads: `src/monitoring/scenarios.py`)
* **Reference dataset**: the processed training split (`data/processed/train.csv`).
* **Current dataset**: by default the held-out validation split; the dashboard can also check simulated drift, uploaded CSV rows, or the live traffic logged by `/predict`.
* **Output**: `data/reports/drift_report.json` (full detail) and `data/reports/monitoring_summary.json`; the dashboard additionally keeps `monitoring_history.jsonl` for trend charts.

## 17. Data Drift Detection
* **Methodology**: two-sample Kolmogorov-Smirnov test on every engineered/scaled feature against the training split.
* **Rule**: a feature drifts when its p-value is below `p_value_alpha` (0.05). Data drift is flagged when more than `max_drifted_feature_ratio` (30%) of features drift.

## 18. Model Performance Monitoring
* **Prediction shift**: Wasserstein distance between training and current predicted probabilities; flagged above `wasserstein_prob_drift_limit` (0.10).
* **Performance degradation** (needs a `Machine failure` label column): flagged when recall falls below `min_acceptable_recall` (0.75) **or** F1 below `min_acceptable_f1` (0.65).
* All thresholds live in `config/config.yaml` under `monitoring.thresholds`.

## 19. Automated Retraining
* **Module Path**: `src/retraining/trigger.py` (gate logic in `src/retraining/promotion.py`)
* **Triggers**: (1) feature data drift, (2) prediction shift, (3) performance degradation, (4) at least `min_new_samples_threshold` (500) new labeled samples passed to `evaluate_retraining_decision`, (5) manual override.
* **Promotion safety gate** (thresholds in `config.yaml` -> `retraining.promotion_criteria`). A candidate trained by the retraining run replaces production only if **all** checks pass:
  1. Candidate test recall >= `min_recall_threshold` (0.75).
  2. Candidate test recall - production test recall >= `min_recall_improvement_pct` / 100 (default 0: not worse).
  3. Production precision - candidate precision <= `max_precision_drop_pct` / 100 (default 5 points).
* **Rejection is a real rollback**: the `@Production` alias is moved back **and** `best_model.joblib`, `evaluation_metrics.json` and `model_metadata.json` (the files the API serves) are restored. (Previously only the alias was reverted, so the API kept serving the rejected model.)
* Every attempt, including the per-check results, is appended to `models/retraining_history.json`.

## 20. Model Versioning & Rollback
* **Versioning**: the MLflow Model Registry tracks sequential versions of `PredictiveMaintenanceModel`.
* **Aliases**: the serving model carries the alias `@Production`.
* **What the API serves**: `models/best_model.joblib` (plus the matching `data/processed/preprocessor.joblib`). The registry alias is the audit trail; it does not by itself change what the API loads.
* **Automatic rollback**: performed by the retraining promotion gate (section 19).
* **Manual rollback**: restore the previous `models/best_model.joblib` (and `evaluation_metrics.json` / `model_metadata.json`) from backup or version control, then press **Reload model** on the System page (or `POST /api/model/reload`). To keep the registry consistent:
  ```python
  from mlflow.tracking import MlflowClient
  MlflowClient().set_registered_model_alias("PredictiveMaintenanceModel", "Production", "<version>")
  ```

## 21. Project Directory Structure
```text
predictive-maintenance-mlops/
├── config/
│   ├── config.yaml               # Master global configuration settings
│   └── model_config.json         # Feature specifications & default parameters
├── dags/
│   └── predictive_maintenance_dag.py # Apache Airflow workflow DAG
├── data/
│   ├── database/                 # SQLite database storage (predictive_maintenance.db)
│   ├── processed/                # Scaled features & saved preprocessor joblib
│   ├── raw/                      # Raw dataset (ai4i2020.csv)
│   └── reports/                  # Data validation reports
├── docker/
│   └── Dockerfile.api            # Multi-stage build: React dashboard + FastAPI
├── frontend/                     # React + Vite dashboard (prebuilt output in frontend/dist)
│   ├── src/pages/                # Overview, Predict, Model, Experiments, Pipeline, Monitoring, Retraining, Data, System
│   ├── src/components/           # UI primitives, SVG charts, job log viewer
│   └── src/lib/                  # API client, hooks, router, domain rules
├── logs/                         # Execution and application log files
├── models/                       # Model joblib binaries and metadata JSONs
├── notebooks/                    # EDA analysis notebook & visualization generator
├── reports/
│   └── figures/                  # Plot artifacts (confusion matrices, ROC/PR curves)
├── src/
│   ├── api/                      # FastAPI service application
│   │   ├── main.py               # App, /health, /predict, serves the dashboard
│   │   ├── routes.py             # Dashboard endpoints (/api/...)
│   │   ├── insights.py           # Read-only analytics for the dashboard
│   │   ├── operations.py         # Run monitoring, start pipeline/retraining jobs
│   │   ├── jobs.py               # Background job runner with captured logs
│   │   ├── mlflow_reader.py      # Runs and registry versions from MLflow
│   │   ├── prediction_log.py     # JSONL log of served predictions
│   │   ├── model_loader.py       # Thread-safe loader, auto-reloads changed artifacts
│   │   └── schemas.py            # Pydantic request/response schemas
│   ├── data/
│   │   ├── ingest.py             # Data ingestion module
│   │   └── validate.py           # Data quality validation module
│   ├── features/
│   │   └── build_features.py     # Preprocessing & domain feature engineer
│   ├── models/
│   │   ├── evaluate.py           # Evaluation metrics computer
│   │   ├── register.py           # MLflow registry manager
│   │   └── train.py              # Candidate model trainer
│   ├── monitoring/
│   │   ├── drift_detector.py     # Statistical drift and monitoring engine
│   │   └── scenarios.py          # Simulated drift, uploaded and live batches
│   ├── retraining/
│   │   ├── trigger.py            # Retraining decision engine with rollback
│   │   └── promotion.py          # Promotion safety gate (pure functions)
│   ├── utils/
│   │   ├── config.py             # Configuration YAML/JSON parser
│   │   └── logger.py             # Centralized structured logger
│   └── pipeline.py               # Master automated pipeline orchestrator
├── tests/                        # Comprehensive unit & integration test suite
│   ├── test_api.py               # FastAPI endpoint tests
│   ├── test_data.py              # Ingestion & validation tests
│   ├── test_e2e_smoke.py         # End-to-end full system smoke test
│   ├── test_features.py          # Preprocessing & feature engineering tests
│   ├── test_model.py             # Training & evaluation tests
│   ├── test_monitoring.py        # Drift & monitoring tests
│   ├── test_pipeline.py          # Master pipeline orchestrator tests
│   └── test_retraining.py        # Retraining decision & safety gate tests
├── .dockerignore                 # Docker build ignore patterns
├── .env                          # Local environment variables
├── .env.example                  # Environment variable blueprint
├── .gitignore                    # Git version control exclusions
├── docker-compose.yml            # Docker Compose orchestration file
├── mlflow.db                     # MLflow tracking SQLite backend database
├── PROJECT_REPORT.md             # Concise academic project report
├── README.md                     # Comprehensive system documentation
└── requirements.txt              # Python package dependencies
```

## 22. Installation Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/predictive-maintenance-mlops.git
   cd predictive-maintenance-mlops
   ```
2. **Create Python Virtual Environment**:
   ```bash
   python -m venv .venv
   ```
3. **Activate Virtual Environment**:
   * Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
   * Linux/macOS: `source .venv/bin/activate`
4. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 23. Environment Setup
Copy `.env.example` to create `.env`:
```bash
cp .env.example .env
```
Default `.env` contents:
```ini
APP_ENV=development
LOG_LEVEL=INFO
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=predictive-maintenance
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=sqlite:///data/database/predictive_maintenance.db
```

## 24. How to Train the Model
Run the master automated pipeline to execute data ingestion, validation, feature engineering, candidate model training, evaluation, MLflow logging, and model registration:
```bash
python -m src.pipeline
```

## 25. How to Start MLflow
Launch the MLflow Tracking UI server locally:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 26. How to Start Airflow
1. Set Airflow Home directory:
   ```bash
   export AIRFLOW_HOME=$(pwd)/airflow
   ```
2. Initialize and start Airflow Standalone:
   ```bash
   airflow standalone
   ```
3. Open browser at [http://127.0.0.1:8080](http://127.0.0.1:8080) and trigger `predictive_maintenance_pipeline` DAG.

## 27. How to Start FastAPI
Start Uvicorn development server:
```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```
* **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## 28. How to Run Docker

### Build Single API Docker Image:
```bash
docker build -t predictive-maintenance-api -f docker/Dockerfile.api .
```

### Run Single Container:
```bash
docker run -d -p 8000:8000 --name predictive_api predictive-maintenance-api
```

### Run Multi-Container Stack (FastAPI + MLflow) with Docker Compose:
```bash
# Launch containers in background
docker compose up -d

# Inspect live container logs
docker compose logs -f

# Shut down containers
docker compose down
```

## 29. How to Run Monitoring
Run statistical drift detection comparing current production batch (`data/processed/current_data.csv`) against reference training baseline (`data/processed/reference_data.csv`):
```bash
python -m src.monitoring.drift_detector
```
Check generated JSON report at `reports/monitoring_report.json`.

## 30. How Automated Retraining Works
CLI (forces a retrain and runs the promotion gate):
```bash
python -m src.retraining.trigger
```
From code: `execute_automated_retraining(force=False)` evaluates monitoring first; `execute_automated_retraining(decision=...)` acts on a precomputed decision (used by the dashboard's *Run automated check*).

### Execution Sequence:
1. Decide: run monitoring (or reuse the latest report) and apply the trigger rules.
2. If no trigger fired, stop (`SKIPPED`).
3. Snapshot the served model files and the current `@Production` version and test metrics.
4. Train and evaluate all candidates; the winner is registered.
5. Apply the promotion gate against the snapshot metrics.
6. On rejection or failure, restore the snapshot and the previous alias.
7. Append the outcome to `models/retraining_history.json`.

## 31. Testing & Quality Assurance
Run complete automated test suite using `pytest`:
```bash
pytest tests/ -v
```
### Test Coverage:
* `test_data.py`: Dataset ingestion and schema validation rules.
* `test_features.py`: Scikit-learn preprocessing pipeline and domain feature engineering.
* `test_model.py`: Candidate model training, metric evaluation, and joblib persistence.
* `test_pipeline.py`: Master automated pipeline orchestration and checksum caching.
* `test_api.py`: FastAPI `/health` and `/predict` endpoints with valid/invalid payloads.
* `test_monitoring.py`: KS-test data drift and performance degradation detection.
* `test_retraining.py`: Retraining trigger evaluation and safety promotion gate.
* `test_dashboard_services.py`: Promotion gate, drift scenarios, prediction log, background jobs, MLflow reader.
* `test_dashboard_api.py`: Smoke tests of the `/api/...` dashboard endpoints.
* `test_e2e_smoke.py`: Comprehensive end-to-end full system lifecycle integration test.

## 32. Future Improvements
* Integrate **Evidently AI Dashboard UI** for live HTML visual drift tracking.
* Expand database support from SQLite to PostgreSQL or BigQuery for production deployment.
* Add streaming telemetry ingestion via Apache Kafka or MQTT protocol for real-time sensor streams.
* Implement Kubernetes (K8s) deployment manifests and Helm charts for cloud auto-scaling.

## 33. Limitations
* **Retraining reuses the same training data**: drift and new-sample counts decide *when* to retrain, but the retraining run trains on the existing processed dataset; merging newly labeled batches into the training set is the most valuable next step.
* **Serving is file-based**: the API loads `models/best_model.joblib`, not the registry directly.
* **Imbalanced Class Ratio**: Dataset contains ~3.39% failure cases, requiring careful threshold tuning.
* **Synthetic Telemetry**: The AI4I 2020 Dataset is a synthetic benchmark dataset. Real-world machinery sensor telemetry may exhibit higher non-stationary noise.
* **Local SQLite Backend**: MLflow and metadata storage use SQLite, which is ideal for local development and mini-projects but should be upgraded to cloud database for distributed teams.

## 34. Example API Request & Response

### HTTP Endpoint: `POST http://127.0.0.1:8000/predict`

#### Sample JSON Request Payload:
```json
{
  "inputs": [
    {
      "Type": "M",
      "Air temperature [K]": 298.1,
      "Process temperature [K]": 308.6,
      "Rotational speed [rpm]": 1551.0,
      "Torque [Nm]": 42.8,
      "Tool wear [min]": 0.0
    },
    {
      "Type": "L",
      "Air temperature [K]": 302.5,
      "Process temperature [K]": 311.2,
      "Rotational speed [rpm]": 1380.0,
      "Torque [Nm]": 68.4,
      "Tool wear [min]": 215.0
    }
  ]
}
```

#### Sample JSON Response Payload:
```json
{
  "success": true,
  "predictions": [
    {
      "prediction": 0,
      "status": "Normal Operation",
      "failure_probability": 0.0100,
      "model_name": "PredictiveMaintenanceModel",
      "model_version": "1",
      "timestamp": "2026-10-02T08:50:12.345678"
    },
    {
      "prediction": 1,
      "status": "Equipment Failure Warning",
      "failure_probability": 0.8920,
      "model_name": "PredictiveMaintenanceModel",
      "model_version": "1",
      "timestamp": "2026-10-02T08:50:12.345678"
    }
  ]
}
```
