# Academic Project Report: Predictive Maintenance MLOps System

**Degree / Course**: Machine Learning Operations (MLOps) Mini-Project  
**Domain**: Industrial IoT & Automated Equipment Failure Forecasting  
**Author / Student**: MLOps System Engineer  

---

## Executive Summary
Unplanned machine breakdown in manufacturing facilities leads to high maintenance costs and production loss. This mini-project presents an end-to-end **Predictive Maintenance MLOps System** built on the **AI4I 2020 Predictive Maintenance Dataset**. The project implements a complete machine learning lifecycle including automated data ingestion and quality validation, domain feature engineering, candidate model training (Logistic Regression, Random Forest, XGBoost), experiment tracking and model registry via MLflow, workflow scheduling using Apache Airflow, REST API deployment with FastAPI, containerization via Docker, statistical data drift monitoring, and automated safety-gated retraining.

---

## 1. Introduction & Problem Statement
Industrial machines experience wear and operational strain over time, eventually leading to component failures such as tool wear failure, heat dissipation breakdown, power overload, or mechanical strain. 

Existing maintenance strategies fall into two categories:
1. **Reactive Maintenance**: Fixing machinery only after catastrophic breakdown occurs. This incurs high repair costs and unscheduled downtime.
2. **Preventive Maintenance**: Servicing equipment on fixed calendar schedules regardless of actual physical condition, which wastes resources.

**Predictive Maintenance** utilizes continuous sensor measurements (temperature, rotational speed, torque, tool wear) to forecast failures prior to breakdown. However, deploying machine learning models in production requires robust MLOps practices to ensure data validity, track model iterations, prevent performance degradation, and automate redeployment seamlessly.

---

## 2. Project Objectives
* **Data Validation**: Enforce schema validation and data sanity rules on incoming raw telemetry datasets.
* **Feature Engineering**: Derive physical interaction features (temperature delta, mechanical power, tool strain index).
* **Model Selection**: Train and compare multiple models focusing on **Recall** to minimize missed equipment failures.
* **MLflow Tracking & Registry**: Track parameters, metrics, confusion matrices, and model versions under a registered `@Production` model alias.
* **API Serving**: Expose high-performance REST endpoints (`GET /health`, `POST /predict`) using FastAPI with Pydantic payload validation.
* **Containerization**: Package the inference application into Docker containers using multi-container Docker Compose.
* **Monitoring & Retraining**: Perform statistical Kolmogorov-Smirnov drift tests and automatically trigger retraining when drift or performance degradation occurs.

---

## 3. Dataset & Domain Feature Engineering

### 3.1 AI4I 2020 Predictive Maintenance Dataset
The dataset consists of 10,000 synthetic industrial milling machine observations featuring:
* Categorical Product Type: `L` (Low quality), `M` (Medium quality), `H` (High quality).
* Environmental Telemetry: `Air temperature [K]`, `Process temperature [K]`.
* Mechanical Telemetry: `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.
* Target Label: `Target` (0 = Normal Operation, 1 = Machine Failure). The failure rate is **3.39%** (339 failure cases out of 10,000).

### 3.2 Domain-Engineered Features
To enhance predictive signal, domain-specific physical relationships were engineered:
1. **Temperature Difference ($T_{\text{diff}}$)**: $T_{\text{process}} - T_{\text{air}}$ — indicates heat dissipation capacity.
2. **Mechanical Power ($P_{\text{kW}}$)**: $\frac{\text{Torque} \times \text{Rotational Speed} \times \frac{2\pi}{60}}{1000}$ — quantifies mechanical energy input.
3. **Tool Wear Speed Ratio**: $\frac{\text{Tool Wear}}{\text{Rotational Speed}}$ — captures stress under rotation.
4. **Tool Wear Torque Index**: $\text{Tool Wear} \times \text{Torque}$ — measures cumulative mechanical strain.

---

## 4. Machine Learning Methodology & Candidate Models

### 4.1 Evaluation Strategy
Equipment failure prediction is an imbalanced classification problem. A **False Negative** (failing to predict a breakdown) leads to expensive physical damage, while a **False Positive** results only in an unnecessary visual check. Therefore, **Recall** is prioritized alongside F1-Score, ROC-AUC, and PR-AUC.

Data was split into 70% Train (7,000 samples), 15% Validation (1,500 samples), and 15% Test (1,500 samples) using **stratified sampling**.

### 4.2 Candidate Model Evaluation Results

| Candidate Algorithm | Validation Recall | Validation F1 | Test Recall | Test F1-Score | Test ROC-AUC | Test PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** (Baseline) | 78.43% | 0.5128 | 80.39% | 0.5256 | 0.9241 | 0.5482 |
| **Random Forest** (Selected) | **82.35%** | **0.7368** | **84.31%** | **0.7350** | **0.9701** | **0.7512** |
| **XGBoost Classifier** | 80.39% | 0.7130 | 82.35% | 0.7241 | 0.9654 | 0.7380 |

**Selected Model**: **Random Forest Classifier** (`n_estimators=100`, `max_depth=10`, `class_weight='balanced'`). It achieved the highest Test Recall (84.31%) and Test ROC-AUC (0.9701).

---

## 5. System Architecture & Component Design

### 5.1 System Modules
1. `src/data/ingest.py`: Manages data ingestion and SQLite storage (`predictive_maintenance.db`).
2. `src/data/validate.py`: Verifies column schemas, missing values, duplicate rows, and value ranges.
3. `src/features/build_features.py`: Implements Scikit-learn preprocessing `ColumnTransformer` and domain feature additions.
4. `src/models/train.py` & `evaluate.py`: Evaluates candidate algorithms with reproducible random seeds.
5. `src/models/register.py`: Logs runs, parameters, metrics, artifacts, and assigns the `@Production` alias in MLflow.
6. `src/pipeline.py`: Master pipeline orchestrator with SHA256 caching and explicit error status codes.
7. `dags/predictive_maintenance_dag.py`: Apache Airflow DAG organizing workflow tasks.
8. `src/api/main.py`: FastAPI server serving endpoints `/health` and `/predict`.
9. `src/monitoring/drift_detector.py`: Evaluates feature drift using two-sample Kolmogorov-Smirnov tests and prediction shift via Wasserstein distance.
10. `src/retraining/trigger.py`: Evaluates drift/degradation triggers and executes retraining with candidate safety gates.

---

## 6. How the Complete System Works

Below is the step-by-step operational lifecycle explaining how data flows through the entire system from raw sensor telemetry to automated retraining:

1. **Step 1: Telemetry Data Ingestion**  
   Raw sensor readings (air/process temperatures, rotational speed, torque, tool wear) are downloaded or received and stored in `data/raw/ai4i2020.csv` and the local SQLite database (`data/database/predictive_maintenance.db`).

2. **Step 2: Automated Data Validation**  
   The validation module checks the ingested data against expected schemas, ensuring correct data types, zero unexpected missing values, no duplicate records, and realistic physical range boundaries (e.g., non-negative tool wear, positive temperatures).

3. **Step 3: Feature Engineering & Preprocessing**  
   Categorical quality types (`L`, `M`, `H`) are One-Hot Encoded, and numeric features are scaled using `StandardScaler`. Domain-specific interaction features ($T_{\text{diff}}$, mechanical power $P_{\text{kW}}$, and tool strain index) are derived dynamically.

4. **Step 4: Candidate Model Training**  
   The preprocessing pipeline and feature matrix are passed to candidate algorithms (Logistic Regression baseline, Random Forest, XGBoost). Models are trained on a stratified 70% split using balanced class weighting to handle imbalance.

5. **Step 5: Metric Evaluation & Model Selection**  
   Models are evaluated on validation and test sets. Metrics including Recall, F1-Score, ROC-AUC, PR-AUC, and confusion matrices are computed. The algorithm with the highest Recall (Random Forest) is automatically selected.

6. **Step 6: MLflow Tracking & Artifact Logging**  
   All run parameters, hyperparameters, metrics, confusion matrix plots, feature transformer pipelines, and trained model binaries are logged to the MLflow experiment tracking database (`sqlite:///mlflow.db`).

7. **Step 7: Model Registration & Production Tagging**  
   The winning Random Forest model is saved into the MLflow Model Registry under `PredictiveMaintenanceModel` and assigned the official alias `@Production`.

8. **Step 8: Orchestration via Airflow DAG**  
   Apache Airflow orchestrates the sequential execution of ingestion, validation, preprocessing, training, evaluation, and registration via DAG tasks, with retries and failure alerts configured.

9. **Step 9: Model Serving via FastAPI REST API**  
   The FastAPI application dynamically loads the `@Production` model and preprocessor. External clients send JSON sensor payloads to `POST /predict`, receiving real-time predictions (`Normal Operation` vs `Machine Failure`), failure probabilities, model version, and timestamps.

10. **Step 10: Docker Containerization**  
    The FastAPI inference service and MLflow UI are packaged into Docker containers using multi-stage `Dockerfile.api` and orchestrated together using `docker-compose.yml`.

11. **Step 11: Statistical Drift & Performance Monitoring**  
    The monitoring engine periodically compares live incoming production batch telemetry against reference training baseline data using Kolmogorov-Smirnov statistical tests for data drift and Wasserstein distance for prediction probability shifts.

12. **Step 12: Performance Degradation Check**  
    When actual ground-truth failure labels become available, the monitoring engine computes current live Recall. If live Recall drops below the safety threshold ($> 0.10$ drop), performance degradation is flagged.

13. **Step 13: Automated Retraining Trigger**  
    If data drift is detected, performance degrades, or new labeled data exceeds 1,000 records, the retraining module triggers `RETRAIN_REQUIRED` and launches the training pipeline on updated data.

14. **Step 14: Safety-Gated Model Promotion**  
    The newly trained candidate model is evaluated on test data. The system checks if Candidate Recall $\ge$ Active Production Recall $- 0.02$. If passed, the candidate model is promoted to `@Production` in MLflow and seamlessly loaded by FastAPI without downtime.

---

## 7. Viva Questions & Answers Guide

* **Q1: Why did you choose Recall as your main evaluation metric?**  
  *Answer*: In predictive maintenance, a False Negative means a machine failure goes undetected, leading to unexpected breakdown and major financial loss. A False Positive results only in an inspector checking a working machine. Therefore, maximizing Recall minimizes undetected breakdowns.

* **Q2: How does MLflow fit into your project?**  
  *Answer*: MLflow provides experiment tracking and model versioning. It logs parameters, metrics, confusion matrices, and model binaries into an SQLite backend, allowing us to compare candidate runs and manage model releases using the `@Production` alias in the Model Registry.

* **Q3: How do you handle Data Drift?**  
  *Answer*: We use two-sample Kolmogorov-Smirnov (KS) tests comparing numerical sensor feature distributions in live production against reference training data. If $p < 0.05$, data drift is flagged, triggering automated retraining.

* **Q4: How does the FastAPI service know when a new model is deployed?**  
  *Answer*: FastAPI uses a dynamic model loader (`src/api/model_loader.py`) that checks the MLflow Model Registry for the latest `@Production` alias or local joblib artifact, ensuring zero-downtime model updates.

---

## 8. Conclusion
The Predictive Maintenance MLOps System demonstrates a robust, production-style machine learning workflow. By integrating dataset validation, domain feature engineering, MLflow tracking, FastAPI serving, Airflow orchestration, Docker containerization, statistical drift monitoring, and safety-gated retraining, the system ensures continuous model accuracy and reliable machinery failure forecasting.
