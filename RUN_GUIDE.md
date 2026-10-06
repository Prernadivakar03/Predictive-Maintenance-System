# How to run this project

This project has three parts that work together:

| Part | What it is | Where it runs |
|---|---|---|
| **ML pipeline** | Ingest, validate, preprocess, train 3 models, log to MLflow, register the best one | `python -m src.pipeline` (or the **Run pipeline** button) |
| **API + dashboard** | FastAPI serves `/predict`, `/health`, the `/api/...` endpoints **and** the React dashboard | http://127.0.0.1:8000 |
| **MLflow UI / Airflow** | Experiment tracking UI and the weekly scheduler (separate processes) | http://localhost:5000 / http://localhost:8080 |

The React dashboard is **already built** in `frontend/dist`, so you do **not** need Node.js just to use it.

---

## 1. Fastest way (one command)

**Windows** (double-click, or in a terminal):
```bat
start.bat
```
**macOS / Linux:**
```bash
./start.sh
```
It creates `.venv`, installs `requirements.txt`, trains a model if none exists, and starts the server.
Then open **http://127.0.0.1:8000**.

---

## 2. Step by step

Requirements: **Python 3.10 – 3.12** (3.11 recommended). Node.js 18+ is only needed to *change* the frontend.

### Windows (PowerShell)
```powershell
cd path\to\MLOOPS
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked:  Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt

python -m src.pipeline                # ~1-2 min: trains models, writes models/, MLflow db, reports
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### macOS / Linux
```bash
cd path/to/MLOOPS
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.pipeline
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** (dashboard) or **http://127.0.0.1:8000/docs** (Swagger API docs).

> Always start commands from the project root (the folder that contains `config/` and `src/`).

---

## 3. MLflow UI (experiment tracking)

In a **second** terminal (venv activated, project root):
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open http://localhost:5000. The dashboard's **Experiments** page links to it.

---

## 4. Airflow (weekly scheduled retraining)

Airflow does not run natively on Windows. Use **WSL2** or **Docker**.

**WSL2 / Linux / macOS:**
```bash
pip install "apache-airflow==2.10.2" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.2/constraints-3.11.txt"
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=false
airflow standalone            # prints the admin password; UI on http://localhost:8080
```
Enable the DAG `predictive_maintenance_mlops_pipeline` (schedule: Sundays 00:00) or trigger it manually.

**Docker (experimental):** `docker compose --profile airflow up` (see `docker-compose.yml`).

> You can also run the same pipeline on demand, with live logs, from the dashboard's **Pipeline** page. Airflow is only needed for the *scheduled* runs.

---

## 5. Docker

```bash
python -m src.pipeline            # once, so ./mlflow.db and ./models exist (Docker would otherwise create a folder named mlflow.db)
docker compose up --build
```
* Dashboard + API: http://localhost:8000
* MLflow UI: http://localhost:5000

The image builds the React dashboard in a Node stage, so no local Node is needed.

---

## 6. Developing the frontend (optional)

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173  (hot reload; proxies API calls to http://127.0.0.1:8000)
npm run build      # writes frontend/dist, which FastAPI serves
```
Keep the API running on port 8000 while using `npm run dev`. To point at another API: `VITE_API_PROXY=http://host:port npm run dev`.

---

## 7. Tests

```bash
pytest tests/ -v
```
Note: several existing tests (`test_pipeline`, `test_e2e_smoke`, `test_retraining`) run the **real pipeline** against your working copy. They retrain the model, overwrite `models/`, and register extra MLflow versions. That is why a long-lived project accumulates dozens of identical model versions. Run them on a copy if you want to keep a specific model.

---

## 8. A 5-minute tour of the dashboard

1. **Overview**: five status lamps (data, model, drift, pipeline, retraining).
2. **Predict**: click *Overstrain* then *Predict failure risk*. The dial shows the failure probability. Try *Batch scoring* with a CSV.
3. **Monitoring**: run *Held-out validation set* (expect no drift), then *Combined drift* at 60%. Drift alarms fire.
4. **Retraining**: the decision now says "recommended". Press **Run automated check** (retrains because triggers fired) or **Retrain now**. The new model replaces production **only if it passes the safety gate**; otherwise the previous model is restored.
5. **Pipeline**: **Run pipeline** shows live stage progress and logs. **Model** and **Experiments** show every run and registry version.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard says **API offline** / "Cannot reach the API" | Start the server (step 2). In `npm run dev` mode the API must be on port 8000. |
| `http://127.0.0.1:8000` shows JSON instead of the dashboard | `frontend/dist` is missing. Run `cd frontend && npm install && npm run build`, or use `npm run dev`. |
| **Model unavailable** / `/health` returns 503 | No trained model yet. Run `python -m src.pipeline` or press *Run pipeline* in the dashboard. |
| `Can't find a usable init.tcl` (Windows) | Fixed in this version (matplotlib now always uses the headless `Agg` backend). |
| Port 8000 already in use | `python -m uvicorn src.api.main:app --port 8001` (and `VITE_API_PROXY` if using `npm run dev`). |
| PowerShell: *running scripts is disabled* | `Set-ExecutionPolicy -Scope Process Bypass`, then activate the venv again. |
| After **moving the project folder**, training fails writing MLflow artifacts (`D:/MLOOPS/...`) | The old experiment stored an absolute path. This version archives that experiment automatically and creates a new one. To start completely clean, delete `mlflow.db` and `mlruns/`, then run the pipeline. |
| scikit-learn `InconsistentVersionWarning` when loading the model | The saved model was trained with another scikit-learn version. Harmless for inference; retrain to remove it. |
| Docker: `mlflow.db` became a *directory* | Delete it, run `python -m src.pipeline` on the host first, then `docker compose up`. |
| Dashboard fonts look generic | Fonts load from Google Fonts; offline machines use the system fallback. Everything else works offline. |
