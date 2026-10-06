import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Ensure project root is on sys.path for Airflow worker processes
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow import DAG
try:  # Airflow 2.x
    from airflow.operators.python import PythonOperator
except ImportError:  # Airflow 3.x (standard provider)
    from airflow.providers.standard.operators.python import PythonOperator
from src.utils.logger import get_logger

logger = get_logger("airflow_dag")

# Default DAG Task Arguments
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
    'start_date': datetime(2026, 1, 1),
}

def on_task_failure_callback(context):
    """Callback function triggered when any Airflow task fails."""
    task_id = context.get('task_instance').task_id
    dag_id = context.get('task_instance').dag_id
    execution_date = context.get('execution_date')
    exception = context.get('exception')
    logger.error(f"❌ AIRFLOW TASK FAILURE: DAG='{dag_id}', Task='{task_id}', ExecutionDate='{execution_date}', Error='{exception}'")

# Airflow Task Callable Functions (Calling modular src/ components)

def task_data_ingestion(**kwargs):
    """Airflow Task 1: Executes Data Ingestion."""
    logger.info("Executing Airflow Task: Data Ingestion...")
    from src.data.ingest import download_dataset, load_raw_data
    
    raw_path = download_dataset(force=False)
    df = load_raw_data(raw_path)
    
    # Push metadata to XCom for downstream tasks
    kwargs['ti'].xcom_push(key='raw_csv_path', value=str(raw_path))
    kwargs['ti'].xcom_push(key='raw_rows', value=len(df))
    logger.info(f"[OK] Data Ingestion Complete. Ingested {len(df)} rows to {raw_path}")

def task_data_validation(**kwargs):
    """Airflow Task 2: Executes Data Quality Validation."""
    logger.info("Executing Airflow Task: Data Validation...")
    from src.data.ingest import load_raw_data
    from src.data.validate import validate_dataset, save_quality_report
    
    raw_csv_path = kwargs['ti'].xcom_pull(key='raw_csv_path', task_ids='data_ingestion')
    if not raw_csv_path:
        from src.utils.config import load_config, get_path
        config = load_config()
        raw_csv_path = str(get_path(config["data"]["raw_dir"]) / config["data"]["dataset_filename"])
        
    df = load_raw_data(Path(raw_csv_path))
    is_valid, report = validate_dataset(df)
    save_quality_report(report)
    
    if not is_valid:
        issues = "; ".join(report.get("issues", []))
        raise ValueError(f"Airflow Task Failed: Data Quality Validation Errors: {issues}")
        
    logger.info("[OK] Data Quality Validation Passed.")

def task_preprocessing(**kwargs):
    """Airflow Task 3: Preprocessing & Domain Feature Engineering."""
    logger.info("Executing Airflow Task: Preprocessing & Feature Engineering...")
    from src.data.ingest import load_raw_data
    from src.features.build_features import preprocess_and_save
    
    raw_csv_path = kwargs['ti'].xcom_pull(key='raw_csv_path', task_ids='data_ingestion')
    if not raw_csv_path:
        from src.utils.config import load_config, get_path
        config = load_config()
        raw_csv_path = str(get_path(config["data"]["raw_dir"]) / config["data"]["dataset_filename"])
        
    df = load_raw_data(Path(raw_csv_path))
    artifacts = preprocess_and_save(df)
    logger.info(f"[OK] Preprocessing Complete. Engineered {len(artifacts['feature_names'])} features.")

def task_model_training(**kwargs):
    """Airflow Task 4: Candidate Model Training & MLflow Tracking."""
    logger.info("Executing Airflow Task: Model Training & Evaluation...")
    from src.models.train import train_and_evaluate_all
    
    eval_summary = train_and_evaluate_all()
    best_run_id = eval_summary["selected_run_id"]
    best_model_name = eval_summary["selected_model"]
    test_recall = eval_summary["selected_model_test_metrics"]["recall"]
    
    # Push to XCom
    kwargs['ti'].xcom_push(key='best_run_id', value=best_run_id)
    kwargs['ti'].xcom_push(key='best_model_name', value=best_model_name)
    kwargs['ti'].xcom_push(key='test_recall', value=test_recall)
    
    logger.info(f"[OK] Model Training Complete. Selected '{best_model_name}' (Run ID: {best_run_id}) with Test Recall={test_recall:.4f}")

def task_mlflow_registration(**kwargs):
    """
    Airflow Task 5: Model Registration & Production Version Alias Tagging.

    The training task already registers the winning model and sets '@Production' (see src/models/train.py).
    This task is therefore idempotent: it verifies the registry state and only registers when the alias
    does not yet point at the run produced upstream. (Previously it always re-registered, creating a
    duplicate model version on every DAG run.)
    """
    logger.info("Executing Airflow Task: MLflow Model Registration...")
    from src.models.register import register_model_from_run, get_production_model_info
    
    run_id = kwargs['ti'].xcom_pull(key='best_run_id', task_ids='model_training')
    if not run_id:
        raise ValueError("Cannot register model: Missing MLflow run_id from upstream training task.")

    prod_info = get_production_model_info("PredictiveMaintenanceModel")
    if prod_info and prod_info.get("run_id") == run_id:
        logger.info(f"✓ Run {run_id} is already registered as Version {prod_info['version']} @Production. Nothing to do.")
        return
        
    reg_info = register_model_from_run(
        run_id=run_id,
        registered_model_name="PredictiveMaintenanceModel",
        alias="Production",
        description="Airflow automated pipeline model deployment"
    )
    logger.info(f"✓ Model Registered as '{reg_info['model_name']}' Version {reg_info['version']} @Production")

# Initialize Airflow DAG
with DAG(
    dag_id='predictive_maintenance_mlops_pipeline',
    default_args=default_args,
    description='Automated Predictive Maintenance MLOps Ingestion, Training, Evaluation, and Registration Pipeline',
    schedule='0 0 * * 0',  # Weekly on Sunday at midnight (or manual trigger)
    catchup=False,
    max_active_runs=1,
    tags=['mlops', 'predictive-maintenance', 'xgboost', 'random-forest']
) as dag:

    ingest_task = PythonOperator(
        task_id='data_ingestion',
        python_callable=task_data_ingestion,
        on_failure_callback=on_task_failure_callback
    )

    validate_task = PythonOperator(
        task_id='data_validation',
        python_callable=task_data_validation,
        on_failure_callback=on_task_failure_callback
    )

    preprocess_task = PythonOperator(
        task_id='preprocessing',
        python_callable=task_preprocessing,
        on_failure_callback=on_task_failure_callback
    )

    train_task = PythonOperator(
        task_id='model_training',
        python_callable=task_model_training,
        on_failure_callback=on_task_failure_callback
    )

    register_task = PythonOperator(
        task_id='mlflow_registration',
        python_callable=task_mlflow_registration,
        on_failure_callback=on_task_failure_callback
    )

    # Define Linear Task Dependencies
    ingest_task >> validate_task >> preprocess_task >> train_task >> register_task
