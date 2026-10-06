import pytest
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_dag_imports_and_structure():
    """Verifies that Airflow DAG file can be imported without syntax errors and has correct tasks & dependencies."""
    try:
        from dags.predictive_maintenance_dag import dag
    except ModuleNotFoundError:
        pytest.skip("Airflow not installed in local environment; skipping live DAG import test.")

    assert dag is not None
    assert dag.dag_id == "predictive_maintenance_mlops_pipeline"
    
    # Check Task Names
    task_ids = [t.task_id for t in dag.tasks]
    expected_tasks = ['data_ingestion', 'data_validation', 'preprocessing', 'model_training', 'mlflow_registration']
    assert set(expected_tasks).issubset(set(task_ids))

    # Check Task Dependencies
    ingest_task = dag.get_task('data_ingestion')
    validate_task = dag.get_task('data_validation')
    preprocess_task = dag.get_task('preprocessing')
    train_task = dag.get_task('model_training')
    register_task = dag.get_task('mlflow_registration')

    assert validate_task in ingest_task.downstream_list
    assert preprocess_task in validate_task.downstream_list
    assert train_task in preprocess_task.downstream_list
    assert register_task in train_task.downstream_list
