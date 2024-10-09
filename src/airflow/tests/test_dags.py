from datetime import datetime

from airflow.models import DAG, DagBag


def test_dag_loads_with_no_errors(dag_id="data_ingestion_dag"):
    dag_bag = DagBag(include_examples=False)
    dag_bag.load(dag_id=dag_id, root_path="/Users/jphil/Desktop/Python Projects/pythonProject/src/airflow_dags")
    assert dag_id in dag_bag.dags
    assert len(dag_bag.import_errors) == 0

def test_dag_task_count(dag_id="data_ingestion_dag"):
    dag_bag = DagBag()
    dag = dag_bag.get_dag(dag_id=dag_id)
    assert dag.tasks  # Check if tasks is not empty
    assert len(dag.tasks) == 3  # Expected number of tasks

def test_dag_schedule(dag_id="data_ingestion_dag"):
    dag_bag = DagBag()
    dag = dag_bag.get_dag(dag_id=dag_id)
    assert dag.schedule_interval == '@hourly'  # Expected schedule interval



def test_emr_dag_loads_with_no_errors(dag_id="nightly_predictive_model_dag"):
    dag_bag = DagBag(include_examples=False)
    dag_bag.load(dag_id=dag_id, root_path="/Users/jphil/Desktop/Python Projects/pythonProject/src/airflow_dags")
    assert dag_id in dag_bag.dags
    assert len(dag_bag.import_errors) == 0

def test_emr_dag_task_count(dag_id="nightly_predictive_model_dag"):
    dag_bag = DagBag()
    dag = dag_bag.get_dag(dag_id=dag_id)
    assert dag.tasks  # Check if tasks is not empty
    assert len(dag.tasks) == 4  # Expected number of tasks

def test_emr_dag_schedule(dag_id="nightly_predictive_model_dag"):
    dag_bag = DagBag()
    dag = dag_bag.get_dag(dag_id=dag_id)
    assert dag.schedule_interval == '0 2 * * *'  # Expected schedule interval
