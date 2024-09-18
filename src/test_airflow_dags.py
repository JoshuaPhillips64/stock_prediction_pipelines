import unittest
from airflow.models import DagBag

class TestAirflowDags(unittest.TestCase):

    def test_dags_load_without_errors(self):
        dag_bag = DagBag(dag_folder='src/airflow_dags/dags', include_examples=False)
        assert len(dag_bag.import_errors) == 0, f"DAG import errors: {dag_bag.import_errors}"

    def test_data_ingestion_dag(self):
        dag_id = "data_ingestion_dag"
        dag_bag = DagBag(dag_folder='src/airflow_dags/dags', include_examples=False)
        dag = dag_bag.get_dag(dag_id=dag_id)
        self.assertIsNotNone(dag, f"DAG '{dag_id}' not found")
        self.assertEqual(len(dag.tasks), 3, f"DAG '{dag_id}' should have 3 tasks")
        self.assertEqual(dag.schedule_interval, '@hourly', f"DAG '{dag_id}' should be scheduled hourly")

    def test_nightly_model_dag(self):
        dag_id = "nightly_predictive_model_dag"
        dag_bag = DagBag(dag_folder='src/airflow_dags/dags', include_examples=False)
        dag = dag_bag.get_dag(dag_id=dag_id)
        self.assertIsNotNone(dag, f"DAG '{dag_id}' not found")
        self.assertEqual(len(dag.tasks), 4, f"DAG '{dag_id}' should have 4 tasks")
        self.assertEqual(dag.schedule_interval, '0 2 * * *', f"DAG '{dag_id}' should be scheduled daily at 2 AM")