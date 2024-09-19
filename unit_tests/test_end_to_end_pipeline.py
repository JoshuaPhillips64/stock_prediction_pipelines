import unittest
import os
import psycopg2
from airflow.models import DagBag
from airflow.utils.state import State

class TestEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        # Load the DAGs
        self.dagbag = DagBag(dag_folder='src/airflow_dags/dags', include_examples=False)

        # Set up database connection
        self.db_host = os.environ.get('DB_HOST')
        self.db_port = os.environ.get('DB_PORT', 5432)
        self.db_name = os.environ.get('DB_NAME')
        self.db_user = os.environ.get('DB_USER')
        self.db_password = os.environ.get('DB_PASSWORD')
        self.conn = psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password
        )

    def tearDown(self):
        # Close the database connection
        self.conn.close()

    def test_data_ingestion_dag(self):
        dag_id = "data_ingestion_dag"
        dag = self.dagbag.get_dag(dag_id)
        self.assertIsNotNone(dag, f"DAG '{dag_id}' not found")

        # Run the DAG
        dag.clear(start_date=dag.start_date, end_date=dag.end_date)
        dag.run(start_date=dag.start_date, end_date=dag.end_date, ignore_ti_state=True)

        # Check if all tasks succeeded
        for task_instance in dag.get_task_instances():
            self.assertEqual(task_instance.state, State.SUCCESS, f"Task '{task_instance.task_id}' failed")

        # Check if data exists in the database
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM stock_prices;")
            count = cur.fetchone()[0]
            self.assertGreater(count, 0, "No data found in 'stock_prices' table after DAG run")

    def test_nightly_model_dag(self):
        dag_id = "nightly_predictive_model_dag"
        dag = self.dagbag.get_dag(dag_id)
        self.assertIsNotNone(dag, f"DAG '{dag_id}' not found")

        # Run the DAG
        dag.clear(start_date=dag.start_date, end_date=dag.end_date)
        dag.run(start_date=dag.start_date, end_date=dag.end_date, ignore_ti_state=True)

        # Check if all tasks succeeded
        for task_instance in dag.get_task_instances():
            self.assertEqual(task_instance.state, State.SUCCESS, f"Task '{task_instance.task_id}' failed")

        # Check if predictions exist in the database
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM stock_predictions;")
            count = cur.fetchone()[0]
            self.assertGreater(count, 0, "No predictions found in 'stock_predictions' table after DAG run")

if __name__ == "__main__":
    unittest.main()
