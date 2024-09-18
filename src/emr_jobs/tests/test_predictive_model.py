import unittest
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession
from emr_jobs.predictive_model import main

class TestPredictiveModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder             .appName("TestPredictiveModel")             .master("local[*]")             .getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    @patch('emr_jobs.predictive_model.psycopg2.connect')
    def test_main(self, mock_psycopg2_connect):
        # Mock the database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg2_connect.return_value = mock_conn

        # Create a sample DataFrame
        data = [("AAPL", "2023-01-01", 150.0, 155.0, 149.0, 152.0, 100000),
                ("GOOG", "2023-01-01", 2500.0, 2550.0, 2490.0, 2520.0, 10000)]
        columns = ["symbol", "date", "open", "high", "low", "close", "volume"]
        sample_df = self.spark.createDataFrame(data, columns)

        # Mock the Spark read operation to return the sample DataFrame
        with patch('emr_jobs.predictive_model.spark.read.jdbc') as mock_read_jdbc:
            mock_read_jdbc.return_value = sample_df

            # Run the main function
            main()

            # Assertions
            mock_read_jdbc.assert_called_once()
            mock_cursor.execute.assert_called()
            mock_conn.commit.assert_called()

if __name__ == "__main__":
    unittest.main()