import unittest
from unittest.mock import patch, MagicMock
from handler import lambda_handler
import io

class TestIngestS3(unittest.TestCase):

    @patch('handler.boto3.client')
    @patch('handler.get_db_connection')
    def test_lambda_handler(self, mock_get_db_connection, mock_boto3_client):
        # Mock S3 response
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            'Body': io.StringIO("col1,col2,col3
value1,value2,value3
")  # Example CSV data
        }
        mock_boto3_client.return_value = mock_s3

        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_db_connection.return_value = mock_conn

        # Invoke the Lambda handler
        event = {}
        context = {}
        response = lambda_handler(event, context)

        self.assertEqual(response['statusCode'], 200)

        # Assert that the database cursor executed the insert statement
        mock_cursor.execute.assert_called()

        # Assert that the database connection and cursor were closed
        mock_conn.commit.assert_called()
        mock_cursor.close.assert_called()
        mock_conn.close.assert_called()

if __name__ == '__main__':
    unittest.main()