import unittest
from unittest.mock import patch, MagicMock
from handler import lambda_handler

class TestIngestPostgresql(unittest.TestCase):

    @patch('handler.get_db_connection')
    def test_lambda_handler(self, mock_get_db_connection):
        # Mock database connections and cursors
        mock_source_conn = MagicMock()
        mock_target_conn = MagicMock()
        mock_source_cursor = MagicMock()
        mock_target_cursor = MagicMock()

        mock_source_conn.cursor.return_value = mock_source_cursor
        mock_target_conn.cursor.return_value = mock_target_cursor
        mock_get_db_connection.side_effect = [mock_source_conn, mock_target_conn]

        # Mock data returned from source table
        mock_source_cursor.fetchall.return_value = [
            ('value1', 'value2', 'value3'),  # Example row
            # Add more rows if needed
        ]

        # Invoke the lambda handler
        event = {}
        context = {}
        response = lambda_handler(event, context)

        self.assertEqual(response['statusCode'], 200)

        # Assert that the target cursor executed the insert statement
        mock_target_cursor.execute.assert_called()

        # Assert that connections and cursors were closed
        mock_source_conn.close.assert_called()
        mock_target_conn.close.assert_called()
        mock_source_cursor.close.assert_called()
        mock_target_cursor.close.assert_called()


if __name__ == '__main__':
    unittest.main()