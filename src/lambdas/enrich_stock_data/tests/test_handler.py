import unittest
from unittest.mock import patch, MagicMock
from handler import lambda_handler

class TestEnrichStockData(unittest.TestCase):

    @patch('handler.requests.get')
    @patch('handler.get_db_connection')
    def test_lambda_handler(self, mock_get_db_connection, mock_requests_get):
        # Mock API response
        mock_requests_get.return_value.json.return_value = {
            'Time Series (Daily)': {
                '2022-01-01': {
                    '1. open': '100.0',
                    '2. high': '110.0',
                    '3. low': '90.0',
                    '4. close': '105.0',
                    '5. volume': '1000000'
                }
            }
        }

        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_db_connection.return_value = mock_conn

        event = {}
        context = {}
        response = lambda_handler(event, context)

        self.assertEqual(response['statusCode'], 200)
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()
        mock_cursor.close.assert_called()
        mock_conn.close.assert_called()

if __name__ == '__main__':
    unittest.main()