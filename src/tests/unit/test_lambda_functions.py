import unittest
from unittest.mock import patch, MagicMock
import os

# Assuming your lambda functions are named like this:
# src/lambdas/ingest_alpha_vantage/handler.py
# src/lambdas/ingest_postgresql/handler.py
# etc.
lambda_functions = [
    'ingest_alpha_vantage',
    'ingest_postgresql',
    'ingest_s3'
]

class TestLambdaFunctions(unittest.TestCase):

    def test_lambda_handlers(self):
        for function_name in lambda_functions:
            # Dynamically import the lambda handler
            module_name = f'src.lambdas.{function_name}.handler'
            handler_module = __import__(module_name, fromlist=['lambda_handler'])
            lambda_handler = handler_module.lambda_handler

            # Mock environment variables
            with patch.dict(os.environ, {
                'DB_HOST': 'mock_db_host',
                'DB_PORT': '5432',
                'DB_NAME': 'mock_db_name',
                'DB_USER': 'mock_db_user',
                'DB_PASSWORD': 'mock_db_password',
                # Add other necessary environment variables
            }):
                # Create a mock event
                event = {}

                # Create a mock context
                context = MagicMock()

                # Call the lambda handler
                response = lambda_handler(event, context)

                # Assert the response
                self.assertEqual(response['statusCode'], 200)
                # Add more assertions based on your specific lambda functions