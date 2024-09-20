import os
from dotenv import load_dotenv

# Load environment variables only once
load_dotenv()

# Database configurations
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')

# Stock symbols
STOCK_SYMBOLS = os.getenv('STOCK_SYMBOLS').split(',')

# S3 Configurations
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_KEY = os.getenv('S3_KEY')

# AWS Configurations
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Environment setting
ENVIRONMENT = os.getenv('ENVIRONMENT', 'local')

# Log settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Timeout settings for API calls
API_CALL_TIMEOUT = int(os.getenv('API_CALL_TIMEOUT', 30)) 