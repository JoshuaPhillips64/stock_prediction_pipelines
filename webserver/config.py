import os
from dotenv import load_dotenv

# Get the path to the root folder (one level up from the webserver folder)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load the .env file from the root directory
load_dotenv(os.path.join(root_dir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # API Endpoints for generate_stock_prediction.py
    API_URL = os.getenv('API_URL')
    ADMIN_API_KEY = os.getenv('ADMIN_API_KEY')

#%% Print the contents of the above class
#print(Config.__dict__)