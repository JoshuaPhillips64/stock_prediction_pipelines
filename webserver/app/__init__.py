import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from datetime import timedelta
from config import Config

db = SQLAlchemy()
migrate = Migrate()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    app.secret_key = os.getenv('SECRET_KEY')

    # Configure session lifetime
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # Adjust as needed

    db.init_app(app)
    migrate.init_app(app, db)

    # Enable CORS
    CORS(app)

    @app.context_processor
    def inject_recaptcha_site_key():
        return dict(recaptcha_site_key=app.config.get('RECAPTCHA_SITE_KEY'))

    with app.app_context():
        from app.models import (
            create_ai_analysis_model,
            create_trained_models_model,
            create_trained_models_binary_model,
            create_predictions_log_model,
            create_basic_stock_data_model,
            create_enriched_stock_data_model,
            create_contact_messages_model
        )
        app.AIAnalysis = create_ai_analysis_model()
        app.TrainedModels = create_trained_models_model()
        app.TrainedModelsBinary = create_trained_models_binary_model()
        app.PredictionsLog = create_predictions_log_model()
        app.BasicStockData = create_basic_stock_data_model()
        app.EnrichedStockData = create_enriched_stock_data_model()
        app.ContactMessage = create_contact_messages_model()

        from app.routes import main_bp
        app.register_blueprint(main_bp)

    return app