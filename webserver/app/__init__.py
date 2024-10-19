from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import Config

db = SQLAlchemy()
migrate = Migrate()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)

    with app.app_context():
        from app.models import (
            create_ai_analysis_model,
            create_trained_models_model,
            create_trained_models_binary_model,
            create_predictions_log_model,
            create_basic_stock_data_model,
            create_enriched_stock_data_model
        )
        app.AIAnalysis = create_ai_analysis_model()
        app.TrainedModels = create_trained_models_model()
        app.TrainedModelsBinary = create_trained_models_binary_model()
        app.PredictionsLog = create_predictions_log_model()
        app.BasicStockData = create_basic_stock_data_model()
        app.EnrichedStockData = create_enriched_stock_data_model()

        from app.routes import main_bp
        app.register_blueprint(main_bp)

    return app