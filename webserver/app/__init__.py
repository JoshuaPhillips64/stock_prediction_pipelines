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
        from app.models import create_ai_stock_predictions_model
        app.AiStockPredictions = create_ai_stock_predictions_model()

        from app.routes import bp as main_bp
        app.register_blueprint(main_bp)

    return app