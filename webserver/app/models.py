from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import MetaData, Table
import json
from datetime import datetime

db = SQLAlchemy()
metadata = MetaData()

def create_dynamic_model(table_name):
    """Creates a dynamic model for the given table name."""
    class DynamicModel(db.Model):
        __table__ = Table(table_name, metadata, autoload_with=db.engine, autoload=True)
        __tablename__ = table_name

        def to_dict(self):
            data = {c.name: getattr(self, c.name) for c in self.__table__.columns}
            # Convert JSON fields to dictionaries if necessary
            for col in self.__table__.columns:
                if str(col.type) in ('JSON', 'JSONB'):
                    if data[col.name] and isinstance(data[col.name], str):
                        data[col.name] = json.loads(data[col.name])
            return data

        def __repr__(self):
            pk = self.__table__.primary_key.columns.keys()
            pk_values = ", ".join(f"{key}={getattr(self, key)}" for key in pk)
            return f"<{table_name.capitalize()} {pk_values}>"
    return DynamicModel

def create_ai_analysis_model():
    return create_dynamic_model('ai_analysis')

def create_trained_models_model():
    return create_dynamic_model('trained_models')

def create_trained_models_binary_model():
    return create_dynamic_model('trained_models_binary')

def create_predictions_log_model():
    return create_dynamic_model('predictions_log')

def create_basic_stock_data_model():
    return create_dynamic_model('basic_stock_data')

def create_enriched_stock_data_model():
    return create_dynamic_model('enriched_stock_data')

def create_contact_messages_model():
    return create_dynamic_model('contact_messages')