import os
import logging
from datetime import timedelta
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
import psycopg2
from sqlalchemy import create_engine, Column, String, Date, Float, Table, MetaData
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

metadata = MetaData()

enriched_stock_data = Table('enriched_stock_data', metadata,
                            Column('symbol', String, primary_key=True),
                            Column('date', Date, primary_key=True),
                            Column('open', Float),
                            Column('high', Float),
                            Column('low', Float),
                            Column('close', Float),
                            Column('volume', Float),
                            Column('upper_band', Float),
                            Column('lower_band', Float),
                            Column('adx', Float),
                            Column('sector_performance', Float),
                            Column('sp500_return', Float),
                            Column('nasdaq_return', Float),
                            Column('sentiment_score', Float),
                            Column('gdp_growth', Float),
                            Column('inflation_rate', Float),
                            Column('unemployment_rate', Float),
                            Column('put_call_ratio', Float),
                            Column('implied_volatility', Float),
                            Column('rsi', Float),
                            Column('macd', Float),
                            Column('macd_signal', Float),
                            Column('macd_hist', Float)
                            )


def get_db_connection():
    return psycopg2.connect(
        host=os.environ['DB_HOST'],
        port=os.environ['DB_PORT'],
        database=os.environ['DB_NAME'],
        user=os.environ['DB_USERNAME'],
        password=os.environ['DB_PASSWORD']
    )


def get_sqlalchemy_engine():
    return create_engine(
        f"postgresql://{os.environ['DB_USERNAME']}:{os.environ['DB_PASSWORD']}@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
    )


def prepare_data(spark: SparkSession, jdbc_url: str, db_properties: Dict[str, str]) -> F.DataFrame:
    df = spark.read.jdbc(url=jdbc_url, table="enriched_stock_data", properties=db_properties)

    # Convert string columns to appropriate types
    numeric_cols = [c.name for c in enriched_stock_data.columns if isinstance(c.type, Float)]
    for col in numeric_cols:
        df = df.withColumn(col, F.col(col).cast("double"))
    df = df.withColumn("date", F.to_date(F.col("date")))

    # Feature engineering
    df = df.withColumn("price_difference", F.col("high") - F.col("low"))
    df = df.withColumn("volatility", (F.col("high") - F.col("low")) / F.col("open"))
    df = df.withColumn("bollinger_bandwidth", (F.col("upper_band") - F.col("lower_band")) / F.col("close"))

    window_spec = Window.partitionBy("symbol").orderBy("date").rowsBetween(-4, 0)
    df = df.withColumn("volume_ma5", F.avg("volume").over(window_spec))
    df = df.withColumn("close_ma5", F.avg("close").over(window_spec))

    df = df.withColumn("returns",
                       (F.col("close") - F.lag("close", 1).over(Window.partitionBy("symbol").orderBy("date"))) /
                       F.lag("close", 1).over(Window.partitionBy("symbol").orderBy("date")))

    # Calculate the target variables (close price and direction after 30 days)
    df = df.withColumn("target_price", F.lead("close", 30).over(Window.partitionBy("symbol").orderBy("date")))
    df = df.withColumn("target_direction", F.when(F.col("target_price") > F.col("close"), 1.0).otherwise(0.0))

    # Drop rows with null values after feature engineering
    df = df.dropna()

    return df


def get_feature_columns() -> List[str]:
    return [
        "open", "high", "low", "volume", "price_difference", "volatility",
        "volume_ma5", "close_ma5", "returns", "upper_band", "lower_band",
        "adx", "sector_performance", "sp500_return", "nasdaq_return",
        "sentiment_score", "gdp_growth", "inflation_rate", "unemployment_rate",
        "put_call_ratio", "implied_volatility", "rsi", "macd", "macd_signal",
        "macd_hist", "bollinger_bandwidth"
    ]


def create_model_pipeline(feature_cols: List[str], model_type: str) -> Pipeline:
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features")
    scaler = StandardScaler(inputCol="assembled_features", outputCol="scaled_features")

    if model_type == "regression":
        model = GBTRegressor(featuresCol="scaled_features", labelCol="target_price", maxIter=100)
    elif model_type == "classification":
        model = GBTClassifier(featuresCol="scaled_features", labelCol="target_direction", maxIter=100)
    else:
        raise ValueError("Invalid model type. Choose 'regression' or 'classification'.")

    return Pipeline(stages=[assembler, scaler, model])


def time_series_split(df: F.DataFrame, n_splits: int) -> List[Tuple[F.DataFrame, F.DataFrame]]:
    df = df.orderBy("date")
    dates = df.select("date").distinct().collect()
    split_size = len(dates) // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        train_end_date = dates[split_size * (i + 1)].date
        test_end_date = dates[split_size * (i + 2)].date

        train = df.filter(F.col("date") <= train_end_date)
        test = df.filter((F.col("date") > train_end_date) & (F.col("date") <= test_end_date))
        splits.append((train, test))

    return splits


def train_and_evaluate_model(df: F.DataFrame, symbol: str, model_type: str) -> Tuple[Any, float]:
    symbol_df = df.filter(F.col("symbol") == symbol)
    feature_cols = get_feature_columns()
    pipeline = create_model_pipeline(feature_cols, model_type)

    paramGrid = ParamGridBuilder() \
        .addGrid(pipeline.getStages()[-1].maxDepth, [5, 10, 15]) \
        .addGrid(pipeline.getStages()[-1].minInstancesPerNode, [1, 2, 4]) \
        .build()

    if model_type == "regression":
        evaluator = RegressionEvaluator(labelCol="target_price", predictionCol="prediction", metricName="rmse")
    else:
        evaluator = BinaryClassificationEvaluator(labelCol="target_direction", rawPredictionCol="rawPrediction",
                                                  metricName="areaUnderROC")

    cv_splits = time_series_split(symbol_df, n_splits=5)

    best_model = None
    best_metric = float('inf') if model_type == "regression" else 0

    for train, test in cv_splits:
        cv = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            numFolds=3)

        model = cv.fit(train)
        predictions = model.transform(test)
        metric = evaluator.evaluate(predictions)

        if (model_type == "regression" and metric < best_metric) or (
                model_type == "classification" and metric > best_metric):
            best_model = model
            best_metric = metric

    return best_model, best_metric


def save_predictions(predictions_df: pd.DataFrame) -> None:
    engine = get_sqlalchemy_engine()
    metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    for _, row in predictions_df.iterrows():
        prediction = {
            'symbol': row['symbol'],
            'date': row['date'] + timedelta(days=30),
            'close': row['price_prediction'],
            'direction': row['direction_prediction']
        }
        session.merge(enriched_stock_data.insert().values(**prediction))

    session.commit()
    session.close()


def main():
    required_env_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USERNAME', 'DB_PASSWORD']
    for var in required_env_vars:
        if var not in os.environ:
            raise EnvironmentError(f"Required environment variable {var} is not set")

    spark = SparkSession.builder.appName("ImprovedPredictiveModel").getOrCreate()

    jdbc_url = f"jdbc:postgresql://{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
    db_properties = {
        "user": os.environ['DB_USERNAME'],
        "password": os.environ['DB_PASSWORD'],
        "driver": "org.postgresql.Driver"
    }

    df = prepare_data(spark, jdbc_url, db_properties)

    symbols = df.select("symbol").distinct().rdd.flatMap(lambda x: x).collect()

    regression_models = {}
    classification_models = {}

    for symbol in symbols:
        logger.info(f"Training models for {symbol}")
        regression_models[symbol], rmse = train_and_evaluate_model(df, symbol, "regression")
        logger.info(f"RMSE for {symbol}: {rmse}")

        classification_models[symbol], auc = train_and_evaluate_model(df, symbol, "classification")
        logger.info(f"AUC for {symbol}: {auc}")

    # Make predictions
    test_df = df.filter(F.col("date") == F.max("date").over(Window.partitionBy("symbol")))

    all_predictions = []
    for symbol in symbols:
        reg_predictions = regression_models[symbol].transform(test_df.filter(F.col("symbol") == symbol))
        class_predictions = classification_models[symbol].transform(test_df.filter(F.col("symbol") == symbol))

        combined_predictions = reg_predictions.select("symbol", "date", F.col("prediction").alias("price_prediction")) \
            .join(class_predictions.select("symbol", "date", F.col("prediction").alias("direction_prediction")),
                  ["symbol", "date"])

        all_predictions.append(combined_predictions)

    all_predictions_df = spark.createDataFrame([])
    for predictions in all_predictions:
        all_predictions_df = all_predictions_df.union(predictions)

    predictions_pandas = all_predictions_df.select("symbol", "date", "price_prediction",
                                                   "direction_prediction").toPandas()
    save_predictions(predictions_pandas)

    spark.stop()
    logger.info("Predictive modeling complete and results saved to PostgreSQL.")


if __name__ == "__main__":
    main()