import os
import logging
from pyspark.sql import SparkSession
import psycopg2
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_db_connection():
    return psycopg2.connect(
        host=os.environ['DB_HOST'],
        port=os.environ['DB_PORT'],
        database=os.environ['DB_NAME'],
        user=os.environ['DB_USERNAME'],
        password=os.environ['DB_PASSWORD']
    )

def main():
    # Check for required environment variables
    required_env_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USERNAME', 'DB_PASSWORD']
    for var in required_env_vars:
        if var not in os.environ:
            raise EnvironmentError(f"Required environment variable {var} is not set")

    try:
        # Initialize Spark session
        spark = SparkSession.builder.appName("PredictiveModel").getOrCreate()

        # Load data from PostgreSQL
        jdbc_url = f"jdbc:postgresql://{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
        db_properties = {
            "user": os.environ['DB_USERNAME'],
            "password": os.environ['DB_PASSWORD'],
            "driver": "org.postgresql.Driver"
        }
        df = spark.read.jdbc(url=jdbc_url, table="stock_prices", properties=db_properties)

        # Preprocessing
        df = df.withColumn("open", df.open.cast("double"))
        df = df.withColumn("close", df.close.cast("double"))

        # Feature Engineering
        assembler = VectorAssembler(inputCols=["open"], outputCol="features")
        df = assembler.transform(df)

        # Train/Test Split
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

        # Model Training
        lr = LinearRegression(featuresCol="features", labelCol="close")
        lr_model = lr.fit(train_df)

        # Model Evaluation
        predictions = lr_model.transform(test_df)
        evaluation = lr_model.evaluate(test_df)
        logger.info(f"Model RMSE: {evaluation.rootMeanSquaredError}")

        # Save predictions back to PostgreSQL
        predictions_df = predictions.select("symbol", "date", "prediction").toPandas()

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                for _, row in predictions_df.iterrows():
                    cursor.execute(
                        """
                        INSERT INTO stock_predictions(symbol, date, predicted_close)
                        VALUES (%s, %s, %s)
                        ON CONFLICT(symbol, date)
                        DO UPDATE SET predicted_close = EXCLUDED.predicted_close;
                        """,
                        (row['symbol'], row['date'], row['prediction'])
                    )
            conn.commit()

        spark.stop()
        logger.info("Predictive modeling complete and results saved to PostgreSQL.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()