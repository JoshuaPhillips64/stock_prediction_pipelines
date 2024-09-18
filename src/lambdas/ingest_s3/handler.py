import os
import json
import logging
import boto3
import csv
from common_utils.utils import get_db_connection

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')

def lambda_handler(event, context):
    bucket_name = os.environ['S3_BUCKET_NAME']
    key = os.environ['S3_KEY']

    response = s3.get_object(Bucket=bucket_name, Key=key)
    content = response['Body'].read().decode('utf-8').splitlines()
    reader = csv.reader(content)

    conn = get_db_connection()
    cursor = conn.cursor()

    for row in reader:
        cursor.execute(
            """
                     INSERT INTO data_table(column1, column2, ...)
            VALUES( % s, % s, ...)
    ON
    CONFLICT
    DO
    NOTHING;
    """,
    row
)
conn.commit()

cursor.close()
conn.close()
logger.info("Data ingestion from S3 complete.")

return {
'statusCode': 200,
'body': 'Data ingestion from S3 complete'
}