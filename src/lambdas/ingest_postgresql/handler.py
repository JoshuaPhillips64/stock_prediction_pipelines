import os
import logging
from common_utils.utils import get_db_connection

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    source_conn = get_db_connection()
    target_conn = get_db_connection()  # Assuming same function works for target

    source_cursor = source_conn.cursor()
    target_cursor = target_conn.cursor()

    source_cursor.execute("SELECT * FROM source_table;")
    rows = source_cursor.fetchall()

    for row in rows:
        target_cursor.execute(
            """
                             INSERT INTO target_table(column1, column2, ...)
            VALUES( % s, % s, ...)
    ON
    CONFLICT
    DO
    NOTHING;
    """,
    row
)
target_conn.commit()

source_cursor.close()
source_conn.close()
target_cursor.close()
target_conn.close()
logger.info("Data transfer complete.")

return {
'statusCode': 200,
'body': 'Data transfer complete'
}