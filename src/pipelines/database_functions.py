import logging
from typing import List, Dict, Any, Optional, Callable
from sqlalchemy import Table, MetaData, inspect
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from io import StringIO
import csv
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_engine_from_url(db_url: str) -> Engine:
    """Create a SQLAlchemy engine from a database URL."""
    if not isinstance(db_url, str):
        raise TypeError("The 'db_url' must be a string.")
    return create_engine(db_url)

def execute_query(engine: Engine, query: str) -> None:
    """Execute a query without returning results."""
    with engine.begin() as conn:
        conn.execute(text(query))


def fetch_dataframe(engine: Engine, query: str) -> pd.DataFrame:
    """Execute a SQL query and return the results as a DataFrame."""
    if not isinstance(engine, Engine):
        raise TypeError("The 'engine' must be a SQLAlchemy Engine object.")
    if not isinstance(query, str):
        raise TypeError("The 'query' must be a string.")

    # Fetch data using the SQLAlchemy connection
    try:
        with engine.connect() as connection:
            df = pd.read_sql(text(query), connection)  # Use SQLAlchemy's text wrapper for safety
            return df
    except SQLAlchemyError as e:
        raise RuntimeError(f"Failed to execute query: {e}")


def psql_insert_copy(table: Table, conn: Any, keys: List[str], data_iter: Any) -> None:
    """
    Use PostgreSQL COPY command for faster data insertion.
    """
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join(f'"{k}"' for k in keys)
        table_name = f'{table.schema}.{table.name}' if table.schema else table.name

        cur.copy_expert(f"COPY {table_name} ({columns}) FROM STDIN WITH CSV", s_buf)


def create_df_from_query(engine: Engine, query: str) -> pd.DataFrame:
    """
    Execute a query and return results as a DataFrame.
    """
    start_time = pd.Timestamp.now()
    try:
        df = fetch_dataframe(engine, query)
        elapsed_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"Query executed successfully in {elapsed_time:.2f} seconds. Returned {len(df)} rows.")
        return df
    except SQLAlchemyError as e:
        logger.error(f"Error executing query: {str(e)}")
        raise


def get_table_data_types(engine: Engine, table: str, schema: Optional[str] = None) -> pd.DataFrame:
    """
    Get data types of columns for a given table.
    """
    query = f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = '{table}'
    """
    if schema:
        query += f" AND table_schema = {schema}"

    return fetch_dataframe(engine, query)


def match_db_data_types_to_pandas(df: pd.DataFrame, engine: Engine, table: str,
                                  schema: Optional[str] = None) -> pd.DataFrame:
    """
    Match database data types to pandas data types.
    """
    data_types_df = get_table_data_types(engine, table, schema)
    dtype_mapping = pd.read_csv('datatype_mapping.csv')

    db_to_pandas_mapping = dtype_mapping.set_index('Postgres Data Types')['Pandas Data Types'].to_dict()

    new_types = data_types_df.set_index('column_name')['data_type'].map(db_to_pandas_mapping)

    return df.astype(new_types.to_dict(), errors='ignore')


def overwrite_table_from_df(engine: Engine, df: pd.DataFrame, schema: str, table_name: str,
                            dtype_sync: bool = False) -> None:
    """
    Overwrite a table in PostgreSQL from a DataFrame.
    """
    with engine.begin() as conn:
        inspector = inspect(engine)

        if inspector.has_table(table_name, schema=schema):
            conn.execute(text(f"SELECT deps_save_and_drop_dependencies('{schema}', '{table_name}')"))

        if dtype_sync:
            df = match_db_data_types_to_pandas(df, engine, table_name, schema)

        df.to_sql(table_name, conn, schema=schema, if_exists='replace', index=False, method=psql_insert_copy)

        if inspector.has_table(table_name, schema=schema):
            conn.execute(text(f"SELECT deps_restore_dependencies('{schema}', '{table_name}')"))

        logger.info(
            f"Table {schema}.{table_name} {'created' if not inspector.has_table(table_name, schema=schema) else 'reloaded'} with {len(df)} rows.")


def truncate_and_load_table_from_df(engine: Engine, df: pd.DataFrame, schema: str, table_name: str) -> None:
    """
    Truncate and load a table in PostgreSQL from a DataFrame.
    """
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {schema}.{table_name}"))
        df.to_sql(table_name, conn, schema=schema, if_exists='append', index=False, method=psql_insert_copy)
        logger.info(f"Table {schema}.{table_name} truncated and loaded with {len(df)} rows.")


def append_data_from_df(engine: Engine, df: pd.DataFrame, schema: str, table_name: str) -> None:
    """
    Append data to a table in PostgreSQL from a DataFrame.
    """
    with engine.begin() as conn:
        df.to_sql(table_name, conn, schema=schema, if_exists='append', index=False, method=psql_insert_copy)
        logger.info(f"Appended {len(df)} rows to table {schema}.{table_name}.")


def upsert_df(df, table_name, upsert_id, postgres_connection, json_columns=[], auto_match_platform=None, auto_match_schema=None):
    """Implements the equivalent of pd.DataFrame.to_sql(..., if_exists='update')
    (which does not exist). Creates or updates the db records based on the
    dataframe records.
    Conflicts to determine update are based on the dataframes index.
    This will set unique keys constraint on the table equal to the index names
    1. Create the table if does not exist
    2. If does exist, Create a temp table from the dataframe
    3. Insert/update from temp table into table_name
    4. Deletes temp table
    Returns: True if successful
    """

    def constraint_exists(table_name, constraint_name, connection, schema=None):
        query = f"""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.table_constraints
                WHERE table_schema = '{schema or "public"}'
                AND table_name = '{table_name}'
                AND constraint_name = '{constraint_name}'
            )
        """
        return connection.execute(text(query)).scalar()

    if auto_match_platform is not None:
        df = match_db_data_types_to_pandas(df.copy(), postgres_connection, table_name, schema=auto_match_schema)

    con = postgres_connection.connect()

    try:
        # If the table does not exist, we should just use to_sql to create it
        if not con.execute(text(
            f"""SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE  table_schema = '{auto_match_schema or "public"}'
                AND    table_name   = '{table_name}');
                """
        )).scalar():
            for json_col in json_columns:
                df[json_col] = df[json_col].apply(json.dumps)

            df.to_sql(table_name, con, schema=auto_match_schema or 'public', if_exists='fail', index=True)

            for col in json_columns:
                con.execute(text(f"ALTER TABLE {auto_match_schema or 'public'}.{table_name} ALTER COLUMN {col} TYPE jsonb USING {col}::jsonb"))

            return True

        # If it already exists...
        temp_table_name = f"temp_{uuid.uuid4().hex[:6]}"
        number_of_rows = df.shape[0]
        # to using csv upload, can change psql_insert_copy to 'replace' to default
        df.to_sql(temp_table_name, con, schema=auto_match_schema or 'public', if_exists='fail', index=True)

        # Cast JSON columns back to jsonb
        for col in json_columns:
            con.execute(text(f"ALTER TABLE {auto_match_schema or 'public'}.{temp_table_name} ALTER COLUMN {col} TYPE jsonb USING {col}::jsonb"))

        index = list(df.index.names)
        index_sql_txt = f"{upsert_id}"
        columns = list(df.columns)
        headers = columns
        headers_sql_txt = ", ".join([f'"{i}"' for i in headers])

        update_column_stmt = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in columns])

        # In order to do on conflict update, we need to have a unique constraint
        constraint_name = f"unique_constraint_for_upsert_{table_name}"
        if not constraint_exists(table_name, constraint_name, con, schema=auto_match_schema):
            query_pk = f"""
            ALTER TABLE "{auto_match_schema or 'public'}"."{table_name}" ADD CONSTRAINT {constraint_name} UNIQUE ({index_sql_txt});
            """
            con.execute(text(query_pk))

        query_upsert = f"""
        INSERT INTO "{auto_match_schema or 'public'}"."{table_name}" ({headers_sql_txt})
        SELECT {headers_sql_txt} FROM "{auto_match_schema or 'public'}"."{temp_table_name}"
        ON CONFLICT ({index_sql_txt}) DO UPDATE
        SET {update_column_stmt};
        """
        con.execute(text(query_upsert))
        con.execute(text(f"DROP TABLE {auto_match_schema or 'public'}.{temp_table_name}"))

    except ValueError as vx:
        logger.error(vx)
        raise ValueError('error in loading to db') from vx
    except Exception as ex:
        logger.error(ex)
        raise RuntimeError('error in loading to db') from ex
    else:
        print(f"Table {table_name} upserted {number_of_rows} rows successfully.")
        return True
    finally:
        con.close()