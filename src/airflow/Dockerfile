# Use the official Apache Airflow image with Python 3.12
FROM apache/airflow:slim-2.10.1-python3.12

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \

# Install psycopg2-binary explicitly
RUN pip install psycopg2-binary

USER airflow

RUN pip install apache-airflow-providers-postgres \
    apache-airflow-providers-amazon

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml ./

# Install Poetry
RUN pip install poetry

RUN rm -f poetry.lock

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install

# Copy your DAGs and entrypoint script
COPY dags/ /opt/airflow/dags/
COPY entrypoint.sh /entrypoint.sh

USER root

# Make entrypoint executable
RUN chmod +x /entrypoint.sh

USER airflow

# Expose the Airflow webserver port
EXPOSE 8080

# Set entrypoint to entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
CMD ["webserver"]