# Use the official Apache Airflow image with Python 3.12
FROM apache/airflow:2.10.1-python3.12

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml ./

# Install Poetry
RUN pip install poetry

RUN rm -f poetry.lock

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install

# Copy your DAGs
COPY dags/ /opt/airflow/dags/

# Expose the Airflow webserver port
EXPOSE 8080

# Set the entrypoint
ENTRYPOINT ["airflow"]
CMD ["webserver"]