FROM python:3.12-slim

# Set environment variables
ENV SPARK_VERSION=3.2.1
ENV HADOOP_VERSION=3.2
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin
ENV DEBIAN_FRONTEND=noninteractive
ENV POETRY_HTTP_TIMEOUT=600

# Update and install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        default-jdk \
        wget \
        curl \
        gnupg \
        libpq-dev \
        python3-distutils \
        build-essential \
        ca-certificates \
        gcc \
        python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \

RUN apt-get update && apt-get install -y python3-distutils

# Install Spark
RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Install Poetry
RUN pip install poetry

# Install PySpark manually to avoid incomplete read errors
RUN pip install pyspark==3.5.2

# Set working directory
WORKDIR /app

# Copy all files to make sure the emr_jobs directory exists
COPY . ./

# Install dependencies with retry mechanism
RUN poetry config virtualenvs.create false && poetry install

# Set the entrypoint
ENTRYPOINT ["spark-submit", "--master", "local[*]", "predictive_model.py"]