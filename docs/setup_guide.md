# Setup Guide

This guide explains how to set up the **Stock Prediction Pipelines** project on your local machine for development and testing.

## Prerequisites

Before setting up the project, make sure you have the following installed:

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Python 3.12](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/#installation)
- [Terraform](https://www.terraform.io/downloads)
- AWS CLI with valid credentials

## Clone the Repository

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/stock-prediction-pipelines.git
   cd stock-prediction-pipelines
   ```

2. Install the dependencies using Poetry:

   ```bash
   poetry install
   ```

## Configure Environment Variables

1. Create a `.env` file from the example file:

   ```bash
   cp .env.example .env
   ```

## Docker Compose Setup

Build and start services:

```bash
docker-compose up --build

docker-compose exec airflow airflow db init
```

This will:

- Start PostgreSQL
- Start Airflow
- Start the EMR Cluster Simulator
- Start the LAMBDA Simulator 

Access the Airflow UI at [http://localhost:8080](http://localhost:8080).

## Local Development

- **Run Tests**: Use `pytest` to run the tests:

  ```bash
  pytest
  ```

- **Linting**: Run `pylint` to ensure your code follows Python style guidelines:

  ```bash
  find . -name "*.py" -not -path "./venv/*" -exec pylint {} \;
  ```

- **Airflow DAG Testing**: To test DAGs locally, ensure that Airflow is running via Docker and execute the following:

  ```bash
  docker exec -it airflow_scheduler airflow dags trigger <dag_id>
  ```

## Conclusion

By following this guide, you can set up your local environment for the Stock Prediction Pipelines project, allowing you to develop, test, and run the pipeline locally before deploying to production.