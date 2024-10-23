# Smartstockpredictor.com

This repository contains a stock prediction system that ingests stock data, enriches it with various features including AI generated market sentiment,
trains predictive models (SARIMAX and XGBoost), generates AI-powered analysis, and serves predictions through a user-friendly web application.

## Architecture

The system is designed with a microservices architecture using Flask, AWS Lambda, Airflow, PostgreSQL, and S3.

1. **Data Ingestion:** Lambda functions ingest historical stock data from Alpha Vantage and other sources.
2. **Data Enrichment:** Data is enriched with technical indicators, AI generated market sentiment, economic data, and more.
3. **Model Training:** SARIMAX and XGBoost models are trained using historical data and tuned with time series cross-validation. Airflow orchestrates daily model retraining.
4. **AI Analysis:** OpenAI's GPT-4 and Anthropic's Claude generate insightful explanations for predictions, enhancing interpretability.
5. **Storage:** Processed data and trained models are stored in S3 and PostgreSQL.
6. **Web Application:** A Flask web application provides a user interface for interacting with the system and visualizing predictions.
7. **CI/CD:** GitHub Actions automates testing, building, and deployment of Lambda functions and the web application. Terraform manages infrastructure provisioning.

## Getting Started

- [Setup Guide](docs/setup_guide.md)
- [Deployment Guide](docs/deployment_guide.md)

## Project Structuree

Refer to the [architecture documentation](docs/architecture.md) for detailed information about the project structure.

## Contributing

Please read the [contributing guidelines](docs/contributing.md) before making any changes.

## License

This project is licensed under the MIT License.

## Tech Notes

If getting python 3.9 error then run after installing python to C drive

``poetry env use C:/Python312/python.exe``

Flask Commands to run to get db going after making model changes

``flask --app run.py db migrate -m "comment about migration"
``
