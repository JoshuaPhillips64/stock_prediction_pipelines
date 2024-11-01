# Smartstockpredictor.com

A stock prediction system that ingests stock data, enriches it with various features including AI generated market sentiment,
trains predictive models (Regression and Binary Classification), generates AI-powered analysis, and serves predictions through a web application.

## Architecture

The system is designed using Flask, API Gateway, Lambda, Airflow, PostgreSQL, and S3.

![Architecture Diagram](docs/architecture.svg)

1. **Data Ingestion:** Lambda functions ingest historical stock data from Alpha Vantage and other sources.
2. **Data Enrichment:** Data is enriched with technical indicators, AI generated market sentiment, economic data, and more.
3. **Model Training:** Regression and Binary Classification models are trained using historical data and tuned with time series cross-validation. Airflow orchestrates daily model retraining.
4. **AI Analysis:** OpenAI's GPT-4o and Anthropic's Claude generate insightful explanations for predictions, enhancing interpretability.
5. **Web Application:** A Flask web application provides a user interface for interacting with the system and visualizing predictions.
6. **CI/CD:** GitHub Actions automates testing, building, and deployment of Lambda functions and the web application. Terraform manages infrastructure provisioning.

## Getting Started

- [Setup Guide](docs/setup_guide.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Architecture documentation](docs/architecture.md)

## Contributing

Please read the [contributing guidelines](docs/contributing.md) before making any changes.

## License

This project is licensed under the MIT License.

## Internal Tech Notes

If getting python 3.9 error then run after installing python to C drive

``poetry env use C:/Python312/python.exe``

Flask Commands to run to get db going after making model changes

``flask --app run.py db migrate -m "comment about migration"
``
