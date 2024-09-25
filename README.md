# Stock Prediction Pipelines on AWS with Terraform

## Overview

This repository contains a Python-based ETL pipelines that runs on AWS, utilizing Terraform for infrastructure as code. The pipeline is designed to handle multiple AWS Lambda functions with different layers, support large PySpark jobs on EMR, and orchestrate workflows using Apache Airflow. It also integrates OpenLineage and Marquez for data lineage tracking.

## Features

- **AWS Lambda Functions**: Modular Lambda functions with shared and individual dependencies managed via Poetry.
- **AWS EMR**: Scalable EMR clusters running PySpark jobs for heavy data processing.
- **Apache Airflow**: Workflow orchestration and scheduling.
- **OpenLineage & Marquez**: Data lineage and metadata management.
- **CI/CD Pipeline**: Automated deployments using Terraform and GitHub Actions.

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
