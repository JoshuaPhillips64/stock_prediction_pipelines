# Architecture Documentation

## Overview

This document provides a detailed overview of the architecture for the Python ETL pipeline running on AWS.

## Components

- **AWS Lambda Functions**: Handle lightweight ETL tasks with modular functions.
- **AWS EMR**: Run PySpark jobs for heavy data processing.
- **Apache Airflow**: Orchestrate and schedule workflows.
- **OpenLineage & Marquez**: Track data lineage and metadata.
- **CI/CD Pipeline**: Use Terraform and GitHub Actions for automated deployments.

## Data Flow

1. **Data Ingestion**: Lambda functions ingest data from various sources.
2. **Processing**: EMR clusters process data using PySpark jobs.
3. **Orchestration**: Airflow orchestrates the workflow, triggering Lambdas and EMR jobs.
4. **Lineage Tracking**: OpenLineage and Marquez track data lineage throughout the pipeline.
5. **Monitoring**: CloudWatch monitors logs and metrics, providing alerts.

## Infrastructure

- **Networking**: All resources are deployed within a VPC for security.
- **Security**: IAM roles and policies provide fine-grained access control.
- **Scalability**: EMR clusters and Lambdas are configured to scale based on load.

Refer to the Visual Architecture Diagram for a high-level overview.