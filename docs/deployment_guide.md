# Deployment Guide

This guide walks you through deploying the **Stock Prediction Pipelines** project to AWS using Terraform and GitHub Actions for CI/CD.

## Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Python 3.12](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/#installation)
- [Terraform](https://www.terraform.io/downloads)
- AWS CLI with valid credentials

## Setup

1. **Terraform Initialization**:  
   Navigate to the `terraform` directory and initialize Terraform:

   ```bash
   cd terraform
   terraform init
   ```

2. **Terraform Apply**:  
   Apply the Terraform plan to provision AWS resources (RDS, EMR, S3, etc.):

   ```bash
   terraform apply
   ```

   This will set up the infrastructure required to run the pipeline, including:

   - AWS RDS (PostgreSQL) for data storage.
   - AWS VPC and Security Groups.
   - S3 buckets for storing data and logs.

3. **AWS Lambda Deployment**:  
   AWS Lambda functions are deployed via Terraform and code is loaded into ECS. Follow instructions in src/pipelines/README.md to deploy the Lambda functions.

4. **Airflow Deployment**:  
   Follow instructions in src/airflow/README.md to deploy Dags to Airflow.

5. **Webserver Deployment**:  
   Follow instructions in webserver/README.md to deploy code to EC2.

6. **Set Up GitHub Actions**:  
   The CI/CD pipeline is configured in `.github/workflows/ci-cd.yml`. When you push changes to the main branch, GitHub Actions will:

## Post-Deployment

After deployment, you can verify the pipeline by accessing:

- **Airflow UI**: Check the DAGs and task statuses.
- **PostgreSQL**: Ensure data is being ingested and the tables are populated.
