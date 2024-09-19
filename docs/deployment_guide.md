# Deployment Guide

This guide walks you through deploying the **Stock Prediction Pipelines** project to AWS using Terraform and GitHub Actions for CI/CD.

## Prerequisites

- [Terraform](https://www.terraform.io/downloads)
- AWS CLI configured with appropriate credentials.
- [GitHub Actions](https://docs.github.com/en/actions) configured for automated deployment.

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
   - AWS EMR cluster for running PySpark jobs.
   - S3 buckets for storing data and logs.

   > Note: You may need to adjust the `.tfvars` files based on your environment, e.g., `terraform/environments/dev/terraform.tfvars`.

3. **AWS Lambda Deployment**:  
   AWS Lambda functions are deployed via Terraform. After running `terraform apply`, the output will include the Lambda function ARNs.

4. **Set Up GitHub Actions**:  
   The CI/CD pipeline is configured in `.github/workflows/ci-cd.yml`. When you push changes to the main branch, GitHub Actions will:

   - Run tests.
   - Build and deploy the Lambda functions.
   - Apply the Terraform changes to AWS.

   Ensure you have GitHub Secrets set up for your AWS credentials:

   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

5. **Deploying Changes**:  
   Push code changes to the main branch, and the GitHub Actions pipeline will automatically deploy them to the production environment.

   ```bash
   git push origin main
   ```

## Post-Deployment

After deployment, you can verify the pipeline by accessing:

- **Airflow UI**: Check the DAGs and task statuses.
- **PostgreSQL**: Ensure data is being ingested and the tables are populated.
