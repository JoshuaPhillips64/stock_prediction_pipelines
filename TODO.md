# Configuration Checklist

## 1. Environment Variables

- **.env.example**: 
  - Replace all placeholders (e.g., `your-rds-endpoint`, `your-alpha-vantage-api-key`, etc.) with your actual values.
  - Rename the file to `.env` after updating.

- **terraform/variables.tf**: 
  - Update `db_username` and `db_password` with the correct credentials for your RDS instance.
  - You might need to adjust the `allowed_cidr_blocks` in module `"rds"` based on your network setup.

- **src/config/settings.yaml**: 
  - Update all placeholder values (e.g., `your-s3-bucket-name`, `your-rds-endpoint`, etc.) with your real settings.
  - If you change the default `emr_release_label` or `applications` in `terraform/modules/emr/variables.tf`, make sure to update `settings.yaml` to match.

- **src/lineage/marquez/config/marquez.yml**: 
  - Update the database connection details (`url`, `user`, `password`) to connect to your Marquez database.

## 2. Terraform

- **terraform/main.tf**: 
  - Update backend `"s3"` with the correct bucket name and key for your Terraform state.

- **terraform/environments//*.tfvars**: 
  - For each environment (dev, staging, prod), update the `terraform.tfvars` files with environment-specific values, including:
    - `aws_region`
    - `db_username`
    - `db_password` (in production, use a secure method like AWS Secrets Manager).

- **terraform/modules/lambda/main.tf**: 
  - Configure additional Lambda functions (besides the `test_lambda` example) with the appropriate handlers, runtimes, and other settings.
  - Define IAM roles for your Lambda functions in this module, granting the necessary permissions to access other AWS services (RDS, S3, etc.).

- **terraform/modules/emr/main.tf**: 
  - Configure your EMR cluster instance groups (master, core, task) with the desired instance types, counts, and configurations.
  - Adjust EMR configurations (e.g., `aws_emr_configuration`) and attach them to the appropriate instance groups.
  - If using different EMR steps or configurations, update `src/airflow_dags/dags/nightly_model_dag.py` with the correct EMR cluster configuration and job steps.

- **terraform/modules/airflow/main.tf**: 
  - The provided code shows an example of using AWS MWAA (Managed Airflow). If you're using a different Airflow setup, replace this module with the necessary Terraform resources for your Airflow environment.

- **terraform/modules/network/**: 
  - Review and adjust the network settings (VPC, subnets, security groups) to meet your security and networking requirements. Ensure that the security groups allow the necessary traffic between your resources.

## 3. Deployment Scripts

- **scripts/deploy.sh**: 
  - Update the script to include the steps to upload your EMR job files (`predictive_model.py` and others) to the correct S3 location (`your-emr-scripts`) as configured in `settings.yaml`.

- **scripts/ci-cd/pre-commit.sh** and **scripts/ci-cd/post-commit.sh**: 
  - Implement your custom pre-commit and post-commit hooks, such as linters, tests, and notifications.

## 4. Airflow

- **src/airflow_dags/dags/ingestion_dag.py** and **src/airflow_dags/dags/nightly_model_dag.py**: 
  - Ensure that the Lambda function names and EMR job file paths in the DAGs match your actual configurations.
  - Customize the DAG schedules (`schedule_interval`) as needed.

## 5. Testing

- **tests/**: 
  - Update all test files with assertions and logic relevant to your specific application and expected outcomes.

## Additional Considerations

- **IAM Roles**: 
  - Make sure all your AWS resources have properly configured IAM roles with the necessary permissions to interact with each other.

- **Security**: 
  - Review and tighten security groups, IAM policies, and other security-related settings.

- **Monitoring and Logging**: 
  - Set up appropriate monitoring and logging using CloudWatch or other tools to track your pipeline's performance and identify potential issues.

By carefully reviewing and updating these configurations, you'll ensure that your ETL pipeline is properly set up and ready to process data in your AWS environment. Remember to test each component thoroughly before deploying to production.