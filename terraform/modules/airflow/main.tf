# Airflow resources
# Example using AWS MWAA (Managed Airflow)
resource "aws_mwaa_environment" "example" {
  name               = "${var.environment}-airflow-environment"
  execution_role_arn = aws_iam_role.mwaa_example.arn

  network_configuration {
    security_group_ids = [aws_security_group.mwaa_example.id]
    subnet_ids         = var.subnet_ids  # List of subnet IDs
  }

  dag_s3_path       = "s3://${var.dag_bucket}/dags"  # Replace with actual S3 bucket for DAGs
  source_bucket_arn = "arn:aws:s3:::${var.dag_bucket}"  # Replace with actual bucket ARN

  airflow_configuration_options = {
    "core.sql_alchemy_conn" = "postgresql+psycopg2://${var.db_username}:${var.db_password}@${var.db_host}:${var.db_port}/${var.db_name}"
  }

  max_workers = 2  # Adjust as needed

  logging_configuration {
    dag_processing_logs {
      log_level = "INFO"
      enabled   = true
    }
    scheduler_logs {
      log_level = "INFO"
      enabled   = true
    }
    webserver_logs {
      log_level = "INFO"
      enabled   = true
    }
    worker_logs {
      log_level = "INFO"
      enabled   = true
    }
  }
}

# IAM role for MWAA environment
resource "aws_iam_role" "mwaa_example" {
  name = "mwaa-example-role"
  assume_role_policy = jsonencode({
    "Version": "2012-10-17",
    "Statement": [
      {
        "Action": "sts:AssumeRole",
        "Principal": {
          "Service": "airflow.amazonaws.com"
        },
        "Effect": "Allow",
        "Sid": ""
      }
    ]
  })
}

# Security Group for MWAA
resource "aws_security_group" "mwaa_example" {
  name = "mwaa-example-sg"
  # ... (configure ingress/egress rules)
}
