output "vpc_id" {
  description = "The ID of the VPC."
  value       = module.network.vpc_id
}

output "lambda_function_arns" {
  description = "ARNs of the deployed Lambda functions."
  value       = module.lambda.lambda_function_arns
}

output "airflow_url" {
  description = "URL of the Airflow web interface."
  value       = module.airflow.airflow_url
}

output "db_endpoint" {
  description = "The endpoint of the RDS instance."
  value       = module.rds.db_endpoint
}