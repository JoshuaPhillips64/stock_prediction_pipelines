output "vpc_id" {
  description = "The ID of the VPC."
  value       = module.network.vpc_id
}

output "lambda_function_arns" {
  description = "ARNs of the deployed Lambda functions."
  value       = module.lambda_functions.lambda_function_arns
}

output "emr_cluster_id" {
  description = "ID of the EMR cluster."
  value       = module.emr_cluster.emr_cluster_id
}

output "airflow_url" {
  description = "URL of the Airflow web interface."
  value       = module.airflow.airflow_url
}

output "db_endpoint" {
  description = "The endpoint of the RDS instance."
  value       = module.rds.db_endpoint
}