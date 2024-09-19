# Variables for Airflow module

variable "environment" {
  description = "Deployment environment (e.g., dev, prod)"
  type        = string
  default     = "dev"
}

variable "subnet_ids" {
  description = "List of subnet IDs for MWAA"
  type        = list(string)
}

variable "dag_bucket" {
  description = "S3 bucket for Airflow DAGs"
  type        = string
}

variable "db_username" {
  description = "Database username for Airflow"
  type        = string
}

variable "db_password" {
  description = "Database password for Airflow"
  type        = string
  sensitive   = true
}

variable "db_host" {
  description = "Database host for Airflow"
  type        = string
}

variable "db_port" {
  description = "Database port for Airflow"
  type        = number
}

variable "db_name" {
  description = "Database name for Airflow"
  type        = string
}

