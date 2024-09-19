# Variables for Airflow module

variable "environment" {
  description = "Deployment environment (e.g., dev, staging, prod)"
  type        = string
}

variable "vpc_id" {
  description = "The ID of the VPC where Airflow will be deployed"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for Airflow deployment"
  type        = list(string)
}

variable "db_host" {
  description = "Hostname of the RDS database for Airflow"
  type        = string
}

variable "db_port" {
  description = "Port number of the RDS database"
  type        = number
}

variable "db_name" {
  description = "Name of the database for Airflow"
  type        = string
}

variable "db_username" {
  description = "Username for the RDS database"
  type        = string
}

variable "db_password" {
  description = "Password for the RDS database"
  type        = string
  sensitive   = true
}

variable "airflow_config_dir" {
  description = "Directory containing Airflow configuration files"
  type        = string
  default     = "../../src/airflow_dags/config"
}

