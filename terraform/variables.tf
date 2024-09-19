variable "aws_region" {
  description = "The AWS region to deploy to."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)."
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.0.0.0/16"
}

variable "db_username" {
  description = "Username for the RDS database"
  type        = string
  default     = ""
}

variable "db_password" {
  description = "Password for the RDS database"
  type        = string
  sensitive   = true
  default     = ""
}

variable "alpha_vantage_api_key" {
  description = "API key for Alpha Vantage"
  type        = string
  sensitive   = true
  default     = ""
}

variable "logs_bucket" {
  description = "S3 bucket for EMR logs"
  type        = string
}

variable "bootstrap_scripts_bucket" {
  description = "S3 bucket for EMR bootstrap scripts"
  type        = string
}