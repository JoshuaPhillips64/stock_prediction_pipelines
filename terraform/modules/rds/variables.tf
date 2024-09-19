variable "db_name" {
  description = "The name of the database"
  type        = string
  default     = "stock-data"
}

variable "db_username" {
  description = "Username for the database"
  type        = string
}

variable "db_password" {
  description = "Password for the database"
  type        = string
  sensitive = true # Mark db_password as sensitive
}

variable "vpc_id" {
  description = "The VPC ID"
  type        = string
}

variable "private_subnets" {
  description = "List of private subnet IDs"
  type        = list(string)
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "allowed_security_groups" {
  description = "List of allowed security group IDs for ingress"
  type        = list(string)
}