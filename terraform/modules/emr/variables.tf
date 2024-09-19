variable "environment" {
  type = string
  default = "dev"
}

variable "subnet_id" {
  type = string
}

variable "logs_bucket" {
  type = string
}

variable "bootstrap_scripts_bucket" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "emr_managed_master_security_group" {
  type = string
}

variable "emr_managed_slave_security_group" {
  type = string
}

variable "instance_profile" {
  type = string
}

variable "ec2_key_name" {
  description = "Name of the SSH key pair for EMR"
  type        = string
}


variable "emr_release_label" {
  description = "The release label of the EMR cluster (e.g., emr-6.4.0)"
  type        = string
}

variable "applications" {
  description = "List of applications to be installed on the EMR cluster"
  type        = list(string)
}

variable "service_role" {
  description = "IAM Role for EMR service"
  type        = string
}