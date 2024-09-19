variable "environment" {
  type = string
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