# Variables for EMR module
variable "environment" {
  type = string
}

variable "emr_release_label" {
  type    = string
  default = "emr-6.4.0" # From settings.yaml
}

variable "applications" {
  type    = list(string)
  default = ["Hadoop", "Spark"] # From settings.yaml
}

variable "master_instance_type" {
  type = string
  default = "m5.xlarge" # Adjust as needed
}

variable "core_instance_type" {
  type = string
  default = "m5.xlarge" # Adjust as needed
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