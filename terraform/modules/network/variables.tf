variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
}

variable "environment" {
  description = "Deployment environment."
  type        = string
}

data "aws_availability_zones" "available" {
  state = "available"
}