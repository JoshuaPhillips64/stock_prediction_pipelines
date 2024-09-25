# Terraform configuration for AWS provider
terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket  = "terraform-setup-josh"
    key     = "terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region
}

# Include modules
module "network" {
  source      = "./modules/network"
  environment = var.environment
  vpc_cidr    = var.vpc_cidr
}

module "rds" {
  source                  = "./modules/rds"
  vpc_id                  = module.network.vpc_id
  private_subnets         = module.network.private_subnets
  environment             = var.environment
  db_username             = var.db_username
  db_password             = var.db_password
  allowed_security_groups = [module.lambda.lambda_security_group_id]
}

module "lambda" {
  source                = "./modules/lambda"
  environment           = var.environment
  vpc_id                = module.network.vpc_id
  subnet_ids            = module.network.private_subnets
  db_host               = module.rds.db_endpoint
  db_port               = module.rds.db_port
  db_name               = module.rds.db_name
  db_username           = var.db_username
  db_password           = var.db_password
  alpha_vantage_api_key = var.alpha_vantage_api_key
}

resource "aws_s3_bucket" "logs_bucket" {
  bucket = var.logs_bucket

  tags = {
    Name        = "${var.environment}-logs-bucket"
    Environment = var.environment
  }
}

resource "aws_s3_bucket" "bootstrap_scripts_bucket" {
  bucket = var.bootstrap_scripts_bucket

  tags = {
    Name        = "${var.environment}-bootstrap-scripts-bucket"
    Environment = var.environment
  }
}


module "emr_cluster" {
  source                            = "./modules/emr"
  environment                       = var.environment
  subnet_id                         = module.network.private_subnets[0]
  logs_bucket                       = aws_s3_bucket.logs_bucket.bucket
  bootstrap_scripts_bucket          = aws_s3_bucket.bootstrap_scripts_bucket.bucket
  vpc_id                            = module.network.vpc_id
  emr_managed_master_security_group = module.network.emr_master_sg_id
  emr_managed_slave_security_group  = module.network.emr_slave_sg_id
  instance_profile = module.network.emr_instance_profile
  emr_release_label                 = "emr-6.4.0"  # Replace with the desired release label
  applications                      = ["Hadoop", "Spark"]  # Example list of applications
  service_role                      = module.network.emr_service_role
  ec2_key_name                      = "emr-key-pair"
}

module "airflow" {
  source      = "./modules/airflow"
  environment = var.environment
  subnet_ids  = module.network.private_subnets
  db_host     = module.rds.db_endpoint
  db_port     = module.rds.db_port
  db_name     = module.rds.db_name
  db_username = var.db_username
  db_password = var.db_password
  dag_bucket = "ussa-data-processing-code-repository"
}

