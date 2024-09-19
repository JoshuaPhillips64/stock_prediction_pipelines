module "emr_cluster" {
  source  = "terraform-aws-modules/emr/aws"
  version = "~> 2.2.0"

  name = "emr-cluster-${var.environment}"

  release_label = "emr-6.4.0"  # EMR version

  applications = ["Hadoop", "Spark"]  # Required applications

  # Master and Core instance groups
  master_instance_group = {
    instance_type  = "m5.large"  # Smallest instance
    instance_count = 1
  }

  core_instance_group = {
    instance_type  = "m5.large"
    instance_count = 1
  }

  # Logging and network configurations
  log_uri = "s3://${var.logs_bucket}/emr-logs/"

  ec2_attributes = {
    key_name   = var.ec2_key_name  # SSH key name
    subnet_id  = var.subnet_id      # Subnet for EMR cluster
  }

  # Bootstrap actions for dependencies
  bootstrap_action = [
    {
      name = "Install Dependencies"
      path = "s3://${var.bootstrap_scripts_bucket}/install_dependencies.sh"
    }
  ]

  tags = {
    Environment = var.environment
  }
}
