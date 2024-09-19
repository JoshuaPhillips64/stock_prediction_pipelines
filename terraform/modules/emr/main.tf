resource "aws_emr_cluster" "emr_cluster" {
  name          = "emr-cluster-${var.environment}"
  release_label = var.emr_release_label
  applications  = var.applications

  # Instance Groups
  instance_group {
    instance_role  = "MASTER"
    instance_type  = var.master_instance_type
    instance_count = 1

    ebs_config {
      size          = 50
      type          = "gp2"
      volumes_per_instance = 1
    }
  }

  instance_group {
    instance_role  = "CORE"
    instance_type  = var.core_instance_type
    instance_count = 2

    ebs_config {
      size          = 50
      type          = "gp2"
      volumes_per_instance = 1
    }
  }

  # Additional configuration
  log_uri = "s3://${var.logs_bucket}/emr-logs/"
  security_configuration = aws_emr_security_configuration.emr_security_configuration.name
  termination_protection = false
  keep_job_flow_alive_when_no_steps = false

  # Network configuration
  ec2_attributes {
    subnet_id                         = var.subnet_id
    emr_managed_master_security_group = var.emr_managed_master_security_group
    emr_managed_slave_security_group  = var.emr_managed_slave_security_group
    instance_profile                  = var.instance_profile
  }

  bootstrap_action {
    name = "Install Dependencies"
    script_bootstrap_action {
      path = "s3://${var.bootstrap_scripts_bucket}/install_dependencies.sh"
    }
  }

  tags = {
    Name        = "${var.environment}-emr-cluster"
    Environment = var.environment
  }
}

# Configure EMR Security Configuration (optional but recommended)
resource "aws_emr_security_configuration" "emr_security_configuration" {
  name = "emr-security-configuration-${var.environment}"
  configuration = <<EOF
{
  "EncryptionConfiguration": {
    "AtRestEncryptionConfiguration": {
      "S3EncryptionConfiguration": {
        "EncryptionMode": "SSE-S3"
      },
      "LocalDiskEncryptionConfiguration": {
        "EncryptionKeyProviderType": "AwsKms",
        "AwsKmsKey": "alias/aws/s3"
      }
    },
    "EnableInTransitEncryption": false,
    "EnableAtRestEncryption": true
  }
}
EOF
}

