resource "aws_emr_cluster" "emr_cluster" {
  name          = "emr-cluster-${var.environment}"
  release_label = var.emr_release_label
  applications  = var.applications

  # Define Instance Groups based on your requirements (adjust instance types and counts)
  instance_group {
    instance_role = "MASTER"
    instance_type = var.master_instance_type
    instance_count = 1

    ebs_config {
      ebs_optimized = true
      volume_size   = 50
      volume_type   = "gp2"
    }
  }

  instance_group {
    instance_role = "CORE"
    instance_type = var.core_instance_type
    instance_count = 2

    ebs_config {
      ebs_optimized = true
      volume_size   = 50
      volume_type   = "gp2"
    }
  }

  # Additional configuration (like LogUri, SecurityConfiguration)
  log_uri                  = "s3://${var.logs_bucket}/emr-logs/"
  security_configuration   = aws_emr_security_configuration.emr_security_configuration.id
  termination_protection = false
  keep_job_flow_alive_when_no_steps = false

  # Network configuration
  ec2_attributes {
    subnet_id = var.subnet_id
  }

  bootstrap_action {
    name = "Install Dependencies"
    script_bootstrap_action {
      path = "s3://${var.bootstrap_scripts_bucket}/install_dependencies.sh" # Replace with your bootstrap script path
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
  security_configuration = <<EOF
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

