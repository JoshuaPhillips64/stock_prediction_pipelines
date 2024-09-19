output "vpc_id" {
  description = "The ID of the VPC."
  value       = aws_vpc.main.id
}

output "private_subnets" {
  description = "List of IDs of private subnets."
  value       = aws_subnet.private_subnets[*].id
}

output "emr_master_sg_id" {
  description = "Security Group ID for EMR master nodes."
  value       = aws_security_group.emr_master.id
}

output "emr_slave_sg_id" {
  description = "Security Group ID for EMR slave nodes."
  value       = aws_security_group.emr_slave.id
}

output "emr_instance_profile" {
  description = "Security Group ID for EMR slave nodes."
  value       = aws_iam_instance_profile.emr_instance_profile.name
}

output "emr_service_role_policy" {
  description = "Security Group ID for EMR slave nodes."
  value       = aws_iam_role_policy_attachment.emr_service_role_policy.policy_arn
}

output "emr_service_role" {
  description = "EMR Service Role"
  value       = aws_iam_role.emr_service_role.name
}