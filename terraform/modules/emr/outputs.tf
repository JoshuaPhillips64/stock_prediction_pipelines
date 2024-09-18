# Outputs for EMR module
output "emr_cluster_id" {
  description = "The ID of the EMR cluster"
  value = aws_emr_cluster.emr_cluster.id
}
