output "db_endpoint" {
  description = "The endpoint of the RDS instance"
  value       = aws_db_instance.postgres.address
}

output "db_port" {
  description = "The port of the RDS instance"
  value       = aws_db_instance.postgres.port
}

output "db_name" {
  value = aws_db_instance.postgres.identifier
}
