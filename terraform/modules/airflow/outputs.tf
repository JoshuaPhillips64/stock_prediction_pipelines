# Outputs for Airflow module
output "airflow_url" {
  value = aws_mwaa_environment.example.web_server_url
  description = "The web server URL of the MWAA environment."
}
