# Outputs for Lambda module
# ...
output "lambda_function_arns" {
  description = "ARNs of the deployed Lambda functions."
  value       = aws_lambda_function.ingest_alpha_vantage.arn
}

output "lambda_security_group_id" {
  description = "The security group ID for the Lambda function."
  value       = aws_security_group.lambda_sg.id
}
