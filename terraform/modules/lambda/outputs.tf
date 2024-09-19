# Outputs for Lambda module
# ...
output "lambda_function_arns" {
  description = "ARN of the deployed Lambda function"
  value       = aws_lambda_function.test_lambda.arn
}

output "lambda_security_group_id" {
  description = "Security Group ID for Lambda functions."
  value       = aws_security_group.lambda_sg.id
}
