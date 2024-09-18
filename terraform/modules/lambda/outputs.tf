# Outputs for Lambda module
# ...
output "lambda_function_arns" {
  description = "ARN of the deployed Lambda function"
  value       = aws_lambda_function.test_lambda.arn
}
