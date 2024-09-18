#!/bin/bash

set -e  # Exit on error
set -o errexit  # Exit on error

# Function to create a timestamped directory
create_timestamped_dir() {
  timestamp=$(date +"%Y%m%d-%H%M%S")
  mkdir "build_$timestamp"
  echo "build_$timestamp"
}

# Check for AWS credentials
if ! aws sts get-caller-identity > /dev/null 2>&1; then
  echo "AWS credentials not found. Please configure your AWS CLI."
  exit 1
fi

# Deploy infrastructure with Terraform
echo "Deploying infrastructure with Terraform..."
cd terraform/ || exit
terraform init
terraform apply -auto-approve

# Build and package Lambda functions
echo "Building and packaging Lambda functions..."
cd ../src/lambdas/ || exit

for lambda_dir in */; do
  lambda_name=$(basename "$lambda_dir")
  echo "Processing Lambda function: $lambda_name"

  # Check for pyproject.toml (PyPoetry) and install dependencies
  if [ -f "$lambda_dir/pyproject.toml" ]; then
    echo "  - Found pyproject.toml. Running poetry install..."
    cd "$lambda_dir" || exit
    poetry install
    cd ..
  fi

  # Create a timestamped build directory
  build_dir=$(create_timestamped_dir)
  echo "  - Created build directory: $build_dir"

  # Copy Lambda function code to the build directory (excluding .venv)
  echo "  - Copying Lambda function code to build directory..."
  cp -r "$lambda_dir" "$build_dir"/  # Assuming you want to copy the entire directory
  find "$build_dir/$lambda_name" -name ".venv" -type d -exec rm -rf {} +

  # Create the zip archive
  echo "  - Creating zip archive: $lambda_name.zip"
  cd "$build_dir" || exit
  zip -r "../$lambda_name.zip" ./*
  cd ..

  echo "  - Lambda function $lambda_name packaged successfully."
done

# Clean up the build directory (optional)
# You might want to keep build artifacts for debugging
# rm -rf "build_*"