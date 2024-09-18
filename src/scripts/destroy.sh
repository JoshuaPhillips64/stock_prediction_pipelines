#!/bin/bash

set -e

echo "Destroying Terraform-managed infrastructure..."
cd terraform/ || exit
terraform destroy -auto-approve