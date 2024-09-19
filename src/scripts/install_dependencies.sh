#!/bin/bash

# Install Python 3 and pip if not already installed
sudo yum install -y python3 python3-pip

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH (EMR bootstrap actions often don't have a persistent environment)
export PATH="$HOME/.local/bin:$PATH"

# Navigate to the EMR jobs directory
cd /path/to/emr_jobs  # You will need to ensure this path is correct on EMR

# Install dependencies using Poetry
poetry install