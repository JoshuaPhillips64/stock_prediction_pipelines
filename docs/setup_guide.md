# Setup Guide

This guide explains how to set up the **Stock Prediction Pipelines** project on your local machine for development and testing.

## Prerequisites

Before setting up the project, make sure you have the following installed:

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Python 3.12](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/#installation)
- [Terraform](https://www.terraform.io/downloads)
- AWS CLI with valid credentials

## Clone the Repository

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/stock-prediction-pipelines.git
   cd stock-prediction-pipelines
   ```

2. Install the dependencies using Poetry:

   ```bash
   poetry install
   ```

## Configure Environment Variables

1. Create a `.env` file from the example file:

   ```bash
   cp .env.example .env
   ```
## **Lambda Setup and Local Development** 
   Follow instructions in src/pipelines/README.md to setup local environment.

## **Airflow Setup and Local Development**  
   Follow instructions in src/airflow/README.md to setup local environment.

## **Webserver Setup and Local Development** 
   Follow instructions in webserver/README.md to to setup local environment.
