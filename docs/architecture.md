# Architecture Overview

The **Stock Prediction Pipelines** project utilizes a combination of AWS services, Apache Airflow, and OpenLineage to provide an end-to-end data pipeline for stock prediction modeling. The infrastructure is provisioned and managed using Terraform, ensuring a reproducible and scalable architecture.

## Key Components

### AWS Lambda
- **Purpose**: Ingest data from external APIs (Alpha Vantage), load data into AWS S3 and PostgreSQL, and transform data for downstream processes.
- **Architecture**: Multiple Lambda functions are deployed, each responsible for specific tasks like fetching data or moving it between S3 and the database. Lambda layers are used to manage dependencies shared across functions.
- **CI/CD**: Managed through GitHub Actions, ensuring the latest version of the Lambda functions is deployed with any code change.

### Apache Airflow
- **Purpose**: Orchestrates the execution of different ETL tasks using DAGs (Directed Acyclic Graphs).
- **Setup**: The Airflow environment is deployed using AWS MWAA (Managed Workflows for Apache Airflow), ensuring scalability and fault tolerance.
- **DAGs**: DAGs are responsible for scheduling tasks such as data ingestion, transformation, and predictive modeling.

### Data Lineage - TODO
- **OpenLineage and Marquez**: These tools are integrated to track and visualize the flow of data through the various stages of the pipeline. This helps in auditing, debugging, and ensuring data integrity throughout the pipeline.
  
### Database
- **PostgreSQL (RDS)**: The relational database is used for persistent storage of stock data and model results. PostgreSQL is deployed using AWS RDS, providing a scalable and managed database solution.

### Infrastructure as Code
- **Terraform**: The entire AWS infrastructure is defined as code using Terraform, allowing for easy provisioning, updating, and tearing down of resources. Terraform modules are used to manage the different components of the architecture.