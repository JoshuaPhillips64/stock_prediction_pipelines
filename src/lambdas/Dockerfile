# Use a Debian-based image that supports Python 3.12
FROM python:3.12-slim

# Install wget and other necessary packages
RUN apt-get update && apt-get install -y wget gcc python3-dev && apt-get clean

# Install Poetry
RUN pip install poetry

# Copy the common layer
COPY common_layer/ /var/task/common_layer/

# Copy all lambdas and install their dependencies
COPY ingest_alpha_vantage/ /var/task/ingest_alpha_vantage/
RUN cd /var/task/ingest_alpha_vantage && \
    poetry config virtualenvs.create false && \
    poetry install

COPY ingest_postgresql/ /var/task/ingest_postgresql/
RUN cd /var/task/ingest_postgresql && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

COPY ingest_s3/ /var/task/ingest_s3/
RUN cd /var/task/ingest_s3 && \
    poetry config virtualenvs.create false && \
    poetry install

# Copy the AWS Lambda Runtime Interface Emulator (RIE) for local testing
ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/local/bin/aws-lambda-rie
RUN chmod +x /usr/local/bin/aws-lambda-rie

# Set the entrypoint to use RIE
ENTRYPOINT [ "/usr/local/bin/aws-lambda-rie", "python", "-m", "awslambdaric" ]

# Set the default handler (can be overridden)
ENV LAMBDA_HANDLER="ingest_alpha_vantage.handler.lambda_handler"

CMD [ "${LAMBDA_HANDLER}" ]