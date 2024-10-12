# Start from the AWS Lambda Python 3.12 base image on Amazon Linux 2023
FROM public.ecr.aws/lambda/python:3.12

# Set the working directory in the container
WORKDIR ${LAMBDA_TASK_ROOT}

# Install necessary tools using dnf (Amazon Linux 2023 uses dnf instead of yum)
RUN dnf update -y && \
    dnf install -y gcc openssl-devel bzip2-devel libffi-devel postgresql-devel && \
    dnf clean all

# Install Poetry
RUN pip install poetry

# Copy only requirements to cache them in docker layer
COPY pyproject.toml poetry.lock* ${LAMBDA_TASK_ROOT}/

# Install project dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --verbose

# Copy the rest of the application code
COPY ./ ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "ai_analysis_handler.lambda_handler" ]