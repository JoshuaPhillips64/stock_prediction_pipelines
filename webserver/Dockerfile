# Use a lightweight Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml ./

# Install Poetry
RUN pip install poetry

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install

# Copy the webserver code
COPY ./ ./

# Expose the port your Flask app listens on
EXPOSE 7237

# Install Gunicorn
RUN pip install gunicorn

# Command to run your Flask app using Gunicorn
CMD ["gunicorn", "--workers 3", "--bind", "0.0.0.0:7237", "run:app"]