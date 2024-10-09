#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to wait for the PostgreSQL database to be ready
wait_for_postgres() {
    echo "Waiting for PostgreSQL..."
    while ! nc -z postgres 5432; do
      sleep 1
    done
    echo "PostgreSQL is ready!"
}

# Initialize the Airflow database if it doesn't already exist
airflow_init() {
    echo "Initializing the Airflow database..."
    airflow db init
}

# Upgrade the Airflow database (optional step to handle migrations)
airflow_upgrade() {
    echo "Upgrading the Airflow database..."
    airflow db upgrade
}

#create admin user
create_admin_user() {
    echo "Creating Airflow admin user if it doesn't exist..."

    airflow users create \
        --username "${AIRFLOW_ADMIN_USERNAME:-admin}" \
        --firstname "${AIRFLOW_ADMIN_FIRSTNAME:-Admin}" \
        --lastname "${AIRFLOW_ADMIN_LASTNAME:-User}" \
        --role Admin \
        --email "${AIRFLOW_ADMIN_EMAIL:-admin@example.com}" \
        --password "${AIRFLOW_ADMIN_PASSWORD:-admin_password}" || echo "Admin user already exists."
}

# Run the command that was passed to the container (default: webserver)
exec_command() {
    exec "$@"
}

# Main entrypoint logic
if [ "$1" = "webserver" ]; then
    wait_for_postgres
    airflow_init
    create_admin_user
    exec_command airflow webserver
elif [ "$1" = "scheduler" ]; then
    wait_for_postgres
    airflow_init
    exec_command airflow scheduler
else
    exec_command "$@"
fi