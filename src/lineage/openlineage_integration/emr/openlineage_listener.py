import os
import json
import logging
from openlineage.client import OpenLineageClient
from openlineage.client.run import RunEvent, RunState, Run, Job
from openlineage.common.dataset import Dataset, Field, Source
from openlineage.common.provider import Provider
from openlineage.common.models import DbColumn, DbTableSchema, DbTableName

log = logging.getLogger(__name__)

OPENLINEAGE_URL = os.environ.get('OPENLINEAGE_URL', 'http://localhost:5000')
OPENLINEAGE_NAMESPACE = os.environ.get('OPENLINEAGE_NAMESPACE', 'default')


def emit_openlineage_event(event_data):
    """Emits an OpenLineage event based on provided event data."""

    # Extract relevant information from event data
    # Example assumes a specific event structure, adjust based on your use case
    job_name = event_data.get('job_name')
    run_id = event_data.get('run_id')
    start_time = event_data.get('start_time')
    end_time = event_data.get('end_time')
    inputs = event_data.get('inputs', [])
    outputs = event_data.get('outputs', [])

    # Create OpenLineage datasets
    input_datasets = [create_openlineage_dataset(dataset) for dataset in inputs]
    output_datasets = [create_openlineage_dataset(dataset) for dataset in outputs]

    # Create OpenLineage job and run
    job = Job(namespace=OPENLINEAGE_NAMESPACE, name=job_name)
    run = Run(runId=run_id)

    # Create and emit OpenLineage run event
    run_event = RunEvent(
        eventType=RunState.COMPLETE,  # Adjust based on event type
        eventTime=end_time or start_time,
        run=run,
        job=job,
        producer='EMR OpenLineage Listener',  # Replace with your producer name
        inputs=input_datasets,
        outputs=output_datasets,
    )

    client = OpenLineageClient(OPENLINEAGE_URL)
    client.emit(run_event)
    log.info(f"Emitted OpenLineage event: {run_event}")


def create_openlineage_dataset(dataset_data):
    """Creates an OpenLineage Dataset from provided dataset data."""
    # Example assumes a specific dataset structure, adjust based on your use case
    data_source = dataset_data.get('data_source')
    database = data_source.get('database')
    schema = data_source.get('schema')
    table = data_source.get('table')
    columns = dataset_data.get('columns', [])

    # Create OpenLineage Source object
    source = Source(
        scheme=data_source.get('scheme', 'postgresql'),  # Adjust based on data source type
        authority=f"{database}:{data_source.get('port', '5432')}",
        connectionUrl=data_source.get('connectionUrl'),
    )

    # Create OpenLineage fields
    fields = [Field(name=column['name'], type=column.get('type')) for column in columns]

    # Create OpenLineage Dataset object
    return Dataset(
        namespace=OPENLINEAGE_NAMESPACE,
        name=f"{schema}.{table}",
        source=source,
        fields=fields,
        schema=DbTableSchema(
            schema_name=schema,
            table_name=DbTableName(schema, table),
            columns=[
                DbColumn(
                    name=field.name,
                    type=field.type,
                    description=field.description,
                ) for field in fields
            ],
        ),
    )