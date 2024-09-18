import os
import functools
from openlineage.client import set_producer, OpenLineageClient
from openlineage.airflow.extractors import ExtractorManager
from openlineage.airflow.utils import get_connection, try_import_from_string
from openlineage.airflow.version import __version__ as OPENLINEAGE_AIRFLOW_VERSION

OPENLINEAGE_CLIENT_CLASS = 'openlineage.client.OpenLineageClient'
ENV_KEY_OPENLINEAGE_URL = 'OPENLINEAGE_URL'

# Backwards compatibility for OpenLineage < 0.10.0
DEFAULT_OPENLINEAGE_URL = 'http://localhost:5000'
DEFAULT_PRODUCER = f'https://github.com/OpenLineage/openlineage-airflow/{OPENLINEAGE_AIRFLOW_VERSION}'

_client = None


def get_openlineage_client(openlineage_url=None):
    global _client
    if not _client:
        if openlineage_url is None:
            openlineage_url = os.getenv(ENV_KEY_OPENLINEAGE_URL, DEFAULT_OPENLINEAGE_URL)
        _client = OpenLineageClient(openlineage_url)
    return _client


def lineage_data(func=None, max_inlets=None, max_outlets=None):
    if func is None:
        return functools.partial(lineage_data,
                                 max_inlets=max_inlets,
                                 max_outlets=max_outlets)
    extractor_manager = ExtractorManager()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # For now, we only support functions that are a member of a class
        # This should be changed in the future to support more function types
        if args and not isinstance(args[0], object):
            return func(*args, **kwargs)
        operator = args[0]
        task = kwargs.get('task') or operator
        try:
            openlineage_url = get_connection(operator.openlineage_conn_id).get_url() if                operator.openlineage_conn_id else None
        except AttributeError:
            openlineage_url = None
        run_id = kwargs.get('run_id') or task.dag_id
        job_name = f'{operator.dag_id}.{operator.task_id}'
        task_id = f'{job_name}.{operator.task_id}'
        client = get_openlineage_client(openlineage_url)
        set_producer(DEFAULT_PRODUCER)
        extractor = extractor_manager.get_extractor_class(operator.__class__)
        if extractor:
            try:
                client.emit(
                    extractor(operator).extract_on_start(
                        task
                    )
                )
            except Exception:
                # Extractor can throw any exception,
                # which should not affect the task itself.
                # We catch it here but let it slip away
                pass
        result = func(*args, **kwargs)
        if extractor:
            try:
                client.emit(
                    extractor(operator, inlets=extractor.get_inlets(
                        operator, max_inlets
                    ), outlets=extractor.get_outlets(
                        operator, max_outlets
                    )).extract_on_complete(
                        task, result
                    )
                )
            except Exception:
                # Extractor can throw any exception,
                # which should not affect the task itself.
                # We catch it here but let it slip away
                pass
        return result
    return wrapper