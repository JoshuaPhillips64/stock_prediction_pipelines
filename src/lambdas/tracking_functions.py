import inspect
import logging
import traceback
import functools
import requests
import os
from typing import Optional, List, Dict, Any, Callable, Set, Tuple
from datetime import datetime
from pathlib import Path
import re

from openlineage.client.event_v2 import Dataset, RunEvent, RunState, Job, Run
from openlineage.client.uuid import generate_new_uuid
from openlineage.client import OpenLineageClient
from openlineage.client.facet import (
    SchemaField,
    SchemaDatasetFacet,
    OutputStatisticsOutputDatasetFacet,
    SqlJobFacet,
    SourceCodeLocationJobFacet
)
from openlineage.client.run import InputDataset, OutputDataset
import pandas as pd
from config import marquez_url, github_repo_url


class LineageTracker:
    """
    A singleton class to track data lineage using OpenLineage.

    This class provides methods to add input and output datasets,
    set job information, and emit lineage events to a Marquez server.

    Attributes:
        client (OpenLineageClient): The client used to emit events to Marquez.
        run_id (str): A unique identifier for the current run.
        producer (str): The name of the producer of the lineage events.
        namespace (str): The namespace for the job and datasets.
        job_name (str): The name of the current job.
        inputs (List[InputDataset]): A list of input datasets for the job.
        outputs (List[OutputDataset]): A list of output datasets for the job.
        job_facets (Dict): A dictionary of job facets.
    """

    _instance = None

    VALID_NAMESPACES = ["etl_jobs", "analytics_db", "dialer_db",
                        "dogacademy_db", "livevox_reporting", "tableau_online",
                        "zendesk_reporting","ussa_db",'google_sheets',"ippy_db",'trinet_reporting',
                        "amplitude_reporting"]

    def __new__(cls, *args, **kwargs):
        """
        Create a singleton instance of the LineageTracker.
        The purpose of this is to ensure that only one instance of the LineageTracker class ever exists. This is so the decorator plays nicely with other inputs.

        Here's how it works:
        The first time you try to create an instance of LineageTracker, cls._instance will be None, so a new instance is created and stored in cls._instance.
        For all subsequent attempts to create an instance, cls._instance will already exist, so the existing instance is returned instead of creating a new one.
        """
        if not cls._instance:
            cls._instance = super(LineageTracker, cls).__new__(cls)
        return cls._instance

    def __init__(self, marquez_url: str):
        """
        Initialize the LineageTracker.

        Args:
            marquez_url (str): The URL of the Marquez server.
        """
        if not hasattr(self, 'client'):
            self.client = OpenLineageClient(url=marquez_url)
            self.reset()

    def reset(self) -> None:
        """Reset the tracker for a new job."""
        self.run_id = str(generate_new_uuid())
        self.producer = "default_data_team_producer"
        self.namespace = "etl_jobs"
        self.job_name: Optional[str] = None
        self.inputs: List[InputDataset] = []
        self.outputs: List[OutputDataset] = []
        self.job_facets: Dict = {}

    def _validate_namespace(self, namespace: str) -> None:
        """
        Validate the given namespace against the list of valid namespaces.

        Args:
            namespace (str): The namespace to validate.

        Raises:
            ValueError: If the namespace is not in the list of valid namespaces.
        """
        if namespace not in self.VALID_NAMESPACES:
            raise ValueError(f"Invalid namespace: '{namespace}'. "
                             f"Please use one of the following namespaces: {', '.join(self.VALID_NAMESPACES)}. "
                             f"If you need to add a new namespace, please update the VALID_NAMESPACES list.")

    def set_job(self, job_name: Optional[str] = None) -> None:
        """
        Set the name of the job.

        If no name is provided, it will use the name of the calling function.

        Args:
            job_name (Optional[str]): The name of the job.
        """
        if job_name is None:
            calling_frame = inspect.currentframe().f_back
            job_name = calling_frame.f_code.co_name if calling_frame else "unknown_job"
        self.job_name = job_name

    def _extract_all_tables(self, sql_query: str) -> Tuple[Set[str], bool]:
        """
        Extract all table names from a SQL query.
        """
        # Remove comments
        sql_query = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE)
        sql_query = re.sub(r'/\*.*?\*/', '', sql_query, flags=re.DOTALL)

        #remove backticks
        if '`' in sql_query:
            logging.warning("Backticks were removed from the SQL query. This may affect parsing in some edge cases.")
        sql_query = sql_query.replace('`', '')

        needs_review = False

        # Define patterns
        cte_pattern = r'\b(\w+)\s+AS\s*\('
        table_pattern = r'\b(?:FROM|JOIN)\s+([`"\[\w\.]+)(?:\s+(?:AS\s+)?\w+)?'
        subquery_pattern = r'\(SELECT'

        # Find all CTEs
        ctes = set(re.findall(cte_pattern, sql_query, re.IGNORECASE))
        #print(f"CTEs found: {ctes}")  # Debugging line

        # Find all table names in FROM and JOIN clauses
        tables = set()
        for match in re.finditer(table_pattern, sql_query, re.IGNORECASE):
            table = match.group(1)
            # Handle quoted identifiers and schema qualifiers
            table = re.sub(r'^[`"\[]|[`"\]]$', '', table)
            tables.add(table.split('.')[-1])  # Take only the table name part
        #print(f"Tables found: {tables}")  # Debugging line

        # Check for subqueries
        if re.search(subquery_pattern, sql_query, re.IGNORECASE):
            needs_review = True

        # Remove CTEs from the table list
        actual_tables = tables - ctes

        # Filter out common SQL keywords and function calls
        keywords_to_exclude = {'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'TABLE', 'INTO', 'VALUES'}
        filtered_tables = {table for table in actual_tables if
                           table.upper() not in keywords_to_exclude and not table.endswith('(')}

        # Check for potentially complex structures
        if len(actual_tables) != len(
                filtered_tables) or 'UNION' in sql_query.upper() or 'INTERSECT' in sql_query.upper() or 'EXCEPT' in sql_query.upper():
            needs_review = True

        logging.info(f"Final filtered input tables: {filtered_tables}")  # Debugging line
        return filtered_tables, needs_review

    def add_input(self, name: str, namespace: str, df: pd.DataFrame = None) -> None:
        """
        Add an input dataset to the lineage tracker.

        Args:
            name (str): The name of the dataset.
            namespace (str): The namespace of the dataset.
            df (pd.DataFrame, optional): The DataFrame containing the dataset.
        """
        self._validate_namespace(namespace)
        dataset = self._create_dataset(name, namespace, df)
        self.inputs.append(InputDataset(namespace=namespace, name=name, facets=dataset.facets))
        logging.info(f"Added {name} dataset in {namespace} as an input for {self.job_name}")

    def generate_inputs_from_sql(self, sql_query: str, namespace: str) -> None:
        """
        Parse the SQL query, extract table names including those in CTEs, and add them as inputs.

        Args:
            sql_query (str): The SQL query to parse.
            namespace (str): The namespace for the extracted tables.
        """
        tables, needs_review = self._extract_all_tables(sql_query)
        for table in tables:
            self.add_input(name=table, namespace=namespace)

        if needs_review:
            print("WARNING: This query contains complex structures that may require manual review.")
            print("Please verify the following extracted tables:")
            for table in tables:
                print(f"  - {table}")

    def add_output(self, name: str, namespace: str, df: pd.DataFrame = None) -> None:
        """
        Add an output dataset to the lineage tracker.

        Args:
            name (str): The name of the dataset.
            namespace (str): The namespace of the dataset.
            df (pd.DataFrame, optional): The DataFrame containing the dataset.
        """
        self._validate_namespace(namespace)
        dataset = self._create_dataset(name, namespace, df)
        self.outputs.append(OutputDataset(namespace=namespace, name=name, facets=dataset.facets))
        logging.info(f"Added {name} dataset in {namespace} as an output for {self.job_name}")



    def _create_dataset(self, name: str, namespace: str, df: pd.DataFrame = None) -> Dataset:
        """
        Create a Dataset object with facets.

        Args:
            name (str): The name of the dataset.
            namespace (str): The namespace of the dataset.
            df (pd.DataFrame, optional): The DataFrame containing the dataset.

        Returns:
            Dataset: A Dataset object with appropriate facets.
        """
        facets = {}
        if df is not None and isinstance(df, pd.DataFrame):
            schema_fields = [
                SchemaField(name=col, type=str(dtype))
                for col, dtype in df.dtypes.items()
            ]
            schema_facet = SchemaDatasetFacet(fields=schema_fields)
            stats_facet = OutputStatisticsOutputDatasetFacet(
                rowCount=len(df),
                size=df.memory_usage(deep=True).sum()
            )
            facets["schema"] = schema_facet
            facets["stats"] = stats_facet

        return Dataset(name=name, namespace=namespace, facets=facets)

    def set_job_facets(self, sql: str = None, code_location: str = None):
        """
        Set job facets like SQL query and source code location.

        Args:
            sql (str, optional): The SQL query used in the job.
            code_location (str, optional): The source code location of the job.
        """
        if sql:
            self.job_facets["sql"] = SqlJobFacet(sql)
            # Always set the sourceCodeLocation, using the provided code_location or generating it

        self.job_facets["sourceCodeLocation"] = SourceCodeLocationJobFacet(
                "git",
                code_location or generate_github_path()
            )

    def emit_start(self) -> None:
        """Emit a START event to Marquez."""
        self._emit_event(RunState.START)

    def emit_complete(self) -> None:
        """Emit a COMPLETE event to Marquez."""
        self._emit_event(RunState.COMPLETE)

    def emit_fail(self, error: Exception) -> None:
        """
        Emit a FAIL event to Marquez.

        Args:
            error (Exception): The error that caused the failure.
        """
        self._emit_event(RunState.FAIL, error)

    def _emit_event(self, state: RunState, error: Optional[Exception] = None) -> None:
        """
        Emit an event to Marquez.

        Args:
            state (RunState): The state of the run (START, COMPLETE, or FAIL).
            error (Exception, optional): The error that caused a failure, if any.
        """
        if not self.job_name:
            raise ValueError("Job name not set. Call set_job() before emitting events.")

        run = Run(runId=self.run_id)
        job = Job(namespace=self.namespace, name=self.job_name, facets=self.job_facets)

        event = RunEvent(
            eventTime=datetime.now().isoformat(),
            producer=self.producer,
            run=run,
            job=job,
            eventType=state,
            inputs=self.inputs,
            outputs=self.outputs
        )

        if error:
            event.run.facets["errorMessage"] = {
                "_producer": "https://github.com/OpenLineage/OpenLineage/tree/0.10.0/client/python",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/ErrorMessageRunFacet.json",
                "message": str(error),
                "programmingLanguage": "Python",
                "stackTrace": traceback.format_exc()
            }

        try:
            self.client.emit(event)
            logging.info(f"Emitted {state} event for job {self.job_name}")
        except Exception as e:
            logging.error(f"Failed to emit {state} event for job {self.job_name}: {str(e)}")

def track_lineage(marquez_url: str) -> Callable:
    """
    A decorator to track lineage for a function.

    This decorator will emit START and COMPLETE/FAIL events to Marquez
    for the decorated function.

    Usage:
        @track_lineage(marquez_url)
        def my_etl_job():
            # ETL code here
            pass

    Args:
        marquez_url (str): The URL of the Marquez server.

    Returns:
        Callable: A decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = LineageTracker(marquez_url)
            tracker.reset()
            tracker.set_job(func.__name__)
            tracker.emit_start()
            try:
                result = func(*args, **kwargs)
                tracker.emit_complete()
                return result
            except Exception as e:
                tracker.emit_fail(e)
                raise
            finally:
                tracker.reset()
        return wrapper
    return decorator



def generate_github_path() -> str:
    """
    Generate a full GitHub URL path for the file where this function is called.

    Returns:
        str: A full GitHub URL path for the file where this function is called.
    """
    # Get the frame where this function is called
    caller_frame = inspect.currentframe().f_back

    # Get the filename of the caller
    caller_filename = caller_frame.f_code.co_filename

    # Check if we're in an interactive environment
    if caller_filename == '<stdin>' or caller_filename == '<input>':
        # We're in an interactive environment, so we can't determine the file path
        return f"{github_repo_url}/tree/master/app"

    # Determine the app root (assuming the function is called from within the 'app' directory)
    app_root = os.path.abspath(os.path.join(os.path.dirname(caller_filename), '..', '..'))
    while os.path.basename(app_root) != 'app' and app_root != os.path.dirname(app_root):
        app_root = os.path.dirname(app_root)

    # If we couldn't find an 'app' directory, use the current working directory
    if os.path.basename(app_root) != 'app':
        app_root = os.getcwd()

    try:
        rel_path = os.path.relpath(caller_filename, app_root)
    except ValueError:
        # If filename is on a different drive than app_root, we'll get a ValueError
        # In this case, we'll use the full path
        rel_path = caller_filename

    # Ensure the path uses forward slashes and starts with 'app/'
    github_path = 'app/' + Path(rel_path).as_posix().replace('\\', '/').lstrip('/')

    return f"{github_repo_url}/blob/master/{github_path}"


def get_downstream_impact(job_name: str, namespace: str) -> str:
    """Fetch and format all downstream impact information up to three levels deep."""
    logging.debug(f"Getting downstream impact for job {job_name} in namespace {namespace}")

    job_id = f"job:{namespace}:{job_name}"
    url = f"{marquez_url}/api/v1/lineage"
    params = {"nodeId": job_id, "depth": 3}  # Set depth to 3 for capturing deeper dependencies
    timeout_seconds = 3

    try:
        response = requests.get(url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        lineage = response.json()

        if not lineage or not lineage.get('graph'):
            return "No lineage information found."

        downstream_datasets = get_downstream_datasets(lineage, job_id)

        if not downstream_datasets:
            return "No downstream impacts found."

        impact = "\n".join(f"- {name} ({ns})" for ns, name in downstream_datasets)
        return impact

    except requests.Timeout as e:
        logging.error(f"Timeout occurred fetching downstream datasets: {str(e)}", exc_info=True)
        pass
    except requests.HTTPError as e:
        logging.error(f"HTTP Error fetching downstream datasets: {str(e)}", exc_info=True)
        pass
    except Exception as e:
        logging.error(f"Error fetching downstream datasets: {str(e)}", exc_info=True)
        pass

def get_downstream_datasets(lineage, job_id):
    outputs = []
    processed_jobs = set()

    def fetch_outputs(node_id, current_depth, max_depth):
        if current_depth > max_depth:
            return
        for node in lineage['graph']:
            if node['type'] == 'JOB' and node['id'] == node_id:
                for edge in node.get('outEdges', []):
                    # Process datasets produced by the job
                    for data_node in lineage['graph']:
                        if data_node['id'] == edge['destination'] and data_node['type'] == 'DATASET':
                            dataset = (data_node['data']['namespace'], data_node['data']['name'])
                            outputs.append(dataset)
                            # Now move to the jobs that use this dataset
                            for downstream_node in lineage['graph']:
                                for downstream_edge in downstream_node.get('inEdges', []):
                                    if downstream_edge['origin'] == data_node['id']:
                                        if downstream_node['type'] == 'JOB' and downstream_node['id'] not in processed_jobs:
                                            processed_jobs.add(downstream_node['id'])
                                            fetch_outputs(downstream_node['id'], current_depth + 1, max_depth)

    # Start from the initial job
    fetch_outputs(job_id, 1, 3)  # Adjust max_depth to explore deeper layers
    return outputs
