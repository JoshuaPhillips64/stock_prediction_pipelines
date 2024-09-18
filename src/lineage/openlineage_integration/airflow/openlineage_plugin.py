from airflow.plugins_manager import AirflowPlugin
from openlineage.airflow import DAG as OpenLineageDAG
from openlineage.airflow import TaskInstance as OpenLineageTaskInstance
from openlineage.airflow.adapter import OpenLineageAdapter
from openlineage.airflow.extractors import ExtractorManager


class OpenLineagePlugin(AirflowPlugin):
    name = "OpenLineagePlugin"
    operators = []
    hooks = []
    executors = []
    macros = []
    admin_views = []
    flask_blueprints = []
    menu_links = []
    appbuilder_views = []
    appbuilder_menu_items = []

    # A list of global operator extractors for operator classes that don't have lineage decorators
    global_operator_extractors = []

    @classmethod
    def get_openlineage_adapter(cls) -> OpenLineageAdapter:
        return OpenLineageAdapter(
            ExtractorManager(cls.global_operator_extractors)
        )

    # This patch is needed to support compatibility with Airflow 1
    # Airflow 1 uses different variable names for task instances, so we need to patch them
    # to match the names used in Airflow 2
    @classmethod
    def patch(cls):
        try:
            from airflow.lineage import apply_lineage, prepare_lineage
            apply_lineage.TaskInstance = OpenLineageTaskInstance
            prepare_lineage.TaskInstance = OpenLineageTaskInstance
        except ImportError:
            # Lineage is not enabled in Airflow 1, so we don't need to patch
            pass
        try:
            from airflow import DAG
            DAG.DAG = OpenLineageDAG
        except ImportError:
            # Lineage is not enabled in Airflow 1, so we don't need to patch
            pass