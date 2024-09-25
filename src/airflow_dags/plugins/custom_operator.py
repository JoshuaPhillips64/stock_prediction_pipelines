from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults

class MyCustomOperator(BaseOperator):

    @apply_defaults
    def __init__(self,
                 my_operator_param: str,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_operator_param = my_operator_param

    def execute(self, context):

        print(f"MyCustomOperator executed with param: {self.my_operator_param}")