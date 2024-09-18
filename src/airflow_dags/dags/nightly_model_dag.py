from airflow import DAG
from airflow.providers.amazon.aws.operators.emr import (
    EmrCreateJobFlowOperator,
    EmrAddStepsOperator,
    EmrTerminateJobFlowOperator,
)
from airflow.providers.amazon.aws.sensors.emr import EmrStepSensor
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
}

# EMR cluster configuration
CLUSTER_CONFIG = {
    "Name": "EMR-Cluster-For-Predictive-Model",
    "ReleaseLabel": "emr-6.4.0",  # Replace with desired EMR version
    "Applications": ["Spark", "Hadoop"],
    "Instances": {
        "InstanceGroups": [
            {
                "Name": "Master nodes",
                "Market": "ON_DEMAND",
                "InstanceRole": "MASTER",
                "InstanceType": "m5.xlarge",
                "InstanceCount": 1,
            },
            {
                "Name": "Worker nodes",
                "Market": "ON_DEMAND",
                "InstanceRole": "CORE",
                "InstanceType": "m5.xlarge",
                "InstanceCount": 2,
            },
        ],
        "KeepJobFlowAliveWhenNoSteps": False,
        "TerminationProtected": False,
    },
    "JobFlowRole": "EMR_EC2_DefaultRole",
    "ServiceRole": "EMR_DefaultRole",
}

JOB_STEPS = [
    {
        'Name': 'Run Predictive Model',
        'ActionOnFailure': 'CONTINUE',
        'HadoopJarStep': {
            'Jar': 'command-runner.jar',
            'Args': [
                'spark-submit',
                '--deploy-mode', 'cluster',
                's3://your-emr-scripts/predictive_model.py'  # Replace with actual S3 path
            ]
        }
    }
]


with DAG(
    'nightly_predictive_model_dag',
    default_args=default_args,
    description='DAG for running nightly predictive model on stock tickers',
    schedule_interval='0 2 * * *',
    start_date=days_ago(1),
    tags=['emr', 'predictive_model'],
) as dag:

    # Create EMR cluster
    create_cluster = EmrCreateJobFlowOperator(
        task_id='create_emr_cluster',
        job_flow_overrides=CLUSTER_CONFIG,
    )

    # Add steps to EMR cluster
    add_steps = EmrAddStepsOperator(
        task_id='add_steps',
        job_flow_id=create_cluster.output,
        steps=JOB_STEPS,
    )

    # Sensor to wait for steps to complete
    step_checker = EmrStepSensor(
        task_id='watch_step',
        job_flow_id=create_cluster.output,
        step_id=f"{{{{ task_instance.xcom_pull('add_steps')[0] }}}}",  # Access step ID dynamically
    )

    # Terminate EMR cluster
    terminate_cluster = EmrTerminateJobFlowOperator(
        task_id='terminate_emr_cluster',
        job_flow_id=create_cluster.output,
        trigger_rule='all_done',  # Terminate even if some tasks fail
    )

    # Define task dependencies
    create_cluster >> add_steps >> step_checker >> terminate_cluster