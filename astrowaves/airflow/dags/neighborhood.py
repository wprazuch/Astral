from datetime import timedelta
# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
# Operators; we need this to operate!
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
import os


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}


dag = DAG(
    'find_neighbours',
    default_args=default_args,
    description='Find neighbouring waves in a timespace',
    schedule_interval=timedelta(days=1),
)

rootdir = '/app/data'

filename = Variable.get("filename")

if filename == 'all':
    files = [file for file in os.listdir(rootdir) if file.endswith('.tif')]
else:
    files = [filename]

tolerance_xy = Variable.get("tolerance_xy")
tolerance_z = Variable.get("tolerance_z")


for file in files:
    filename = file
    directory = filename.split('.')[0]

    # t1, t2 and t3 are examples of tasks created by instantiating operators
    t1 = BashOperator(
        task_id=f'split_tiffs_{directory}',
        bash_command=f'python -m astrowaves.tasks.NeighbourFinder --directory {directory} --tolerance_xy {tolerance_xy} --tolerance_z {tolerance_z}',
        dag=dag,
    )
