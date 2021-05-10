from datetime import timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
import os
from astrowaves.config import ROOT_DATA_DIR
from astrowaves.airflow.utils import process_task_name


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
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
    "0_preprocess_czi",
    default_args=default_args,
    description="Preprocess CZI files by correcting movement and F0 correction",
    schedule_interval=timedelta(days=1),
)

filename = Variable.get("filename")

if filename == "all":
    files = [file for file in os.listdir(ROOT_DATA_DIR) if file.endswith(".czi")]
else:
    files = [filename]

intensity_correction_method = Variable.get("intensity_correction_method")
drift_correction_window_size = Variable.get("drift_correction_window_size")


for file in files:

    filename = os.path.join(ROOT_DATA_DIR, file)
    name = process_task_name(file)

    tiff_filename = filename + ".tif"

    # The task for performing f0 correction - either f0 or PAFFT
    t1 = BashOperator(
        task_id=f"correct_intensity_{name}",
        bash_command=f'python -m astrowaves.tasks.preprocessing.IntensityCorrector perform_intensity_correction "{filename}" --method {intensity_correction_method}',
        dag=dag,
    )

    # The task to perform drift correction for the timelapse
    t2 = BashOperator(
        task_id=f"correct_drift_{name}",
        bash_command=f'python -m astrowaves.tasks.preprocessing.DriftCorrector perform_drift_correction "{tiff_filename}" --window_size {drift_correction_window_size}',
        dag=dag,
    )

    t1 >> t2
