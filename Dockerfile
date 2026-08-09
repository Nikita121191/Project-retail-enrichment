FROM apache/airflow:3.3.0-python3.10

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    -r /tmp/requirements.txt

COPY src /opt/airflow/project/src
COPY dags /opt/airflow/dags