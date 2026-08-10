from pathlib import Path

from airflow.sdk import dag, task


@dag(
    dag_id="smoke_test",
    schedule=None,
    catchup=False,
    tags=["smoke"],
)
def smoke_test():

    @task
    def check_project_paths():
        paths = [
            Path("/opt/airflow/project/src"),
            Path("/opt/airflow/project/data"),
            Path("/opt/airflow/project/artifacts"),
            Path("/opt/airflow/dags"),
        ]

        missing = [str(path) for path in paths if not path.exists()]

        if missing:
            raise FileNotFoundError(
                f"Required paths are missing: {missing}"
            )

        print("All required project paths exist.")

        for path in paths:
            print(f"OK: {path}")

        return [str(path) for path in paths]


    @task
    def confirm_execution(paths):
        print("Smoke test task executed successfully.")
        print("Received paths from previous task:")

        for path in paths:
            print(path)


    checked_paths = check_project_paths()
    confirm_execution(checked_paths)


smoke_test()