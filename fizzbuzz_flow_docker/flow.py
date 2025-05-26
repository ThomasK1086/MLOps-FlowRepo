from prefect import flow, task
from task1 import main as imported_task1
import pandas as pd
from datetime import datetime
import sys
import json
import os


@task
def task1(arg1):
    return imported_task1(arg1)


@flow(
    name='fizzbuzz',
    description='This is just for testing'
)
def run_flow(arg1, arg2):
    version = "1.0.2"
    task1(arg1)
    print(f"This is version {version}")
    print(f"Received {arg1, arg2}")
    with open("out.txt", "w+") as outfile:
        outfile.write(f"This is version {version} at {datetime.now().isoformat()}")
    return


def main():
    if len(sys.argv) > 1:
        try:
            # sys.argv[1] is a JSON string
            param_blob = sys.argv[1]
            params = json.loads(param_blob)
            args = params.get("args", [])
            kwargs = params.get("kwargs", {})
        except Exception as e:
            print(f"❌ Failed to parse args: {e}")
            args = []
            kwargs = {}
    else:
        args = []
        kwargs = {}


    print(f"🚀 Running with args={args} kwargs={kwargs}")
    prefect_url = os.getenv("PREFECT_API_URL")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    print(f"Env variables are {prefect_url} and {mlflow_uri}")
    run_flow(*args, **kwargs)


if __name__ == "__main__":
    main()