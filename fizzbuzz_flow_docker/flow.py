from prefect import flow, task
from task1 import main as imported_task1
import pandas as pd
from datetime import datetime
import sys
import json
import os
from prefect.artifacts import create_markdown_artifact, get_run_context


@task
def task1(arg1):
    return imported_task1(arg1)


@flow(
    name='fizzbuzz_in_dockercontainer',
    description='This is just for testing'
)
def run_flow(arg1, arg2, commit_id=None):
    version = "1.0.2"
    task1(arg1)
    print(f"This is version {version}")
    print(f"Received {arg1, arg2}")
    with open("out.txt", "w+") as outfile:
        outfile.write(f"This is version {version} at {datetime.now().isoformat()}")
    print(f"Commit ID/Hexsha: {commit_id}")

    flow_id = get_run_context().flow_run.id

    metadata = {
        "flow_run_id": flow_id,
        "inputs": {"arg1": arg1, "arg2": arg2},
        "version": version,
        "git_commit_hexsha": commit_id,
    }

    # Convert to JSON string
    metadata_json = json.dumps(metadata, indent=2)

    # Create artifact with JSON content
    artifact_id = create_markdown_artifact(
        key="flow-run-fizzbuzz",
        markdown=f"```json\n{metadata_json}\n```",
        description="Flow metadata serialized as JSON"
    )

    return flow_id, artifact_id



def main():
    if len(sys.argv) > 1:
        try:
            # sys.argv[1] is a JSON string
            param_blob = sys.argv[1]
            params = json.loads(param_blob)
            args = params.get("args", [])
            kwargs = params.get("kwargs", {})
            commit_id = params.get("commit_id", None)
        except Exception as e:
            print(f"❌ Failed to parse args: {e}")
            args = []
            kwargs = {}
            commit_id = None
    else:
        args = []
        kwargs = {}
        commit_id = None

    print(f"🚀 Running with args={args} kwargs={kwargs}")
    prefect_url = os.getenv("PREFECT_API_URL")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    print(f"Env variables are {prefect_url} and {mlflow_uri}")
    kwargs.update({"commit_id": commit_id})
    flow_id, artifact_id = run_flow(*args, **kwargs)

    print(f"Flow ID: {flow_id}, Artifact ID: {artifact_id}")

if __name__ == "__main__":
    main()
