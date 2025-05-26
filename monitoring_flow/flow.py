from prefect import flow, task
from datetime import datetime
timestamp = datetime.now()
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from prefect.artifacts import create_markdown_artifact, get_run_context


import os
import json
import sys



from task1 import main as run_drift_test
@task(
    name="Step 1 of Monitoring Flow"
)
def step_one(*args, **kwargs):
    run_drift_test(*args, **kwargs)


@flow(
    name="MLOpsEx3 Monitoring Flow",
    flow_run_name=f"MLOpsEx3 model monitoring flow at {timestamp.strftime('%Y%m%d-%H%M')}",
    description="This flow monitors the performance of a particular model on unseen data, by performing drift tests.",
    version="1.0.0",
    retries=0,
    timeout_seconds=600
)
def myflow_runner(
        working_dir,
        dataset_name,
        model_name,
        report_name,
        model_version=None,
        model_alias=None,
        model_path=None,
        cutoff_year=2020,
        commit_id=None
):
    step_one(working_dir,
             dataset_name,
             report_name,
             model_name,
             model_version=model_version,
             model_alias=model_alias,
             model_path=model_path,
             cutoff_year=cutoff_year)

    flow_id = get_run_context().flow_run.id

    metadata = {
        "flow_run_id": str(flow_id),
        "kwargs": {
            "working_dir": working_dir,
            "dataset_name": dataset_name,
            "report_name": report_name,
            "model_name": model_name,
            "cutoff_year": cutoff_year
        },
        "git_commit_hexsha": commit_id,
        "model_version": model_version,
        "model_alias": model_alias,
        "model_path_full": model_path,
        "timestamp_start": timestamp.isoformat(),
        "timestamp_end": datetime.now().isoformat(),
    }

    # Convert to JSON string
    metadata_json = json.dumps(metadata, indent=2)

    # Create artifact with JSON content
    artifact_id = create_markdown_artifact(
        key="monitoring_flow",
        markdown=f"```json\n{metadata_json}\n```",
        description="Flow metadata serialized as JSON"
    )

    # Dump the artifact locally too
    with open("Flow_Artifacts_Local.txt", "a+", encoding="utf-8") as f:
        f.write(metadata_json)

    with open("Flow_Ids.txt", "a+", encoding="utf-8") as f:
        f.write(str(flow_id))


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
    flow_id, artifact_id = myflow_runner(*args, **kwargs)

    print(f"Flow ID: {flow_id}, Artifact ID: {artifact_id}")


if __name__ == "__main__":
    main()