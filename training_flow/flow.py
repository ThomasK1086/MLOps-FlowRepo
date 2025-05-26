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



from task1 import main as run_data_tests
from task2 import main as train_model
from task3 import main as test_model

@task(
    name="Step 1 of MLOps Ex2"
)
def step_one(*args, **kwargs):
    run_data_tests(*args, **kwargs)

@task(
    name="Step 2 of MLOps Ex2"
)
def step_two(*args, **kwargs):
    model_info = train_model(*args, **kwargs)
    return model_info

@task(
    name="Step 3 of MLOps Ex2",
    retries=0,
    timeout_seconds=60,
)
def step_three(*args, **kwargs):
    metrics = test_model(*args, **kwargs)
    return metrics

@flow(
    name="MLOpsEx3 Training Flow",
    flow_run_name=f"MLOpsEx3 model training flow at {timestamp.strftime('%Y%m%d-%H%M')}",
    description="This flow includes data tests of the Steam Games Dataset, training of a Random Forest Model and validation of the model on a test set",
    version="2.0.0",
    retries=0,
    timeout_seconds=600
)
def myflow_runner(
        output_dir,
        outfile_name,
        report_name,
        model_name,
        cutoff_year=2020,
        commit_id=None
):
    output_dir_pth = Path(output_dir)

    step_one(output_dir_pth,
             outfile_name,
             report_name)

    model_training_results = step_two(output_dir_pth,
                                      outfile_name,
                                      model_name,
                                      cutoff_year,
                                      return_state=True)


    # Check whether or not the training step was successful
    # Load backup model in case the training failed
    if model_training_results.is_failed():
        model_alias = "Backup"
        model_version = None
        model_info = None
        train_metrics = {'accuracy': 0, 'balanced_accuracy': 0, "f1": 0}
        model_path_full = None
    else:
        model_info, train_metrics = model_training_results.result()
        model_version = model_info.registered_model_version
        model_alias = None
        model_path_full = f"models:/{model_name}/{model_version}"


    metrics = step_three(output_dir_pth,
                         outfile_name,
                         model_name,
                         model_version=model_version,
                         model_alias=model_alias,
                         cutoff_year=cutoff_year,
                         acc_threshold=0.9*train_metrics['accuracy'],
                         f1_threshold=0.9*train_metrics['f1'])

    flow_id = get_run_context().flow_run.id

    metadata = {
        "flow_run_id": str(flow_id),
        "kwargs": {
            "output_dir": output_dir,
            "outfile_name": outfile_name,
            "report_name": report_name,
            "model_name": model_name,
            "cutoff_year": cutoff_year
        },
        "git_commit_hexsha": commit_id,
        "metrics": {
            "accuracy": metrics[0],
            "balanced_accuracy": metrics[1],
            "f1-score": metrics[2]
        },
        "model_training_successful": not model_training_results.is_failed(),
        "model_path_full": model_path_full,
        "timestamp_start": timestamp.isoformat(),
        "timestamp_end": datetime.now().isoformat(),
    }

    # Convert to JSON string
    metadata_json = json.dumps(metadata, indent=2)

    # Create artifact with JSON content
    artifact_id = create_markdown_artifact(
        key="training_flow",
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