from prefect import flow, task
from datetime import datetime
timestamp = datetime.now()
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from prefect.artifacts import create_markdown_artifact, get_run_context
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterId
from prefect.client.orchestration import get_client

import asyncio

import json
import sys
import re






from task1 import main as run_ab_test


@task(
    name="Step 1 of AB Test Flow"
)
def step_one(*args, **kwargs):
    return run_ab_test(*args, **kwargs)

def get_artifact(flow_run_id: str) -> dict:
    async def _read_json_from_artifact():
        result = None
        async with get_client() as client:
            artifacts = await client.read_artifacts(
                flow_run_filter=FlowRunFilter(
                    id=FlowRunFilterId(any_=[flow_run_id])
                )
            )
            if not artifacts or len(artifacts) == 0:
                raise ValueError(f"Encountered no artifacts for given flow run id: {flow_run_id}")
            elif len(artifacts) > 1:
                raise ValueError(f"Encountered more than one artifact for given flow run id: {flow_run_id}")

            artifact = artifacts[0]

            # Extract JSON from markdown code block
            match = re.search(r"```json\s*\n(.*?)\n```", artifact.data, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(1))
                except json.JSONDecodeError as e:
                    print(f"[ERROR] JSON decode error: {e}")
            else:
                print("[ERROR] No JSON code block found")
        return result

    artifact = asyncio.run(_read_json_from_artifact())

    if not artifact:
        raise FileNotFoundError(f"Could not load artifact for flow run ID '{flow_run_id}'")

    return artifact


@flow(
    name="MLOpsEx3 AB-Test Flow",
    flow_run_name=f"MLOpsEx3 model A/B test flow at {timestamp.strftime('%Y%m%d-%H%M')}",
    description="This flow performs an A/B test one two different models, splitting input data randomly.",
    version="1.0.0",
    retries=0,
    timeout_seconds=600
)
def myflow_runner(
        working_dir,
        dataset_name,
        flow_run_id_A,
        flow_run_id_B,
        hash_function_string = None,
        split_function_string = None,
        seed = 42,
        cutoff_year=2020,
        commit_id=None
):
    A_artifact = get_artifact(flow_run_id_A)
    A_modelpath = A_artifact["model_path_full"]

    B_artifact = get_artifact(flow_run_id_B)
    B_modelpath = B_artifact["model_path_full"]

    ab_testresults = step_one(
        working_dir=Path(working_dir),
        dataset_name=dataset_name,
        A_modelpath=A_modelpath,
        B_modelpath=B_modelpath,
        hash_function_string=hash_function_string,
        split_function_string=split_function_string,
        seed=seed,
        cutoff_year=cutoff_year,
    )


    flow_id = get_run_context().flow_run.id

    metadata = {
        "flow_run_id": str(flow_id),
        "kwargs": {
            "working_dir": working_dir,
            "dataset_name": dataset_name,
            "flow_run_id_A": flow_run_id_A,
            "flow_run_id_B": flow_run_id_B,
            "hash_function_string": hash_function_string,
            "split_function_string": split_function_string,
            "seed": seed,
            "cutoff_year": cutoff_year,
        },
        "Results": ab_testresults,
        "git_commit_hexsha": commit_id,
        "timestamp_start": timestamp.isoformat(),
        "timestamp_end": datetime.now().isoformat(),
    }

    # Convert to JSON string
    metadata_json = json.dumps(metadata, indent=2)

    # Create artifact with JSON content
    artifact_id = create_markdown_artifact(
        key="abtest-flow",
        markdown=f"```json\n{metadata_json}\n```",
        description="Flow metadata serialized as JSON"
    )

    # Dump the artifact locally too
    with open("Flow_Artifacts_Local.txt", "a+", encoding="utf-8") as f:
        f.write(metadata_json)

    with open("Flow_Ids.txt", "a+", encoding="utf-8") as f:
        f.write(str(flow_id) + '\n')


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



    #print(f"🚀 Running with args={args} kwargs={kwargs}")
    #prefect_url = os.getenv("PREFECT_API_URL")
    #mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    #print(f"Env variables are {prefect_url} and {mlflow_uri}")
    kwargs.update({"commit_id": commit_id})
    flow_id, artifact_id = myflow_runner(*args, **kwargs)

    print(f"Flow ID: {flow_id}, Artifact ID: {artifact_id}")


if __name__ == "__main__":
    main()