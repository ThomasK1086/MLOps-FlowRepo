from prefect import flow, task
from task1 import main as imported_task1
import pandas as pd

@task
def task1(arg1):
    df = pd.read_csv('data_flow1/steam_games_dataset.csv', nrows=5)
    print(df)
    return imported_task1(arg1)


@flow(
    name='fizzbuzz',
    description='This is just for testing'
)
def run_flow(arg1, arg2):
    version = "1.0.1"
    task1(arg1)
    print(f"This is version {version}")
    print(f"Received {arg1, arg2}")
    return

def main(arg1, arg2, commit_id=None):
    run_flow(arg1, arg2)
    print(f"Current commitId: {commit_id}")


if __name__ == "__main__":
    main()

