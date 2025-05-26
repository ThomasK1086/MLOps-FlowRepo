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
    version = "1.0.2"
    task1(arg1)
    print(f"This is version {version}")
    print(f"Received {arg1, arg2}")
    return

def run(arg1, arg2):
    run_flow(arg1, arg2)


if __name__ == "__main__":
    run()

