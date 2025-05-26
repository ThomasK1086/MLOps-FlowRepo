from prefect import flow, task
from .task1 import main as imported_task1

@task
def task1():
    return imported_task1()


@flow
def run_flow():
    print("Hello World!")
    task1()
    print("Bye World!")
    return

if __name__ == "__main__":
    run_flow()