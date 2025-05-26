from prefect import flow, task
from .task1 import main as imported_task1

@task
def task1():
    return imported_task1()


@flow
def run_flow():
    version = "1.0.1"
    print("Hello World!")
    task1()
    print("Bye World!")
    print("This is version {version}")
    return

if __name__ == "__main__":
    run_flow()