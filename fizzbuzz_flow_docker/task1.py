from prefect import task

def main(arg1):
    print("This is task 1")
    print(f"Received {arg1}")
    return