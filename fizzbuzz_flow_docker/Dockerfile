# Use a lightweight Python base image
FROM python:3.11.4-slim

# Set the working directory inside the container
# This matches the mount point from FlowExecutor ("/project")
WORKDIR /project

# Install any system dependencies if needed (optional, minimal by default)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy only the flow's requirements first (to leverage Docker layer caching)
COPY requirements.txt ./requirements.txt

# Install Python dependencies for the flow
RUN pip install uv
RUN uv pip install --no-cache-dir -r ./flows_git/fizzbuzz_flow_docker/requirements.txt --system

# Set default command to run the flow — this is overridden by FlowExecutor
CMD ["python", "flow.py"]
