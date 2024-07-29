# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev

# Install Python packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the inference script
COPY src/inference.py /opt/program/inference.py

# Health check endpoint
COPY src/ping.py /opt/program/ping.py

# Copy the model file
COPY src/optimal-model-rfr.pkl /opt/ml/model/optimal-model-rfr.pkl

# Set the environment variable for the model path
ENV MODEL_PATH /opt/ml/model/optimal-model-rfr.pkl

# Set the entry point
ENV SAGEMAKER_PROGRAM inference.py

# Define the entry point
ENTRYPOINT ["python3", "/opt/program/inference.py"]