# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /insurance-policy-pricing-model

# Copy the requirements.txt file into the container
COPY requirements.txt /insurance-policy-pricing-model/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the src directory contents into the container at /app/src
COPY src/ /insurance-policy-pricing-model/src

# Copy the data directory contents into the container at /app/data
COPY data/ /insurance-policy-pricing-model/data

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run the inference script when the container launches
CMD ["python", "src/inference.py"]