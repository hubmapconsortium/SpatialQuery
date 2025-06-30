# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python package source code to the container
COPY . /app

# Install system dependencies for scientific computing packages
RUN apt-get update && apt-get install -y \
    g++ \
    liblapack-dev \
    libblas-dev \
    pkg-config \
    libhdf5-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Build and install the Python package
RUN python setup.py install

# Expose the port the app runs on
EXPOSE 8080

# Install Gunicorn
RUN pip install gunicorn

# Use Gunicorn as the entry point
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "spatial_q_api:app"]