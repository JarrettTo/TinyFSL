# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Set PYTHONPATH to include the directory where signjoey module is located
ENV PYTHONPATH="/app/slt:${PYTHONPATH}"

# Install system dependencies required for building numpy and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir pip==23.3.2 \
    && pip install --no-cache-dir setuptools wheel

# Copy the project files into the container
COPY . /app

# Install Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r slt/requirements.txt
