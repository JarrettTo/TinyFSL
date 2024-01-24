#!/bin/bash

# Build the Docker image
docker build -t docker-image .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker image has been successfully built."
else
    echo "Failed to build the Docker image. Check the build logs for more details."
    exit 1  # Exit the script with an error code
fi

# Run the Python command inside the container
docker run -it docker-image python -m slt.signjoey train changed_files/sign.yaml
