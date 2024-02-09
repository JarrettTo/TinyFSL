#!/bin/bash

# Use the absolute path to your project directory
PROJECT_PATH="/home/jupyter-alyanna_mari_abalo-e312b/TinyFSL"

# Set PYTHONPATH to include the "slt" directory within your project
export PYTHONPATH="${PROJECT_PATH}/slt:${PYTHONPATH}"

# Run your Python module command
python -m slt.signjoey train changed_files/mobilenet.yaml
