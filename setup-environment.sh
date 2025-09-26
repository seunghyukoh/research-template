#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M:%S')" "Conda could not be found. Please install Anaconda or Miniconda."
    exit 1
else
    echo "$(date '+%Y-%m-%d %H:%M:%S')" "Conda is installed."
fi

# Check if the environment 'rtemp' exists
if conda env list | grep -qE '(^|\s)rtemp($|\s)'; then
    echo "$(date '+%Y-%m-%d %H:%M:%S')" "Environment 'rtemp' exists. Updating environment..."
    conda env update -f environment.yaml
else
    echo "$(date '+%Y-%m-%d %H:%M:%S')" "Environment 'rtemp' does not exist. Creating environment..."
    conda env create -f environment.yaml
fi

# Install flash-attn if gpu is available
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S')" "GPU is available. Installing flash-attn..."
    pip install flash-attn --no-build-isolation
else
    echo "$(date '+%Y-%m-%d %H:%M:%S')" "GPU is not available. Skipping flash-attn installation."
    exit 0
fi

# Activate the environment
source activate rtemp

# Install pre-commit hooks
pre-commit install
