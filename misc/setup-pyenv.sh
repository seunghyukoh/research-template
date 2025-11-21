#!/bin/bash

source .env

ENV_PREFIX=${CONDA_PREFIX:-"$HOME/miniconda3"}/envs/$PROJECT_NICKNAME/bin

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M:%S')" "Conda could not be found. Please install Anaconda or Miniconda."
    exit 1
else
    echo "$(date '+%Y-%m-%d %H:%M:%S')" "Conda is installed."
fi

# Check if the environment $PROJECT_NICKNAME exists
if conda env list | grep -qE "(^|\s)$PROJECT_NICKNAME($|\s)"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S')" "Environment '$PROJECT_NICKNAME' exists. Updating environment..."
    conda env update -f environment.yaml --prune
else
    echo "$(date '+%Y-%m-%d %H:%M:%S')" "Environment '$PROJECT_NICKNAME' does not exist. Creating environment..."
    conda env create -f environment.yaml
fi

# Install flash-attn if gpu is available
if $ENV_PREFIX/python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S')" "GPU is available. Installing flash-attn..."
    $ENV_PREFIX/pip install flash-attn --no-build-isolation
else
    echo "$(date '+%Y-%m-%d %H:%M:%S')" "GPU is not available. Skipping flash-attn installation."
fi


# Install pre-commit hooks
$ENV_PREFIX/pre-commit install
