#!/bin/bash

if ! command -v conda &> /dev/null; then
    echo "conda could not be found. Please install Anaconda or Miniconda."
    exit 1
else
    echo "conda is installed."
fi

conda env create -f environment.yaml

source activate rtemp

pre-commit install
