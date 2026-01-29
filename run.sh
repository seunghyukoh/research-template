#!/bin/bash

set -euo pipefail

IMAGE_NAME="research-dev"
CONTAINER_NAME="research-template-dev"

if [ ! -f .env ]; then
    echo "Warning: .env file not found. Copying from .env.example"
    cp .env.example .env
    echo "Please edit .env with your credentials before running again."
    exit 1
fi

set +u
source .env
set -u

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Warning: WANDB_API_KEY not set in .env"
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Warning: HF_TOKEN not set in .env (required for private models)"
fi

echo "Building Docker image..."
docker build -t ${IMAGE_NAME} .

echo ""
echo "Detecting GPU availability..."

GPU_FLAGS=""
NVIDIA_ENV=""

if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✓ GPU detected. Using GPU acceleration."
    GPU_FLAGS="--gpus all"
    NVIDIA_ENV="-e NVIDIA_VISIBLE_DEVICES=all"
    CONTAINER_NAME="${CONTAINER_NAME}-gpu"
else
    echo "⚠ No GPU detected. Running in CPU-only mode."
    CONTAINER_NAME="${CONTAINER_NAME}-cpu"
fi

echo ""
echo "Starting container: ${CONTAINER_NAME}"
echo ""

docker run --rm -it \
    ${GPU_FLAGS} \
    --name ${CONTAINER_NAME} \
    -v "$(pwd)":/workspace \
    -v /workspace/.venv \
    -v "${HOME}/.cache/huggingface:/home/appuser/.cache/huggingface" \
    -p 8888:8888 \
    ${NVIDIA_ENV} \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e WANDB_ENTITY="${WANDB_ENTITY:-}" \
    -e WANDB_PROJECT="${WANDB_PROJECT:-}" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    ${IMAGE_NAME} \
    "$@"
