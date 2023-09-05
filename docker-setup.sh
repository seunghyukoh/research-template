docker build -t seunghyukoh/research-template:latest \
    --build-arg UID=$(id -u) \
    --build-arg USER_NAME=$(id -un) \
    --build-arg UBUNTU_VERSION=20.04 \
    --build-arg CUDA_VERSION=11.7.1 \
    --build-arg CUDA=11.7 \
    --build-arg PYTHON_VERSION=3.10 \
    --build-arg CONDA_ENV_NAME=research-template \
    -f Dockerfile \
    .
