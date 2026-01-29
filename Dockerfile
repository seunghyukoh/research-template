FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.9.10 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/workspace/.venv

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    wget \
    ca-certificates \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN groupadd --system --gid 1000 appuser \
 && useradd --system --gid 1000 --uid 1000 --create-home appuser

WORKDIR /workspace

RUN chown -R appuser:appuser /workspace

USER appuser

ENV PATH="/workspace/.venv/bin:$PATH"

COPY --chown=appuser:appuser uv.lock pyproject.toml ./
COPY --chown=appuser:appuser packages packages

RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-install-workspace --no-install-project

COPY --chown=appuser:appuser .env.example .env

EXPOSE 8888

CMD ["/bin/bash"]
