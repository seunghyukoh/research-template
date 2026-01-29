# AI Research Template

Batteries-included template for ML/AI research projects with integrated model evaluation capabilities.

## Features

-   Quick start with minimal setup (just configure `.env`)
-   UV workspace-based monorepo structure for scalable project organization
-   Integrated [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for comprehensive model evaluation
-   Docker support with automatic CPU/GPU detection
-   W&B (Weights & Biases) integration for experiment tracking
-   HuggingFace Hub integration
-   Pre-configured development environment with Python 3.12

## Quick Start

### 1. Environment Setup

#### Option A: Docker Environment (Recommended)

```bash
# Create and configure .env file
cp .env.example .env
# Edit .env to add your WANDB_API_KEY, WANDB_ENTITY, HUB_ID, etc.

# Method 1: Auto-detect GPU (recommended)
bash run.sh
# The script automatically detects GPU and runs in CPU or GPU mode

# Method 2: Docker Compose with explicit profile
# CPU-only environment
docker compose --profile cpu up -d
docker compose exec research-dev-cpu bash

# GPU environment (requires NVIDIA GPU + Docker runtime)
docker compose --profile gpu up -d
docker compose exec research-dev-gpu bash

# Method 3: Manual docker run (CPU-only)
docker build -t research-dev .
docker run -it \
  -v $(pwd):/workspace \
  -v /workspace/.venv \
  -p 8888:8888 \
  research-dev
```

#### Option B: Local Environment with UV

```bash
# Create and configure .env file
cp .env.example .env
# Edit .env to add your WANDB_API_KEY, WANDB_ENTITY, HF_TOKEN, etc.

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### 2. Running a Demo Evaluation

```bash
# Run a quick evaluation demo (HellaSwag with GPT-2)
bash scripts/demo_eval.sh
```

### 3. VSCode Extensions (Recommended)

-   Better Commits
-   EditorConfig
-   Error Lens
-   Pre-commit Helper
-   Python Environment Manager
-   Black Formatter

## Directory Structure

```text
.
├── packages/             # UV workspace packages
│   ├── lm-eval/         # Integrated lm-evaluation-harness (submodule)
│   └── sft/             # SFT training package (for future implementation)
├── scripts/             # Execution scripts
│   └── demo_eval.sh    # Model evaluation demo
├── src/                 # Your research code
├── notebooks/           # Jupyter notebooks for experiments
├── data/                # Datasets
├── outputs/             # Experiment outputs and checkpoints
├── docker-compose.yml   # Docker Compose configuration
├── Dockerfile           # Docker image definition
├── run.sh              # Docker startup script with GPU auto-detection
└── pyproject.toml      # Project configuration and dependencies
```

## Running Evaluations

### Model Evaluation with lm-eval

The template integrates [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for comprehensive model evaluation across various benchmarks.

```bash
# Run the demo evaluation (GPT-2 on HellaSwag)
bash scripts/demo_eval.sh

# Custom evaluation with different model and tasks
uv run accelerate launch -m lm_eval run \
    --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks mmlu,truthfulqa,gsm8k \
    --output_path outputs/eval_results

# Evaluate with specific settings
uv run lm_eval run \
    --model hf \
    --model_args pretrained=your-model,dtype=bfloat16 \
    --tasks hellaswag \
    --batch_size 8 \
    --log_samples \
    --output_path outputs/
```

For available tasks and more options, see the [lm-eval documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs).

## Workspace Structure

This template uses UV workspace for managing multiple packages:

-   **Root package**: Main project with shared dependencies
-   **`packages/lm-eval`**: Evaluation harness (editable install from local path)
-   **`packages/sft`**: Placeholder for SFT training code (workspace member)

To add your own training code:

```bash
# Add your training scripts to packages/sft/
# Update packages/sft/pyproject.toml with required dependencies
# Dependencies will be shared across the workspace where possible
```

## Environment Variables

Configure these in your `.env` file:

```bash
PROJECT_NAME=research-template
PROJECT_NICKNAME=rtemp

# Weights & Biases configuration
WANDB_API_KEY=your_api_key
WANDB_ENTITY=your_wandb_entity
WANDB_PROJECT=${PROJECT_NAME}

# HuggingFace Hub token (for private models/datasets)
HF_TOKEN=your_hf_token
```

## Docker Development

### CPU vs GPU Mode

The Docker setup automatically supports both CPU and GPU environments:

-   **`run.sh`**: Automatically detects GPU availability and runs in the appropriate mode
-   **Docker Compose**: Use profiles to explicitly choose CPU or GPU mode
    -   `--profile cpu`: CPU-only mode (works everywhere)
    -   `--profile gpu`: GPU mode (requires NVIDIA GPU + Docker runtime)

### Container Structure

When you run the container, your project files are mounted at `/workspace`:

```text
/workspace/                   # Your project root (mounted)
├── .venv/                   # Virtual environment (container-only)
├── packages/                # UV workspace packages
│   ├── lm-eval/            # Evaluation harness
│   └── sft/                # SFT package
├── src/                     # Source code
├── scripts/                 # Execution scripts
├── notebooks/              # Jupyter notebooks
├── data/                   # Datasets
└── outputs/                # Experiment outputs
```

### Running Jupyter in Container

```bash
# Start container with docker-compose (CPU)
docker compose --profile cpu up -d
docker compose exec research-dev-cpu bash

# Or with GPU
docker compose --profile gpu up -d
docker compose exec research-dev-gpu bash

# Or use run.sh (auto-detects GPU)
bash run.sh

# Inside container, run Jupyter
uv run jupyter lab --ip=0.0.0.0 --allow-root --no-browser

# Access at http://localhost:8888
```

### Tips

-   **Auto GPU Detection**: `run.sh` checks for GPU availability and runs in the appropriate mode
-   **CPU Fallback**: Docker Compose profiles allow explicit CPU/GPU selection
-   **No GPU Required**: The container works fine on CPU-only systems
-   The `.venv` directory is isolated in the container to avoid platform conflicts
-   Your code changes are immediately reflected in the container
-   HuggingFace cache is mounted from your host to save bandwidth
-   Use `docker compose watch` for auto-sync during development

### Troubleshooting

**Error: "could not select device driver with capabilities: [[gpu]]"**

This means GPU is not available. Use CPU mode:
```bash
# Use run.sh (auto-detects)
bash run.sh

# Or explicitly use CPU profile
docker compose --profile cpu up
```

## Links

-   [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Comprehensive evaluation framework
-   [UV Documentation](https://docs.astral.sh/uv/) - Fast Python package installer and resolver
-   [Weights & Biases Quickstart](https://docs.wandb.ai/quickstart)
-   [HuggingFace Hub Documentation](https://huggingface.co/docs/hub)
-   [Accelerate Documentation](https://huggingface.co/docs/accelerate)
