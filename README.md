# AI Research Template

Batteries-included template for ML/AI research projects with a focus on supervised fine-tuning (SFT) workflows.

## Features

-   Quick start with minimal setup (just configure `.env`)
-   Clear directory structure for easy scaling
-   W&B (Weights & Biases) and HuggingFace Hub integration
-   Code quality management with Black, isort, and Pre-commit hooks
-   Built-in SFT training pipeline with TRL
-   Remote debugging and development utilities

## Quick Start

### 1. Environment Setup

```bash
# Create and configure .env file
cp .env.example .env
# Edit .env to add your WANDB_API_KEY, WANDB_ENTITY, HUB_ID, etc.

# Setup Python environment
bash misc/setup-pyenv.sh
```

### 2. VSCode Extensions (Recommended)

-   Better Commits
-   EditorConfig
-   Error Lens
-   Pre-commit Helper
-   Python Environment Manager
-   Black Formatter

## Directory Structure

```text
.
├── configs/          # Experiment configurations (YAML)
│   └── sft/         # SFT-specific configs
├── misc/            # Setup and utility scripts
│   ├── setup-pyenv.sh
│   ├── remote-debug.sh
│   ├── wait-for-pid.sh
│   └── debug.sh
├── outputs/         # Experiment outputs and checkpoints
├── scripts/         # Training execution scripts
│   └── sft.sh      # SFT training launcher
├── utils/           # Python utility modules
│   ├── args.py     # Argument configurations
│   ├── batch_size.py
│   └── parse_args.py
└── sft.py           # Main SFT training script
```

## Running Experiments

### Supervised Fine-Tuning (SFT)

```bash
# Run with default settings (1 process, 100 steps)
bash scripts/sft.sh

# Run with custom settings
bash scripts/sft.sh <num_processes> <max_steps>
# Example: bash scripts/sft.sh 4 1000
```

The SFT script uses Accelerate for distributed training and automatically:

-   Logs to Weights & Biases
-   Pushes checkpoints to HuggingFace Hub
-   Handles mixed precision training (bf16)

## Utility Scripts

### setup-pyenv.sh

Set up the Python environment with all required dependencies

```bash
bash misc/setup-pyenv.sh
```

### wait-for-pid.sh

Wait for a process to finish before running the next command

```bash
bash misc/wait-for-pid.sh <PID> && bash scripts/next-script.sh
```

### remote-debug.sh

Enable VS Code debugging on remote servers

```bash
source misc/remote-debug.sh
debug your-script.py --args arg1
```

Then start debugging in VS Code with "Python Debugger: Remote Attach" configuration (port 5678 forwarding required).

### debug.sh

Sync files to a remote server and optionally execute scripts remotely

```bash
# Sync files only
source misc/debug.sh
dsync

# Sync and run a script remotely
drun "python sft.py --args"

# Configure connection settings
export HOST=your-server.com
export USER=your-username
export PORT=22
export DST=~/remote-debug-dir
```

## Environment Variables

Configure these in your `.env` file:

```bash
PROJECT_NAME=research-template
PROJECT_NICKNAME=rtemp

WANDB_ENTITY=your_wandb_entity
WANDB_PROJECT=${PROJECT_NAME}

# s3://$BUCKET_NAME/$BUCKET_PREFIX
BUCKET_NAME="your_bucket_name"
BUCKET_PREFIX="your_bucket_prefix"
```

## Links

-   [Weights & Biases Quickstart](https://docs.wandb.ai/quickstart)
-   [HuggingFace Hub Documentation](https://huggingface.co/docs/hub)
-   [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl)
