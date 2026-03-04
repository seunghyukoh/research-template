![AI Research Template](.github/assets/hero.jpg)

Batteries-included template for ML/AI research. Clone it, let your agent configure it, start experimenting.

## Installation

### For Humans

1. Create a new repo from this template (or fork it)
2. Clone it:

```bash
git clone --recurse-submodules <your-repo-url>
cd <your-repo>
```

3. Hand it off to your AI tool. Paste this into Claude Code, Cursor, Windsurf, or any agent:

```text
Read SETUP.md and follow it to configure this project.
```

### For LLM Agents

Read `SETUP.md` in the project root. It has the full setup guide — interview questions, file edits, and verification steps.

---

## Features

-   UV workspace monorepo with Python 3.12
-   [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) as a local package
-   SFT training package with LoRA support (`packages/sft`)
-   Docker with automatic CPU/GPU detection
-   W&B experiment tracking, HuggingFace Hub integration
-   Pre-commit hooks (ruff, markdownlint, secret detection)
-   Claude Code rules, hooks, skills, and project conventions baked in

## Manual Setup

If you don't want an agent doing it, here's the old-fashioned way.

### 1. Clone

```bash
git clone --recurse-submodules <repository-url>

# If already cloned without submodules
git submodule update --init --recursive
```

### 2. Environment Setup

#### Option A: Docker (Recommended)

```bash
cp .env.example .env
# Fill in WANDB_API_KEY, HF_TOKEN, etc.

bash run.sh          # Auto-detects GPU
# Or pick explicitly:
# docker compose --profile cpu up -d
# docker compose --profile gpu up -d

# First time only — install workspace packages inside the container
uv sync
```

#### Option B: Local with UV

```bash
cp .env.example .env

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
```

### 3. Run a Demo Evaluation

```bash
bash scripts/demo_eval.sh    # GPT-2 on HellaSwag
```

### 4. Claude Code Setup

```bash
# Copy the example and fill in your project details
cp CLAUDE.md.example CLAUDE.md
```

The template ships with pre-configured Claude Code settings:

-   `.claude/rules/` — Commit conventions, branch strategy, coding style, safety rules
-   `.claude/hooks/` — Guards against accidental test modification and `print()` in source code
-   `.claude/skills/` — `/experiment-pipeline`, anti-AI writing
-   `.claude/settings.json` — Plugins and permissions
-   `.mcp.json` — MCP servers (sequential thinking)

Edit `CLAUDE.md` to describe your specific research project. The rules and hooks work as-is.

## Directory Structure

```text
.
├── .claude/              # Claude Code configuration
│   ├── rules/            # Project rules (commit, branch, coding, safety)
│   ├── hooks/            # Pre-tool-use guard scripts
│   ├── skills/           # Custom skills (anti-ai-writing, experiment-pipeline)
│   └── settings.json     # Permissions, hooks, plugins
├── packages/             # UV workspace packages
│   ├── lm-eval/          # lm-evaluation-harness (submodule)
│   └── sft/              # SFT training package
├── scripts/              # Shell scripts
│   └── demo_eval.sh
├── src/                  # Your research code
│   └── utils/logger.py   # Logging utility
├── tests/                # Test suite
├── docker-compose.yml
├── Dockerfile
├── run.sh                # GPU auto-detection launcher
├── pyproject.toml
├── SETUP.md              # Agent-friendly setup guide
└── CLAUDE.md.example     # Template for Claude Code project context
```

## Running Evaluations

```bash
# Demo
bash scripts/demo_eval.sh

# Custom evaluation
uv run accelerate launch -m lm_eval run \
    --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks mmlu,truthfulqa,gsm8k \
    --output_path outputs/eval_results

# With specific settings
uv run lm_eval run \
    --model hf \
    --model_args pretrained=your-model,dtype=bfloat16 \
    --tasks hellaswag \
    --batch_size 8 \
    --log_samples \
    --output_path outputs/
```

See the [lm-eval docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for available tasks.

## Workspace Packages

The project uses UV workspace to manage multiple packages from a single root:

-   **Root** (`./`) — Shared dependencies, research code in `src/`
-   **`packages/lm-eval`** — Evaluation harness (editable install from submodule)
-   **`packages/sft`** — SFT training with LoRA/full fine-tuning

To add a new package:

```bash
mkdir -p packages/<name>/<name>
# Create packages/<name>/pyproject.toml
# Add to [tool.uv.workspace] members in root pyproject.toml
uv sync
```

## Environment Variables

Configure in `.env` (copy from `.env.example`):

```bash
PROJECT_NAME=research-template
PROJECT_NICKNAME=rtemp

WANDB_API_KEY=your_api_key
WANDB_ENTITY=your_wandb_entity
WANDB_PROJECT=${PROJECT_NAME}

HF_TOKEN=your_hf_token
```

## Docker

### CPU vs GPU

`run.sh` detects GPU availability and picks the right mode. You can also choose manually:

```bash
docker compose --profile cpu up -d    # CPU only
docker compose --profile gpu up -d    # Needs NVIDIA GPU + runtime
```

### First Time Setup

After starting the container, run `uv sync` once to install workspace packages:

```bash
docker compose exec research-dev-cpu bash
uv sync
python -c "from sft import SFTTrainer; print('Setup complete')"
```

The Docker image pre-installs heavy dependencies (torch, transformers) so startup is fast. `uv sync` wires up your local workspace packages on top.

Re-run `uv sync` when you add dependencies to `pyproject.toml`, pull changes to `uv.lock`, or recreate the container.

### Jupyter

```bash
# Inside container
uv run jupyter lab --ip=0.0.0.0 --allow-root --no-browser
# Access at http://localhost:8888
```

### Tips

-   Code changes on the host are reflected immediately in the container (mounted volume)
-   HuggingFace cache is mounted from the host to avoid re-downloading models
-   `.venv` lives only inside the container to prevent platform conflicts
-   `docker compose watch` enables automatic file sync during development

### Troubleshooting

**"could not select device driver with capabilities: [[gpu]]"** — GPU not available. Use `bash run.sh` (auto-fallback) or `docker compose --profile cpu up`.

## VSCode Extensions (Recommended)

-   EditorConfig
-   Error Lens
-   Pre-commit Helper
-   Python Environment Manager
-   Ruff

## Links

-   [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
-   [UV](https://docs.astral.sh/uv/)
-   [Weights & Biases](https://docs.wandb.ai/quickstart)
-   [HuggingFace Hub](https://huggingface.co/docs/hub)
-   [Accelerate](https://huggingface.co/docs/accelerate)
-   [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
