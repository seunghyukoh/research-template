---
name: experiment-pipeline
user-invokable: true
description: >
  Skill for orchestrating LLM research experiment pipelines within Claude Code sessions.
  Trigger when the user says "run experiment", "train model", "evaluate model",
  "experiment pipeline", "start training", "launch training", "run eval", "run evaluation",
  "prepare data", "data pipeline", "ablation study", "run ablation", or invokes
  /experiment-pipeline. Also trigger when the user describes a workflow involving
  branching, config creation, training, evaluation, or PR creation for experiments.
---

# Experiment Pipeline Skill

Orchestrate LLM research experiment workflows: branch creation, config generation, training, evaluation, and reporting. This skill follows the project's research workflow conventions and delegates long-running tasks to the user when necessary.

---

## Pre-flight Checks

Before starting any pipeline, verify the following:

| Check | Command | Action on Failure |
| ------ | ------- | ----------------- |
| Git clean | `git status` | Ask user to commit or stash changes |
| On correct branch | `git branch --show-current` | Warn if already on an experiment branch |
| Dependencies synced | `uv sync --check` | Run `uv sync` |
| `.env` exists | Check file | Warn about missing `WANDB_API_KEY`, `WANDB_PROJECT` |
| GPU available | `python -c "import torch; print(torch.cuda.is_available())"` | Warn, training may be slow |
| `outputs/` exists | Check directory | Create it (gitignored) |

---

## Pipeline Types

Ask the user which pipeline to run. Do NOT guess.

### 1. SFT Training Pipeline (Full)

**Stages**: Setup -> Train -> Monitor -> Evaluate -> Report

Use when the user wants to fine-tune a model end-to-end.

### 2. Evaluation-Only Pipeline

**Stages**: Setup -> Evaluate -> Report

Use when the user already has a trained model and wants to run benchmarks.

### 3. Data Preparation Pipeline

**Stages**: Setup -> Data Processing -> Report

Use when the user needs to prepare or transform datasets before training.

### 4. Ablation Pipeline

**Stages**: Setup -> Multi-Config Generation -> Sequential Runs -> Compare -> Report

Use when the user wants to compare multiple configurations (e.g., learning rate sweep, LoRA rank comparison).

---

## Stage 1: Setup

### Branch Creation

Create an experiment branch following the project's branch strategy:

```bash
git checkout -b exp/<experiment-name>
```

- Use lowercase and hyphens: `exp/lora-rank-ablation`, not `exp/LoRA_Rank_Ablation`
- Ask the user for a descriptive experiment name if not provided

### Config Generation

Use the existing `SFTConfig` dataclass from `packages/sft/sft/config.py`:

```python
from sft.config import SFTConfig, ModelConfig, DataConfig, TrainingConfig
```

**Available config fields:**

- `ModelConfig`: `model_name`, `use_lora`, `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules`
- `DataConfig`: `dataset_name`, `dataset_path`, `train_split`, `validation_split`, `max_seq_length`
- `TrainingConfig`: `output_dir`, `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `warmup_steps`, `logging_steps`, `save_strategy`, `report_to`

**CRITICAL**: Always ask the user for:
- Model name (e.g., `meta-llama/Llama-3.1-8B`)
- Dataset name or path
- Whether to use LoRA or full fine-tuning
- Output directory name (suggest `outputs/<experiment-name>`)

Do NOT assume default values for model or dataset. Other hyperparameters can use defaults unless the user specifies otherwise.

### Training Script Generation

Generate a training script in `scripts/` using the project conventions:

```python
"""<Experiment description>."""

from datasets import load_dataset
from sft.config import SFTConfig, ModelConfig, DataConfig, TrainingConfig
from sft.trainer import SFTTrainer
from src.utils.logger import setup_logger

logger = setup_logger("experiment_name")
```

- Use `setup_logger` from `src/utils/logger.py` instead of `print()`
- Use `pathlib.Path` for file operations
- Use expressive naming (e.g., `generate_dataset` not `gen_ds`)

---

## Stage 2: Training

### Launching Training

Use `uv run` with accelerate:

```bash
uv run accelerate launch scripts/<training_script>.py
```

### Long-Running Training

Claude Code sessions have timeout limits. For training that takes more than a few minutes:

1. **Inform the user** that training will take a long time
2. **Provide the exact command** for the user to run manually:
   ```bash
   nohup uv run accelerate launch scripts/<training_script>.py > logs/<experiment-name>.log 2>&1 &
   ```
3. **Create the logs directory** if needed: `mkdir -p logs`
4. **Do NOT attempt** to run long training jobs within the Claude Code session
5. **Offer to continue** with evaluation after training completes

---

## Stage 3: Monitoring

Claude Code cannot directly access W&B dashboards. Provide the user with monitoring guidance:

- **W&B Dashboard**: "Check your W&B project at <https://wandb.ai> for live training metrics"
- **Log tailing**: `tail -f logs/<experiment-name>.log`
- **GPU monitoring**: `nvidia-smi` or `watch -n 1 nvidia-smi`

Suggest the user return after training completes to continue with evaluation.

---

## Stage 4: Evaluation

### Using lm-evaluation-harness

Follow the pattern from `scripts/demo_eval.sh`:

```bash
uv run accelerate launch -m lm_eval run \
    --model hf \
    --model_args pretrained=<model_path> \
    --tasks <task_list> \
    --output_path outputs/<experiment-name>/eval_results
```

**Ask the user for:**
- Model path (local checkpoint or HuggingFace model ID)
- Evaluation tasks (e.g., `hellaswag`, `mmlu`, `gsm8k`, `arc_challenge`)
- Number of few-shot examples if applicable (`--num_fewshot`)

### LoRA Model Evaluation

If the model was trained with LoRA, inform the user about two options:

1. **Direct evaluation with adapter**: Use `--model_args pretrained=<base_model>,peft=<adapter_path>`
2. **Merge first, then evaluate**: Merge LoRA weights into the base model, then evaluate the merged model

### Custom Evaluation Scripts

If the user needs evaluation beyond lm-eval benchmarks, generate a custom evaluation script in `scripts/` following the same conventions (logger, pathlib, expressive naming).

---

## Stage 5: Reporting

### Commit Changes

Follow the project's commit message convention:

```text
<verb>: <summary>
```

Verbs: `add`, `update`, `fix`, `exp`, `data`, `eval`

Examples:
- `add: lora sft training script for llama-3.1`
- `exp: learning rate ablation on gsm8k`
- `eval: hellaswag benchmark for fine-tuned model`

### Create Pull Request

Use the project's PR template format:

```bash
gh pr create --title "<concise title>" --body "$(cat <<'EOF'
# Updates
- <item 1>
- <item 2>
- <item 3>
EOF
)"
```

Include in the PR body:
- What was trained/evaluated
- Key hyperparameters
- Results summary (metrics, comparison with baseline if applicable)
- Link to W&B run if available

---

## Error Handling

| Stage | Common Error | Solution |
| ------ | ------- | ------- |
| Setup | `uv sync` fails | Check `pyproject.toml` for dependency conflicts, run `uv sync --reinstall` |
| Setup | Branch already exists | Ask user: switch to existing branch or create new one? |
| Training | CUDA OOM | Reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps`, enable LoRA |
| Training | `wandb` not configured | Check `.env` for `WANDB_API_KEY`, or run `wandb login` |
| Training | Dataset not found | Verify `dataset_name` on HuggingFace Hub or `dataset_path` exists locally |
| Evaluation | Model path not found | Check `outputs/` directory for checkpoint, verify path |
| Evaluation | Task not available | Run `lm_eval --tasks list` to see available tasks |
| Report | Pre-commit hook fails | Run `uv run ruff check --fix` and `uv run ruff format`, then re-commit |

---

## Ablation-Specific Workflow

For ablation studies, generate multiple configs varying one or more hyperparameters:

1. **Ask the user** which parameters to vary and their ranges
2. **Generate a config list** (e.g., learning rates: `[1e-5, 2e-5, 5e-5]`)
3. **Create a runner script** that iterates over configs
4. **Each run** should have a distinct `output_dir` and W&B run name
5. **After all runs**, generate a comparison table or suggest W&B report

Example ablation naming:
- Branch: `exp/lr-ablation-llama3`
- Output dirs: `outputs/lr-ablation-llama3/lr-1e5/`, `outputs/lr-ablation-llama3/lr-2e5/`, etc.

---

## Project Rules Reference

These rules come from the project's `.claude/rules/` and must always be followed:

- **Commits**: `<verb>: <summary>` format (see `commit-messages.md`)
- **Branches**: `exp/<name>` for experiments, lowercase with hyphens (see `branch-strategy.md`)
- **Package manager**: Always use `uv run`, `uv add` (see `coding-conventions.md`)
- **Logging**: Use `setup_logger()` from `src/utils/logger.py`, never `print()` (see `coding-conventions.md`)
- **Config**: Use existing `SFTConfig` dataclass, do NOT create new config systems (see `packages/sft/sft/config.py`)
- **Safety**: Never commit `.env`, model checkpoints, or files > 10MB (see `safety.md`)
- **Gitignored dirs**: `data/`, `outputs/`, `wandb/`, `logs/` stay gitignored (see `safety.md`)
