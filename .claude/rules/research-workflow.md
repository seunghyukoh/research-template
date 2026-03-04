# Research Workflow

## Experiment Lifecycle

```text
1. Branch    → git checkout -b exp/<experiment-name>
2. Configure → Create/modify config in configs/ or as CLI args
3. Train     → uv run accelerate launch <script> --config <config>
4. Log       → Results auto-logged to W&B (wandb)
5. Evaluate  → Use lm-eval or custom eval scripts
6. Record    → Save outputs to outputs/ directory
7. PR        → Create PR with results summary
```

## Experiment Tracking

- Use Weights & Biases (`wandb`) for all experiment logging
- Set `WANDB_PROJECT` in `.env` to group runs by project
- Log hyperparameters, metrics, and artifacts to W&B

## Evaluation

- Use `lm-evaluation-harness` (`packages/lm-eval`) for standardized benchmarks
- Evaluation scripts go in `scripts/`
- Save evaluation results to `outputs/`

## Workspace Packages

| Package     | Path                | Purpose                                  |
|-------------|---------------------|------------------------------------------|
| **sft**     | `packages/sft/`     | SFT training with LoRA/full fine-tuning  |
| **lm-eval** | `packages/lm-eval/` | Model evaluation harness (submodule)     |

To add a new workspace package:

```bash
mkdir -p packages/<name>/<name>
# Create packages/<name>/pyproject.toml
# Add to root pyproject.toml [tool.uv.workspace] members
# Add to root pyproject.toml [project] dependencies
uv sync
```

## Data Management

- Raw datasets go in `data/` (gitignored)
- Use HuggingFace `datasets` library for loading/processing
- Document dataset format and preprocessing steps in code or configs
