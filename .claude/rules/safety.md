# Safety Rules

## Secrets and Credentials

- NEVER commit `.env` files, API keys, or tokens
- Use `.env.example` for documenting required environment variables (without real values)
- If a secret is accidentally staged, unstage it before committing

## Large Files

- Do NOT commit model checkpoints, weights, or large binary files
- Do NOT commit datasets (use `data/` which is gitignored)
- Upload large files to HuggingFace Hub or cloud storage instead
- If a file exceeds 10MB, it probably should not be in git

## Gitignored Directories

These directories are gitignored and should stay that way:

- `data/` - Datasets
- `outputs/` - Experiment outputs and checkpoints
- `wandb/` - W&B local logs
- `logs/` - Log files
- `.env` - Environment variables with secrets

## Code Safety

- Validate file paths before reading/writing to prevent path traversal
- Do not hardcode server addresses, ports, or credentials in source code
- Use environment variables for all external service configuration
