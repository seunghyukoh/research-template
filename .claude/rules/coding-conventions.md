# Coding Conventions

## Python Style

- Use Python 3.12+ features (type hints, `match` statement, etc.)
- Use `ruff` for linting and formatting (configured in pre-commit)
- Prefer expressive naming: `generate_dataset` not `gen_ds`, `calculate_loss` not `calc_l`
- Use `pathlib.Path` instead of `os.path` for file operations

## Imports

- Group imports: stdlib, third-party, local (ruff handles this automatically)
- Use absolute imports for cross-package references
- Use relative imports within the same package

## Logging

- Use `src.utils.logger.setup_logger` instead of `print()` for any output
- Exception: Jupyter notebooks may use `print()` for quick inspection
- Use appropriate log levels: `DEBUG` for verbose, `INFO` for progress, `WARNING` for recoverable issues, `ERROR` for failures

## Configuration

- Use `hydra` / `omegaconf` for experiment configuration
- Use `dataclass` for typed configuration objects
- Keep hardcoded values out of training/evaluation scripts

## Documentation

- Write ALL documents in English (README, comments, docstrings, markdown files, commit messages)
- No exceptions: even inline comments and TODO notes must be in English

## Package Manager

- This project uses `uv` (not pip, poetry, or conda)
- Always use `uv run` to execute scripts, `uv add` to add dependencies
- Lock file (`uv.lock`) must be committed
