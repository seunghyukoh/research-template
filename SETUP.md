# Setup Guide

This guide configures a fresh clone of the AI Research Template for your project. Follow every step in order.

If you are a Claude Code user, you can run `/setup` instead of following this guide.

---

## Step 1: Gather Information

Ask the user the following questions. Do not guess. Do not skip.

| # | Question | Example | Required |
| --- | -------- | ------- | -------- |
| 1 | Project name (lowercase, hyphens) | `math-reasoning` | Yes |
| 2 | Short nickname (2-5 chars) | `mr` | No (default: first letters of each word) |
| 3 | Research purpose (1-2 sentences) | "Fine-tuning Llama-3 on math reasoning with CoT" | Yes |
| 4 | Research domain | NLP / CV / RL / Multimodal / Other | No (default: NLP) |
| 5 | W&B entity (team or username) | `my-team` | No (skip if unknown) |
| 6 | Need HuggingFace token? | Yes / No | No (default: No) |

Show a summary and confirm with the user before proceeding.

---

## Step 2: Create `CLAUDE.md`

Copy `CLAUDE.md.example` to `CLAUDE.md`. Then make these edits:

### Project Overview section

Replace the TODO comment block (lines 6-10) with the research purpose from Question 3:

```markdown
## Project Overview

<research purpose from Q3>
```

### Key Files section

Replace the TODO comment block (lines 53-59) with:

```markdown
## Key Files

- `src/`: Research code
- `configs/`: Configuration files
- `scripts/`: Training and evaluation scripts
- `packages/sft/`: SFT training package
```

### Workspace Structure section

Replace the TODO comment (line 51) with:

```markdown
<!-- Add your own packages as needed. See research-workflow.md for instructions. -->
```

### Research Notes section

Replace the TODO comment block (lines 70-74) based on the domain (Question 4):

- **NLP**: `This is an NLP research project. We use lm-evaluation-harness for benchmarks.`
- **CV**: `This is a computer vision research project. Evaluation uses domain-specific metrics.`
- **RL**: `This is a reinforcement learning research project. Environment setup and seed management are critical.`
- **Multimodal**: `This is a multimodal research project. Data loading handles multiple modalities.`
- **Other**: `Research domain: <user's input>`

Keep all other sections (Common Commands, Environment Variables) unchanged.

---

## Step 3: Create `.env`

Copy `.env.example` to `.env`. Then edit the following values:

```bash
PROJECT_NAME=<project-name from Q1>
PROJECT_NICKNAME=<nickname from Q2>

# Weights & Biases configuration
WANDB_API_KEY=your_api_key
WANDB_ENTITY=<entity from Q5, or keep "your_wandb_entity" if skipped>
WANDB_PROJECT=${PROJECT_NAME}

# HuggingFace Hub token (for private models/datasets)
HF_TOKEN=your_hf_token
```

**IMPORTANT**: Keep `your_api_key` and `your_hf_token` as placeholders. Never write real API keys or tokens.

---

## Step 4: Update `pyproject.toml`

Edit only these two fields:

- `name`: `"research-template"` -> `"<project-name from Q1>"`
- `description`: `"Add your description here"` -> `"<research purpose from Q3>"`

Do NOT modify dependencies, workspace config, or any other fields.

---

## Step 5: Update `README.md`

- Replace the title line (`# AI Research Template`) with `# <Title-Cased Project Name>`
- Replace the first paragraph (description) with the research purpose from Question 3
- Keep all other sections unchanged

---

## Step 6: Initialize Submodules

```bash
git submodule update --init --recursive
```

If this fails (network issues, missing URLs), warn the user and continue. Do not abort.

---

## Step 7: Install Dependencies

```bash
uv sync
```

If this fails (Python version mismatch, platform issues), warn the user and continue. Suggest running `uv sync` manually later.

---

## Step 8: Show Summary

Show the user what was configured:

```text
Setup complete!

Configured files:
  - CLAUDE.md        - Project context for Claude Code
  - .env             - Environment variables (API keys still need filling)
  - pyproject.toml   - Project name and description
  - README.md        - Project title and description

Next steps:
  1. Fill in API keys in .env:
     - WANDB_API_KEY: Get from https://wandb.ai/authorize
     - HF_TOKEN: Get from https://huggingface.co/settings/tokens (if needed)
  2. Review CLAUDE.md and add project-specific details
  3. Start your first experiment
```

---

## Rules

- **Never** ask for or write real API keys, tokens, or passwords
- If `CLAUDE.md` or `.env` already exists, ask the user before overwriting
- If `pyproject.toml` name is not `research-template`, warn that setup may have already run
- Project name validation: lowercase, hyphens only, no spaces or special characters
