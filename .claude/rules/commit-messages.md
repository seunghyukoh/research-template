# Commit Message Convention

This is a research repository. Use a simple and consistent commit message pattern.

## Format

```text
<verb>: <summary>
```

## Verbs

| Verb | Usage | Example |
| ------ | ------- | --------- |
| **add** | Add new features, scripts, experiments | `add: trajectory mix training script` |
| **update** | Modify existing code/config/docs | `update: dataset format for compatibility` |
| **fix** | Fix bugs | `fix: memory leak in data loader` |
| **exp** | Add/modify experiments | `exp: ablation study on learning rate` |
| **data** | Dataset preparation/changes | `data: prepare GSM8K with CoT format` |
| **eval** | Evaluation code/metrics | `eval: add accuracy metric for math tasks` |

## Rules

- Write commit messages in English
- Keep the summary concise (under 50 characters if possible)
- Use lowercase for the verb
- Do not include Claude Code signature in commit messages
