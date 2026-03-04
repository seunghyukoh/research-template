# Branch Strategy

This is a research repository. Use GitHub Flow combined with experiment branches.

## Branch Types

| Branch | Pattern | Usage | Example |
| ------ | ------- | ----- | ------- |
| **main** | `main` | Stable, reproducible code | - |
| **experiment** | `exp/<name>` | Research experiments, ablations | `exp/focal-loss-ablation` |
| **feature** | `feature/<name>` | New features, scripts | `feature/tulu3-dataloader` |
| **fix** | `fix/<name>` | Bug fixes | `fix/memory-leak-trainer` |
| **data** | `data/<name>` | Dataset preparation | `data/gsm8k-cot-format` |
| **eval** | `eval/<name>` | Evaluation changes | `eval/math-accuracy-metric` |

## Rules

- Always branch from `main`
- Use lowercase and hyphens for branch names (e.g., `exp/lr-ablation`, not `exp/LR_Ablation`)
- Keep branch names concise but descriptive
- Delete branches after merging to `main`
- Experiment branches (`exp/*`) can be kept for reproducibility if needed

## Workflow

```text
1. Create branch    → git checkout -b exp/my-experiment
2. Make changes     → git add . && git commit
3. Push branch      → git push -u origin exp/my-experiment
4. Create PR        → gh pr create
5. Review & Merge   → Merge to main after review
6. Delete branch    → git branch -d exp/my-experiment
```

## Main Branch Protection

- `main` should always be in a working state
- All changes to `main` should go through pull requests
- Experimental code should be validated before merging
