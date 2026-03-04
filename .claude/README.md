# .claude/ Directory Guide

This directory contains project-level settings for Claude Code. Modify, add, or remove files here to match your research needs.

## Directory Structure

```text
.claude/
├── settings.json          # Permissions, hook registration, plugin config
├── hooks/                 # Validation scripts that run before/after tool calls
│   ├── check-print-usage.sh
│   └── check-test-modification.sh
├── rules/                 # Rules auto-injected into every conversation (.md files)
│   ├── branch-strategy.md
│   ├── coding-conventions.md
│   ├── commit-messages.md
│   ├── research-workflow.md
│   └── safety.md
└── skills/                # Extensions triggered by keywords or /commands
    ├── anti-ai-writing/
    │   └── SKILL.md
    └── experiment-pipeline/
        └── SKILL.md
```

---

## Components

### `settings.json` — Project Settings

Controls Claude Code session behavior.

```jsonc
{
  "permissions": {
    "allow": [...]       // Tools to auto-approve without prompting
  },
  "hooks": {
    "PreToolUse": [...]  // Scripts to run before Edit/Write tool calls
  },
  "enabledPlugins": {
    // Activated plugins
  }
}
```

**Customization points:**
- Add frequently used MCP tools to `permissions.allow` to skip approval prompts
- Remove unnecessary validations or add new hooks in `hooks`
- Toggle plugins on/off in `enabledPlugins`

### `hooks/` — Automated Validation Scripts

Shell scripts that run automatically when Edit/Write tools are called. Must be registered in `settings.json` under `hooks.PreToolUse` to take effect.

| File | Purpose |
| ------ | ------- |
| `check-print-usage.sh` | Warns against `print()` in `src/` and `packages/` `.py` files, suggests logger |
| `check-test-modification.sh` | Warns when modifying test files without explicit user approval |

**Customization examples:**
- Add a hook to warn against `cv2.imshow()` in CV projects (prefer saving to file)
- Add a hook to warn when modifying files in `configs/`
- Remove the `print()` hook if not needed by deleting its entry from `settings.json`

**Adding a new hook:**

1. Create a shell script in `.claude/hooks/` (access tool input via `$TOOL_INPUT`)
2. Register it in `settings.json` under `hooks.PreToolUse`

```jsonc
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/your-hook.sh",
            "statusMessage": "Running your check"
          }
        ]
      }
    ]
  }
}
```

### `rules/` — Conversation Rules

All `.md` files in this directory are automatically included in every Claude Code session context. Claude follows these rules when writing code or running commands.

| File | Purpose | When to modify |
| ------ | ------- | -------------- |
| `branch-strategy.md` | Branch naming conventions | Add/remove branch types (e.g., add `bench/`) |
| `coding-conventions.md` | Code style rules | When changing languages/frameworks (e.g., add JAX rules) |
| `commit-messages.md` | Commit message format | Add/change commit verbs (e.g., add `bench:`) |
| `research-workflow.md` | Experiment lifecycle | Add workspace packages, change eval tools |
| `safety.md` | Security and large file rules | Change gitignore targets, adjust security policies |

**Customization by research domain:**

- **NLP**: Use the default template mostly as-is. Add additional benchmarks beyond `lm-eval` in `research-workflow.md`
- **CV**: Add image processing conventions to `coding-conventions.md` (e.g., `torchvision` transform patterns). Replace `lm-eval` with your evaluation method in `research-workflow.md`
- **RL**: Add environment setup, seed management, and episode logging rules to `research-workflow.md`
- **Multimodal**: Add media file management rules to `safety.md`, add data loading patterns to `coding-conventions.md`

**Adding a new rule:** Create a `.md` file in `.claude/rules/` — it is auto-recognized.

### `skills/` — Extensions

Skills respond to specific keywords or `/command` triggers. Each skill consists of a single `SKILL.md` file.

| Skill | Trigger | Purpose |
| ------ | ------- | ------- |
| `anti-ai-writing` | "write naturally", "sound human", etc. | Write/review natural English text |
| `experiment-pipeline` | `/experiment-pipeline`, "run experiment", etc. | Orchestrate experiment workflows |

**Customization points:**

The `experiment-pipeline` skill is currently designed around SFT (Supervised Fine-Tuning). Adapt it to your research:

- **Pipeline types**: Add/remove pipelines (e.g., RLHF Pipeline, Pretraining Pipeline)
- **Config structure**: Modify the dataclass in `packages/sft/sft/config.py` and update the skill's Config Generation section accordingly
- **Evaluation**: If using tools other than `lm-eval`, modify Stage 4
- **Training command**: If using `torchrun`, `deepspeed`, etc. instead of `accelerate launch`, modify Stage 2

**Adding a new skill:**

1. Create `.claude/skills/<skill-name>/SKILL.md`
2. Write YAML front matter with `name` and `description` (include trigger conditions in description)
3. Write instructions for Claude in the markdown body

```yaml
---
name: my-skill
user-invokable: true  # Enables /my-skill slash command
description: >
  When to trigger this skill...
---

# My Skill

Instructions for Claude...
```

---

## Customization Checklist

Review these items when using this template for a new project.

### Must modify

- [ ] `rules/research-workflow.md` — Workspace packages, eval tools, experiment lifecycle
- [ ] `rules/coding-conventions.md` — Libraries and code patterns for your domain

### Modify as needed

- [ ] `rules/commit-messages.md` — Add commit verbs (e.g., `bench:`, `pretrain:`)
- [ ] `rules/branch-strategy.md` — Add/remove branch types
- [ ] `rules/safety.md` — Gitignore targets, file size limits
- [ ] `skills/experiment-pipeline/SKILL.md` — Pipeline types, train/eval commands, config structure
- [ ] `hooks/` — Remove unnecessary hooks, add domain-specific validation hooks
- [ ] `settings.json` — Hook registration, plugin settings

### Usually keep as-is

- `hooks/check-test-modification.sh` — Test file protection is universally useful
- `skills/anti-ai-writing/` — Useful when writing papers or blog posts
