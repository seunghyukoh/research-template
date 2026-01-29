# SFT Training Package

Supervised Fine-Tuning utilities for language models with LoRA support.

## Features

- 🚀 Simple wrapper around TRL's SFTTrainer
- 🔧 Built-in LoRA configuration for parameter-efficient fine-tuning
- 📊 W&B integration for experiment tracking
- ⚙️ Dataclass-based configuration management

## Quick Start

```python
from sft import SFTTrainer
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("your-dataset")

# Initialize trainer
trainer = SFTTrainer(
    model_name="meta-llama/Llama-2-7b-hf",
    dataset=dataset["train"],
    output_dir="./outputs/my-model",
    use_lora=True,
)

# Start training
trainer.train()
```

## Configuration

Use `SFTConfig` for structured configuration:

```python
from sft.config import SFTConfig, ModelConfig, TrainingConfig

config = SFTConfig(
    model=ModelConfig(
        model_name="gpt2",
        use_lora=True,
        lora_r=16,
    ),
    training=TrainingConfig(
        num_train_epochs=3,
        learning_rate=2e-5,
    ),
)
```

## Development

This package is part of the research-template workspace and automatically inherits common dependencies (torch, transformers, etc.).

Package-specific dependencies:
- `peft`: For LoRA fine-tuning
- `trl`: For supervised fine-tuning utilities
