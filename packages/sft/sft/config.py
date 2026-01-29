"""Configuration management for SFT training."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str = "gpt2"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class DataConfig:
    """Dataset configuration."""

    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    train_split: str = "train"
    validation_split: Optional[str] = "validation"
    max_seq_length: int = 512


@dataclass
class TrainingConfig:
    """Training configuration."""

    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    logging_steps: int = 10
    save_strategy: str = "epoch"
    report_to: str = "wandb"


@dataclass
class SFTConfig:
    """Complete SFT training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
