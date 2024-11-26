from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments as TA


@dataclass
class DataArguments:
    data_name: Optional[str] = field(
        metadata={"help": "The name of the dataset"},
    )

    max_train_samples: Optional[int] = field(
        default=-1,
        metadata={"help": "The maximum number of train samples."},
    )

    max_validation_samples: Optional[int] = field(
        default=-1,
        metadata={"help": "The maximum number of validation samples."},
    )

    shuffle_seed: Optional[int] = field(
        default=42,
        metadata={"help": "The seed to shuffle the dataset."},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."},
    )

    # Optional
    tokenizer_name_or_path: str = field(
        default=None,
        metadata={"help": "The model checkpoint for tokenizer initialization."},
    )

    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The template for the chat."},
    )

    bos_token: Optional[str] = field(
        default=None,
        metadata={"help": "The beginning of sequence token."},
    )

    eos_token: Optional[str] = field(
        default=None,
        metadata={"help": "The end of sequence token."},
    )

    pad_token: Optional[str] = field(
        default=None,
        metadata={"help": "The padding token."},
    )

    tokenizer_post_processor_single: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration for the post processor."},
    )

    # LoRA (Low Rank Adaptation)
    use_lora: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use Low Rank Adaptation",
        },
    )
    lora_rank: Optional[int] = field(
        default=16,
        metadata={
            "help": "Set the rank for Low Rank Adaptation",
        },
    )
    lora_alpha: Optional[float] = field(
        default=4,
        metadata={
            "help": "Set the alpha for Low Rank Adaptation",
        },
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Set the dropout for Low Rank Adaptation",
        },
    )
    lora_target_modules: Optional[List[str]] = field(
        default_factory=list,
        metadata={
            "help": "Set the target modules for Low Rank Adaptation",
        },
    )


@dataclass
class TrainingArguments(TA):
    loss_type: Optional[str] = field(
        default="cross_entropy",
        metadata={"help": "The loss function to use for training."},
    )


@dataclass
class ExperimentArguments:
    wandb_group: str = field(
        metadata={"help": "Group for the experiment"},
    )

    # Optional
    wandb_tags: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "Tags for the experiment"},
    )


FinetuneArguments = [
    DataArguments,
    ModelArguments,
    TrainingArguments,
    ExperimentArguments,
]
