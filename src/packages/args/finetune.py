import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from transformers import TrainingArguments as TA

from packages.args.base import BaseArguments


@dataclass
class DataArguments:
    dataset_path: str = field(
        metadata={"help": "The path to the dataset."},
    )

    dataset_name: Optional[str] = field(
        metadata={"help": "The name of the dataset for logging."},
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

    def __post_init__(self):
        if self.dataset_name is None:
            self.dataset_name = os.path.basename(self.dataset_path)

    def to_dict(self) -> dict:
        """Convert the arguments to a dictionary.

        Returns:
            Dictionary containing all the arguments
        """
        return asdict(self)


@dataclass
class ModelArguments:
    model_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."},
    )

    model_name: Optional[str] = field(
        metadata={"help": "The model name for logging."},
    )

    tokenizer_path: str = field(
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

    def __post_init__(self):
        if self.model_name is None:
            self.model_name = os.path.basename(self.model_path)

    def to_dict(self) -> dict:
        """Convert the arguments to a dictionary.

        Returns:
            Dictionary containing all the arguments
        """
        return asdict(self)


@dataclass
class TrainingArguments(TA):
    run_name: str = None
    hub_model_revision: str = None

    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    base_output_dir: Optional[str] = field(
        default="output/",
        metadata={
            "help": "The base output directory where the model predictions and checkpoints will be written."
        },
    )
    loss_type: Optional[str] = field(
        default="cross_entropy",
        metadata={"help": "The loss function to use for training."},
    )


@dataclass
class ExperimentArguments:
    id: str = field(
        metadata={"help": "The id for the experiment"},
    )

    wandb_group: str = field(
        metadata={"help": "Group for the experiment"},
    )

    # Optional
    wandb_tags: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "Tags for the experiment"},
    )

    def to_dict(self) -> dict:
        """Convert the arguments to a dictionary.

        Returns:
            Dictionary containing all the arguments
        """
        return asdict(self)


class FinetuneArguments(BaseArguments):
    ARG_COMPONENTS = [
        DataArguments,
        ModelArguments,
        TrainingArguments,
        ExperimentArguments,
    ]

    data_args: DataArguments = None
    model_args: ModelArguments = None
    training_args: TrainingArguments = None
    experiment_args: ExperimentArguments = None

    def __init__(
        self,
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        experiment_args: ExperimentArguments,
    ):
        super().__init__()

        assert data_args is not None
        assert model_args is not None
        assert training_args is not None
        assert experiment_args is not None

        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        self.experiment_args = experiment_args

        # Run name, Output dir
        self.training_args.run_name = self.get_run_name()
        self.training_args.output_dir = self.get_output_dir()

    def to_dict(self):
        return {
            "id": self.experiment_args.id,
            **self.data_args.to_dict(),
            **self.model_args.to_dict(),
            **self.training_args.__dict__,
            **self.experiment_args.to_dict(),
        }

    def get_run_name(self):
        wandb_group = self.experiment_args.wandb_group
        model_name = self.model_args.model_name
        dataset_name = self.data_args.dataset_name

        tags_str = (
            "-".join(sorted(self.experiment_args.wandb_tags))
            if self.experiment_args.wandb_tags
            else "default"
        )

        run_name = f"{wandb_group}-{model_name}-{dataset_name}-tags.{tags_str}".lower()
        return run_name

    def get_output_dir(self):
        wandb_group = self.experiment_args.wandb_group
        model_name = self.model_args.model_name
        dataset_name = self.data_args.dataset_name

        tags_str = (
            "-".join(sorted(self.experiment_args.wandb_tags))
            if self.experiment_args.wandb_tags
            else "default"
        )

        base_output_dir = self.training_args.base_output_dir

        # {base_output_dir}/{wandb_group}/{model_name}/{dataset_name}/{tags_str}-{version_id}
        output_dir = os.path.join(
            base_output_dir,
            wandb_group,
            model_name,
            dataset_name,
            tags_str,
            self.experiment_args.id,
        ).lower()

        # Log file
        self.training_args.log_file = os.path.join(output_dir, "train.log")
        if not os.path.exists(self.training_args.log_file):
            os.makedirs(os.path.dirname(self.training_args.log_file), exist_ok=True)

        return output_dir
