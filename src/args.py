from dataclasses import dataclass, field
from typing import List, Optional, Union

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers import TrainingArguments as HfTrainingArguments

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model checkpoint for initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use"},
    )


@dataclass
class TrainingArguments(HfTrainingArguments):
    max_position_embeddings: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum position embedding per segment."},
    )


@dataclass
class ExperimentalArguments:
    fast_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use fast attention or not (experimental)"},
    )
