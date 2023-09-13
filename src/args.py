import argparse
from dataclasses import dataclass, field
from typing import Optional

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", type=str, default=None)
    parsed = parser.parse_args()

    config = parsed.config

    if parsed.config is None:
        config = HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments, ExperimentalArguments)
        ).parse_args_into_dataclasses()

    if config.endswith(".json"):
        config = HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments, ExperimentalArguments)
        ).parse_json_file(json_file=config)

    elif config.endswith(".yaml"):
        config = HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments, ExperimentalArguments)
        ).parse_yaml_file(yaml_file=config)

    else:
        raise ValueError("Config must be either a .json or .yaml file.")

    return config
