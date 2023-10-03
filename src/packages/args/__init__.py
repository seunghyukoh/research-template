import argparse
from typing import List

from transformers import HfArgumentParser

from .base_args import BaseArguments
from .base_validator import BaseArgumentValidator
from .data_args import DataArguments
from .experimental_args import ExperimentalArguments
from .model_args import ModelArguments
from .training_args import TrainingArguments
from .validator import ArgumentValidator


def parse_args(
    dataclass_types: List[BaseArguments] = (
        ModelArguments,
        DataArguments,
        TrainingArguments,
        ExperimentalArguments,
    ),
    validator: BaseArgumentValidator = ArgumentValidator,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", dest="run_name", type=str, default=None)
    parser.add_argument("--config", dest="config", type=str, default=None)
    parsed = parser.parse_args()

    run_name = parsed.run_name
    config_file = parsed.config

    if config_file is None:
        configs = HfArgumentParser(dataclass_types).parse_args_into_dataclasses()

    if config_file.endswith(".json"):
        configs = HfArgumentParser(dataclass_types).parse_json_file(
            json_file=config_file
        )

    elif config_file.endswith(".yaml"):
        configs = HfArgumentParser(dataclass_types).parse_yaml_file(
            yaml_file=config_file
        )

    else:
        raise ValueError("Config must be either a .json or .yaml file.")

    # Validate configs
    validator.run(configs)

    config_dict = {}
    for config in configs:
        config_dict.update(config.__dict__)

    return configs, config_dict, run_name
