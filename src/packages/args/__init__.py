import argparse

from transformers import HfArgumentParser

from .data_args import DataArguments
from .experimental_args import ExperimentalArguments
from .model_args import ModelArguments
from .training_args import TrainingArguments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", dest="run_name", type=str, default=None)
    parser.add_argument("--config", dest="config", type=str, default=None)
    parsed = parser.parse_args()

    run_name = parsed.run_name
    config_file = parsed.config

    if config_file is None:
        configs = HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments, ExperimentalArguments)
        ).parse_args_into_dataclasses()

    if config_file.endswith(".json"):
        configs = HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments, ExperimentalArguments)
        ).parse_json_file(json_file=config_file)

    elif config_file.endswith(".yaml"):
        configs = HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments, ExperimentalArguments)
        ).parse_yaml_file(yaml_file=config_file)

    else:
        raise ValueError("Config must be either a .json or .yaml file.")

    config_dict = {}
    for config in configs:
        config_dict.update(config.__dict__)

    return configs, config_dict, run_name
