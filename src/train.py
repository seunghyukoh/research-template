import os
import sys
from typing import Any, Tuple

from transformers import HfArgumentParser

from args import DataArguments, ExperimentalArguments, ModelArguments, TrainingArguments


def parse_args() -> (
    Tuple[
        ModelArguments,
        DataArguments,
        TrainingArguments,
        ExperimentalArguments,
    ]
):
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ExperimentalArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (
            model_args,
            data_args,
            training_args,
            experimental_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            model_args,
            data_args,
            training_args,
            experimental_args,
        ) = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args, experimental_args


def main():
    model_args, data_args, training_args, experimental_args = parse_args()
