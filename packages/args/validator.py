from typing import List

from .base_args import BaseArguments
from .base_validator import BaseArgumentValidator
from .data_args import DataArguments
from .experimental_args import ExperimentalArguments
from .model_args import ModelArguments
from .training_args import TrainingArguments


class ArgumentValidator(BaseArgumentValidator):
    @staticmethod
    def run(
        arguments: List[BaseArguments] = (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            ExperimentalArguments,
        ),
    ):
        ArgumentValidator._validate_each(arguments=arguments)
        # TODO: Validate across arguments

    @staticmethod
    def _validate_each(arguments: List[BaseArguments]):
        for argument in arguments:
            argument.validate()
