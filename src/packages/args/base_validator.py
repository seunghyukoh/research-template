from abc import ABC, abstractmethod
from typing import List

from .base_args import BaseArguments


class BaseArgumentValidator(ABC):
    @abstractmethod
    def run(self, arguments: List[BaseArguments]):
        raise NotImplementedError

    def _validate_each(self, arguments: List[BaseArguments]):
        for argument in arguments:
            argument.validate()
