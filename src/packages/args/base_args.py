from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class BaseArguments(ABC):
    @abstractmethod
    def validate(self):
        raise NotImplementedError
