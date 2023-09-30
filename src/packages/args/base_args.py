from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseArguments(ABC):
    @abstractmethod
    def validate(self):
        raise NotImplementedError
