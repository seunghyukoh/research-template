from abc import ABC, abstractmethod


class BaseArguments(ABC):
    ARG_COMPONENTS = []

    id: str = None

    @abstractmethod
    def to_dict(self):
        pass
