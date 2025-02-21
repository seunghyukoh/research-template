from abc import ABC, abstractmethod
from uuid import uuid4


class BaseArguments(ABC):
    ARG_COMPONENTS = []

    uuid: str = None

    def __init__(self, *args, **kwargs):
        self.uuid = str(uuid4())

    @abstractmethod
    def to_dict(self):
        pass
