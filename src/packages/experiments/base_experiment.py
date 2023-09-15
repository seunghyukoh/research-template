from abc import ABC, abstractmethod


class BaseExperiment(ABC):
    def __init__(
        self,
        name,
        config,
        config_dict,
        device_map=None,
    ):
        self.name = name
        self.config = config
        self.config_dict = config_dict

        self.device_map = device_map

    @abstractmethod
    def run(self, run_name):
        raise NotImplementedError
