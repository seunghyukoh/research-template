from abc import ABC, abstractmethod
import os


class BaseExperiment(ABC):
    def __init__(
        self,
        run_name,
        config,
        config_dict,
        device_map=None,
        use_tracker=True,
    ):
        self.name = self.__get_experiment_name()
        self.run_name = run_name
        self.config = config
        self.config_dict = config_dict

        self.device_map = device_map

        self.use_tracker = use_tracker

    def __get_experiment_name(self):
        cur_dir = os.path.abspath(__file__)
        return os.path.dirname(cur_dir).split("/")[-1]

    @abstractmethod
    def run(self, run_name):
        raise NotImplementedError
