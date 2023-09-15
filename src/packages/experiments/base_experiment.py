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
        # Experiment name
        self.name = self.__get_experiment_name()
        # Run name (The name of the current run)
        self.run_name = run_name
        # Experiment config
        self.config = config
        # Experiment config as dict (For the tracker)
        self.config_dict = config_dict
        # Add experiment name to config_dict
        # * Update this dict with any other info you want to track
        self.config_dict.update(
            dict(
                experiment_name=self.name,
            )
        )

        # Device map (GPU's to use)
        self.device_map = device_map

        # Tracker
        self.use_tracker = use_tracker

    def __get_experiment_name(self):
        cur_dir = os.path.abspath(__file__)
        return os.path.dirname(cur_dir).split("/")[-1]

    @abstractmethod
    def run(self, run_name):
        raise NotImplementedError
