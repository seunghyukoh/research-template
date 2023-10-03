from abc import ABC, abstractmethod
from typing import Dict, List

from packages.utils import tracker_init

from .base_experiment import BaseExperiment


class WandBExperiment(BaseExperiment, ABC):
    def __init__(
        self,
        name: str,
        run_name: str,
        config,
        config_dict: Dict,
        device_map: List[int] = None,
    ):
        assert config_dict is not None, "Experiment config dict cannot be None"

        self.config_dict = config_dict
        # Add experiment name to config_dict
        # * Update this dict with any other info you want to track
        self.config_dict.update(
            dict(
                experiment_name=name,
                device_map=device_map,
            )
        )

        super().__init__(name, run_name, config, device_map)

    def run(self):
        with tracker_init(name=self.run_name, config=self.config_dict) as tracker:
            self.tracker = tracker
            return self._run()
