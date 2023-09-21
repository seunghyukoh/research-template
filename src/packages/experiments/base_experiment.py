from abc import ABC, abstractmethod
from typing import Dict, List


class BaseExperiment(ABC):
    def __init__(
        self,
        name: str,
        run_name: str,
        config,
        config_dict: Dict,
        device_map: List[int] = None,
    ):
        assert name is not None, "Experiment name cannot be None"
        assert run_name is not None, "Run name cannot be None"
        assert config is not None, "Experiment config cannot be None"
        assert config_dict is not None, "Experiment config dict cannot be None"

        # Experiment name
        self.name = name
        # Run name (The name of the current run)
        self.run_name = run_name
        # Experiment config
        self.config = config
        # Experiment config as dict (For the tracker)
        self.config_dict = config_dict
        # Device map (GPU's to use)
        self.device_map = device_map

        # Add experiment name to config_dict
        # * Update this dict with any other info you want to track
        self.config_dict.update(
            dict(
                experiment_name=name,
                device_map=device_map,
            )
        )

        self.update_args()

    @abstractmethod
    def update_args(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError
