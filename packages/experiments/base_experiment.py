from abc import ABC, abstractmethod
from typing import Dict, List


class BaseExperiment(ABC):
    def __init__(
        self,
        name: str,
        run_name: str,
        config,
        device_map: List[int] = None,
    ):
        assert name is not None, "Experiment name cannot be None"
        assert run_name is not None, "Run name cannot be None"
        assert config is not None, "Experiment config cannot be None"

        # Experiment name
        self.name = name
        # Run name (The name of the current run)
        self.run_name = run_name
        # Experiment config
        self.config = config
        # Device map (GPU's to use)
        self.device_map = device_map

        self.update_args()

    @abstractmethod
    def update_args(self):
        # Ex)
        # self.model_args = ...
        # self.data_args = ...
        raise NotImplementedError

    @abstractmethod
    def run(self):
        # This single function should run the entire experiment
        raise NotImplementedError
