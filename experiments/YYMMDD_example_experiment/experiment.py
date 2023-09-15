import os
import sys

import wandb


def get_workspace():
    from dotenv import load_dotenv

    load_dotenv()

    workspace_name = os.getenv("WORKSPACE_NAME")

    workspace_dir_path = os.path.abspath(__file__).split(workspace_name)[0]
    workspace_path = os.path.join(workspace_dir_path, workspace_name)

    return workspace_path


def cd_to_root():
    workspace = get_workspace()
    os.chdir(workspace)


cd_to_root()
sys.path.append("./src")


### End of snippet ###
from packages.experiments import BaseExperiment
from packages.utils import set_wandb


class Experiment(BaseExperiment):
    def run(self):
        if self.use_tracker:
            return self._run_with_tracker()

        return self.__core()

    def _run_with_tracker(self):
        set_wandb()
        with wandb.init(name=self.run_name, config=self.config_dict) as run:
            self.tracker = run
            self.__core()
            self.tracker = None

    def __core(self):
        self.tracker.log({"accuracy": 0.2}, step=0)
        self.tracker.log({"accuracy": 0.3}, step=1)
        self.tracker.log({"accuracy": 0.4}, step=2)
