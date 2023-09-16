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
from packages.utils import tracker_init


class ExampleExperiment(BaseExperiment):
    def run(self):
        with tracker_init(
            name=self.run_name, config=self.config_dict, use_wandb=self.use_wandb
        ) as tracker:
            self.tracker = tracker
            return self._run()

    def _run(self):
        self.tracker.log({"accuracy": 0.2}, step=0)
        self.tracker.log({"accuracy": 0.3}, step=1)
        self.tracker.log({"accuracy": 0.4}, step=2)
