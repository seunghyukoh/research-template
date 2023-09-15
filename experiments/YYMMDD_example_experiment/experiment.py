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


def get_experiment_name():
    cur_dir = os.path.abspath(__file__)
    return os.path.dirname(cur_dir).split("/")[-1]


cd_to_root()
sys.path.append("./src")


### End of snippet ###
from packages.experiments import BaseExperiment

EXPERIMENT_NAME = get_experiment_name()


class Experiment(BaseExperiment):
    def __init__(self, config, **kwargs):
        super().__init__(name=EXPERIMENT_NAME, config=config, **kwargs)

    def run(self, run_name):
        with wandb.init(name=run_name, config=self.config_dict) as run:
            print(f"Running {run_name}...")

            run.log({"example_metric": 0.5})
