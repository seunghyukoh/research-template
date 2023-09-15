import os
import sys


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

from experiment import Experiment

import wandb
from args import parse_args
from packages.utils import set_wandb


if __name__ == "__main__":
    config, config_dict, run_name = parse_args()

    set_wandb()

    experiment = Experiment(
        config=config,
        config_dict=config_dict,
    )

    experiment.run(run_name)
    experiment.run(run_name + "_2")
    experiment.run(run_name + "_3")
