import os
import sys
from datetime import datetime

from accelerate import Accelerator
from example_log_to_wandb.experiment import main as example_experiment

"""
Warning!!! This is a naive implementation.
If you want to use this script in other directory, please modify this part.
"""
os.chdir(os.getcwd().split("/experiments")[0])
sys.path.append("./src")

from packages.utils import set_wandb


def main():
    set_wandb()

    project_name = "research_template"

    accelerator = Accelerator(log_with="wandb")

    experiments = [example_experiment, example_experiment]
    configs = [
        {
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
        {
            "learning_rate": 0.02,
            "architecture": "U-Net",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
    ]
    logger = accelerator.log

    for experiment, config in zip(experiments, configs):
        now = datetime.utcnow()
        run_name = f"{config['architecture']}_{config['dataset']}_lr{config['learning_rate']}_{now}"

        accelerator.init_trackers(
            project_name,
            config=config,
            init_kwargs={"wandb": {"name": run_name}},
        )

        experiment(config, logger)

        accelerator.end_training()


if __name__ == "__main__":
    main()
