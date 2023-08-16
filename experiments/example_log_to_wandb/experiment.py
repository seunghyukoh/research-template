import os
import random
import sys

from accelerate import Accelerator

"""
Warning!!! This is a naive implementation.
If you want to use this script in other directory, please modify this part.
"""
os.chdir(os.getcwd().split("/experiments")[0])
sys.path.append("./src")

from packages.utils import set_wandb


def main(config, log):
    # simulate training
    epochs = config["epochs"]
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset

        # log metrics to wandb
        log({"acc": acc, "loss": loss}, step=epoch)


if __name__ == "__main__":
    from datetime import datetime

    set_wandb()

    accelerator = Accelerator(log_with="wandb")
    config = {
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    }

    now = datetime.utcnow()
    run_name = f"{config['architecture']}_{config['dataset']}_lr{config['learning_rate']}_{now}"

    accelerator.init_trackers(
        "research_template", config=config, init_kwargs={"wandb": {"name": run_name}}
    )

    main(config, accelerator.log)

    accelerator.end_training()
