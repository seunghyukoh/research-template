import random
from datetime import datetime

from accelerate import Accelerator

"""
! Run this code at the root directory of this project.
! Otherwise, you will get an error while importing following packages.
"""
from src.packages.utils import set_wandb

set_wandb()

accelerator = Accelerator(log_with="wandb")
config = {
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
}

now = datetime.utcnow()
run_name = (
    f"{config['architecture']}_{config['dataset']}_lr{config['learning_rate']}_{now}"
)

accelerator.init_trackers(
    "research_template", config=config, init_kwargs={"wandb": {"name": run_name}}
)


# simulate training
epochs = config["epochs"]
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # log metrics to wandb
    accelerator.log({"acc": acc, "loss": loss}, step=epoch)

accelerator.end_training()
