import random

import wandb
from packages.utils import ConfLoader, directory_setter, random_seeder


def main(args):
    lr = args.get("learning_rate", None)
    epochs = args.get("epochs", None)

    offset = random.random() / 5

    print(f"lr: {lr}")

    # simulating a training run
    for epoch in range(2, epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset
        print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
        wandb.log({"accuracy": acc, "loss": loss})

    return


if __name__ == "__main__":
    wandb.login()

    epochs = 10
    lr = 0.01

    run = wandb.init(
        # Set the project where this run will be logged
        project="research-template-project",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": epochs,
        },
    )

    main(wandb.config)
