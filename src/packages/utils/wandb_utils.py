import os

from dotenv import load_dotenv


def set_wandb():
    load_dotenv()

    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    WANDB_PROJECT = os.environ.get("WANDB_PROJECT")

    assert WANDB_API_KEY != "", "Please set WANDB_API_KEY in .env"
    assert WANDB_PROJECT != "", "Please set WANDB_PROJECT in .env"

    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
