from dotenv import load_dotenv
import os


def set_wandb():
    load_dotenv()

    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    WANDB_PROJECT = os.environ.get("WANDB_PROJECT")

    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
