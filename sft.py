import os

from dotenv import load_dotenv
from omegaconf import DictConfig

from utils.hydra_decorators import hydra_main_with_logging

load_dotenv()


# Loads config from `configs/sft/config.yaml`
@hydra_main_with_logging(config_path="configs/sft", config_name="config")
def main(cfg: DictConfig):
    pass


if __name__ == "__main__":
    main()
