from dotenv import load_dotenv
from omegaconf import DictConfig
from trl import SFTConfig

import wandb
from utils.hydra_decorators import hydra_main_with_logging
from utils.parse_args import Parser

load_dotenv()


# Loads config from `configs/sft/config.yaml`
@hydra_main_with_logging(config_path="configs/sft", config_name="config")
def main(cfg: DictConfig):
    parser = Parser([SFTConfig])
    [training_args] = parser.parse_dict(cfg.training)

    if wandb.run is not None:  # rank 0 only
        wandb.config.update({"training": training_args.to_dict()}, allow_val_change=True)


if __name__ == "__main__":
    main()
