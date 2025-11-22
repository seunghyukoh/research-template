import subprocess

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers.utils.logging import get_logger

import wandb

logger = get_logger(__name__)


def dict_to_hydra_args(args_dict: dict, prefix: str = "", is_custom_args: bool = False) -> list:
    """Convert a dictionary to Hydra command-line arguments."""
    hydra_args = []
    for key, value in args_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            hydra_args.extend(
                dict_to_hydra_args(value, prefix=full_key, is_custom_args=is_custom_args)
            )
        elif isinstance(value, list):
            hydra_args.append(f"{'+' if is_custom_args else ''}{full_key}='{','.join(value)}'")
        else:
            # Regular key=value format
            hydra_args.append(f"{'+' if is_custom_args else ''}{full_key}={value}")

    return hydra_args


def check_if_done(run_id: str) -> bool:
    logger.info(f"Checking if done: seunghyukoh-kaist/research-template/{run_id}")
    api = wandb.Api()
    try:
        api.run(f"seunghyukoh-kaist/research-template/{run_id}")
        return True
    except Exception as e:
        logger.error(f"Error checking if done: {e}")
        return False


# Loads config from `experiments/000-demo-sft.yaml`
@hydra.main(config_path="experiments", version_base=None)
def main(cfg: DictConfig):
    for run in cfg.runs:
        command = run.command  # accelerate launch run_sft.py
        args = run.args
        custom_args = run.custom_args

        # Convert args dict to Hydra command-line arguments
        args_dict = OmegaConf.to_container(args, resolve=True)
        custom_args_dict = OmegaConf.to_container(custom_args, resolve=True)
        hydra_args = dict_to_hydra_args(args_dict)
        custom_hydra_args = dict_to_hydra_args(custom_args_dict, is_custom_args=True)
        hydra_args.extend(custom_hydra_args)

        # Build full command
        if "{args}" in command:
            # Replace {args} placeholder with formatted arguments
            args_str = " ".join(hydra_args)
            full_command = command.format(args=args_str)
            # Split into command and arguments for subprocess
            cmd_parts = full_command.split()
        else:
            # Append args to command
            cmd_parts = command.split() + hydra_args

        is_done = check_if_done(run.args.logging.run_id)
        if is_done:
            logger.info(f"Run {run.args.logging.run_id} is already done")
            continue

        logger.info(f"Executing: {' '.join(cmd_parts)}")
        result = subprocess.run(cmd_parts, check=False)
        if result.returncode != 0:
            logger.error(f"Command failed with return code {result.returncode}")
        else:
            logger.info("Command executed successfully")

        # TODO: Run command


if __name__ == "__main__":
    main()
