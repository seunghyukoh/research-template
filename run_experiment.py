import subprocess
from typing import Literal, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers.utils.logging import get_logger

import wandb

logger = get_logger(__name__)


def ifelse(cond, a, b):
    return a if cond else b


OmegaConf.register_new_resolver("ifelse", ifelse)


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
            hydra_args.append(f"{'+' if is_custom_args else ''}{full_key}={value}")
        else:
            # Regular key=value format
            hydra_args.append(f"{'+' if is_custom_args else ''}{full_key}={value}")

    return hydra_args


def build_command(command: str, hydra_args: list) -> list:
    if "{args}" in command:
        # Replace {args} placeholder with formatted arguments
        args_str = " ".join(hydra_args)
        full_command = command.format(args=args_str)
        # Split into command and arguments for subprocess
        cmd_parts = full_command.split()
    else:
        # Append args to command
        cmd_parts = command.split() + hydra_args
    return cmd_parts


def check_run_status_from_wandb(
    run_id: str,
) -> Tuple[bool, Literal["running", "finished", "crashed", "killed", "failed"]]:
    """
    Check the status of a WandB run.

    Returns:
        (exists, is_running):
            - exists: True if the run exists in WandB
            - status:
    """
    api = wandb.Api()
    run_path = f"seunghyukoh-kaist/research-template/{run_id}"

    try:
        run = api.run(run_path)
        state = run.state  # 'running', 'finished', 'crashed', 'killed', 'failed'

        if state == "running":
            logger.info(f"Run {run_id} is already running (state: {state}): {run.url}")
            return True, "running"
        elif state in ["finished", "crashed", "killed", "failed"]:
            logger.info(f"Run {run_id} already exists with state: {state}: {run.url}")
            return True, state
        else:
            logger.warning(f"Run {run_id} has unknown state: {state}")
            return True, state
    except wandb.errors.CommError as e:
        # Run doesn't exist yet
        logger.info(f"Run {run_id} does not exist yet: {e}")
        return False, None
    except Exception as e:
        logger.error(f"Error checking run status for {run_id}: {e}")
        # On error, assume run doesn't exist to be safe
        return False, None


# Loads config from `experiments/000-demo-sft.yaml`
@hydra.main(config_path="experiments", version_base=None)
def main(cfg: DictConfig):
    global_resume = cfg.resume
    global_skip_killed = cfg.skip_killed
    global_skip_crashed = cfg.skip_crashed
    global_skip_failed = cfg.skip_failed

    shared_args = cfg.get("shared_args", None)
    shared_hydra_args = (
        dict_to_hydra_args(
            OmegaConf.to_container(shared_args, resolve=True),
        )
        if shared_args
        else []
    )

    runs = cfg.runs or []
    for run in runs:
        try:
            assert "command" in run, "command is required"
            assert "args" in run, "args is required"
            assert "logging" in run.args, "logging is required"
            assert "run_id" in run.args.logging, "run_id is required"

            command = run.command
            resume = run.args.logging.get("resume", global_resume)
            skip_killed = run.args.get("skip_killed", global_skip_killed)
            skip_crashed = run.args.get("skip_crashed", global_skip_crashed)
            skip_failed = run.args.get("skip_failed", global_skip_failed)
            run_id = run.args.logging.run_id

            # Check run status using WandB API (works across multiple servers)
            exists, state = check_run_status_from_wandb(run_id)

            if exists and state == "running":
                logger.info(f"Run {run_id} is already running, skipping")
                continue
            elif state in ["finished", "crashed", "killed", "failed"]:
                if (not resume or resume == "never") and state == "finished":
                    logger.info(f"Run {run_id} is finished, skipping")
                    continue
                if skip_killed and state == "killed":
                    logger.info(f"Run {run_id} is killed, skipping")
                    continue
                if skip_crashed and state == "crashed":
                    logger.info(f"Run {run_id} is crashed, skipping")
                    continue
                if skip_failed and state == "failed":
                    logger.info(f"Run {run_id} is failed, skipping")

            # Convert args dict to Hydra command-line arguments
            hydra_args = dict_to_hydra_args(
                OmegaConf.to_container(run.args, resolve=True),
            )
            custom_hydra_args = dict_to_hydra_args(
                OmegaConf.to_container(run.custom_args, resolve=True), is_custom_args=True
            )
            hydra_args.extend(custom_hydra_args)
            if shared_hydra_args:
                # FIXME: Override shared_hydra_args with hydra_args
                shared_hydra_args.extend(hydra_args)

            cmd_parts = build_command(command, hydra_args)

            # Run doesn't exist or is not running, safe to execute
            logger.info(f"Executing: {' '.join(cmd_parts)}")
            result = subprocess.run(cmd_parts, check=False)
            if result.returncode != 0:
                logger.error(f"Command failed with return code {result.returncode}")
            else:
                logger.info("Command executed successfully")

        except Exception as e:
            logger.error(f"Error running experiment: {e}")
            continue


if __name__ == "__main__":
    main()
