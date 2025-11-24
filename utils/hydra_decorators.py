import os
import traceback
from functools import wraps
from pprint import pprint

import hydra
import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from utils.env import get_is_debug_mode, get_rank


def mark_status(status: str, message: str = ""):
    work_dir = os.getcwd()
    status_file = os.path.join(work_dir, ".hydra", "status.yaml")

    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    with open(status_file, "w") as f:
        yaml.dump({"status": status, "message": message}, f)


def log_hydra_config(cfg: DictConfig, print_config: bool = True):
    rank = get_rank()

    # Print config only for rank 0
    if print_config and rank == 0:
        pprint(OmegaConf.to_object(cfg))

    if rank != 0 or get_is_debug_mode():
        return

    if cfg.logging.log_to == "wandb":
        import wandb

        # Use exp_id as group, with fallback to explicit group setting
        group = cfg.logging.get("group") or cfg.get("exp_id")
        tags = cfg.logging.tags if "tags" in cfg.logging else None
        notes = cfg.logging.notes if "notes" in cfg.logging else None

        wandb.init(
            id=cfg.logging.run_id,
            project=cfg.logging.project,
            name=cfg.task_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            group=group,
            tags=tags,
            notes=notes,
            save_code=True,
            settings=wandb.Settings(code_dir=to_absolute_path(".")),
            resume=cfg.logging.resume,
        )
        wandb.save(".hydra/config.yaml")
        wandb.save(".hydra/overrides.yaml")

    elif cfg.logging.log_to == "no":
        pass
    else:
        raise ValueError(f"Unsupported log_to: {cfg.logging.log_to}")


def hydra_main_with_logging(
    config_path="configs",
    config_name="config",
    version_base=None,
    print_config=True,
):
    def decorator(fn):

        @wraps(fn)
        def wrapper(cfg: DictConfig):

            log_hydra_config(cfg, print_config)

            try:
                result = fn(cfg)
                mark_status("success")
                return result

            except KeyboardInterrupt:
                mark_status("interrupted")
                raise

            except Exception as e:
                traceback_str = traceback.format_exc()
                mark_status("failure", traceback_str)
                raise e

        return hydra.main(
            config_path=config_path,
            config_name=config_name,
            version_base=version_base,
        )(wrapper)

    return decorator
