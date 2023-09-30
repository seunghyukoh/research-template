from typing import Any, Dict, Optional

import wandb

from .wandb_utils import set_wandb


class Tracker:
    def __init__(self, name, config) -> None:
        self.name = name
        self.config = config

    def __enter__(self):
        return self

    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: Optional[bool] = None,
        sync: Optional[bool] = None,
    ):
        print(data)
        # TODO: implement logging

    def log_artifact(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def init(name, config, use_wandb=True):
    set_wandb()
    if use_wandb:
        return wandb.init(name=name, config=config)

    return Tracker(name, config)


if __name__ == "__main__":
    with init(name="tracker", config={}, use_wandb=True) as tracker:
        tracker.log({"accuracy": 0.899})
