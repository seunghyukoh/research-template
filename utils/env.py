import os


def get_rank() -> int:
    """Get current rank in DDP/torchrun/accelerate environment."""
    # torchrun, accelerate, deepspeed 모두 RANK를 노출함
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    if "LOCAL_RANK" in os.environ:  # fallback
        return int(os.environ["LOCAL_RANK"])

    # 분산 아닌 경우
    return 0


def get_is_debug_mode() -> bool:
    debug = os.environ.get("DEBUG", "false").lower()
    if debug in ["true", "1"]:
        return True
    elif debug in ["false", "0"]:
        return False
    else:
        return False
