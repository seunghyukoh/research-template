from typing import Callable, Tuple

from accelerate.utils.memory import clear_device_cache, should_reduce_batch_size
from transformers.utils.logging import get_logger

logger = get_logger(__name__)


AUTO_BATCH_SIZE_TRAIN_STEPS = 3


def get_max_batch_size(func: Callable[[int], None], starting_batch_size=1) -> int:
    """Find maximum possible batch size

    Args:
        func: Function that takes batch size as an argument and executes it. Should raise RuntimeError if memory is insufficient.
        starting_batch_size: Starting batch size

    Returns:
        Maximum possible batch size
    Raises:
        RuntimeError: If no executable batch size is found
    """
    clear_device_cache(garbage_collection=True)

    def reduce_batch_size_fn():
        nonlocal batch_size
        batch_size = batch_size // 2
        return batch_size

    batch_size = starting_batch_size
    while True:
        if batch_size == 0:
            raise RuntimeError("No executable batch size found, reached zero.")
        try:
            logger.info(f"Trying batch size: {batch_size}")
            func(batch_size)  # Check if the batch size is executable
            return batch_size  # Return the successful batch size
        except Exception as e:
            if should_reduce_batch_size(e):
                clear_device_cache(garbage_collection=True)
                batch_size = reduce_batch_size_fn()
                logger.info(f"Reduced batch size to {batch_size}")
            else:
                raise


def optimize_batch_size(
    demo_run_fn: Callable[[int], None],
    effective_batch_size: int,
    starting_batch_size: int,
) -> Tuple[int, int]:
    """Find optimal batch size and gradient accumulation steps

    Args:
        demo_run_fn: Function that takes batch size as an argument and executes it. Should raise RuntimeError if memory is insufficient.
        effective_batch_size: Desired effective batch size (per_device_batch_size * gradient_accumulation_steps)
        starting_batch_size: Starting batch size

    Returns:
        Optimal batch size and gradient accumulation steps
    Raises:
        RuntimeError: If the batch size is not found or the gradient accumulation steps are not found
    """
    batch_size = get_max_batch_size(
        demo_run_fn,
        starting_batch_size=starting_batch_size,
    )
    gradient_accumulation_steps = max(effective_batch_size // batch_size, 1)

    return batch_size, gradient_accumulation_steps
