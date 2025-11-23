import math
import os
import subprocess
from typing import Literal, Tuple

import hydra
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from transformers.utils.logging import get_logger

import wandb

logger = get_logger(__name__)


def ifelse(cond, a, b):
    return a if cond else b


OmegaConf.register_new_resolver("ifelse", ifelse)


def detect_gpu_memory_per_device() -> dict:
    """Detect GPU memory capacity for each device and group by memory size.

    Returns:
        dict: Mapping from memory size (GB) to list of GPU indices
        Example: {40: [0, 1], 80: [2, 3]} means GPUs 0,1 have 40GB and GPUs 2,3 have 80GB
    """
    if not torch.cuda.is_available():
        return {}

    gpu_groups = {}
    num_gpus = torch.cuda.device_count()

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)  # Convert bytes to GB
        memory_gb_rounded = round(memory_gb)  # Round to nearest GB

        if memory_gb_rounded not in gpu_groups:
            gpu_groups[memory_gb_rounded] = []
        gpu_groups[memory_gb_rounded].append(i)

        logger.info(
            f"GPU {i}: {props.name}, {memory_gb:.2f} GB memory (grouped as {memory_gb_rounded} GB)"
        )

    return gpu_groups


def calculate_num_gpus_needed(
    required_memory_gb: float, gpu_memory_groups: dict
) -> Tuple[int, int]:
    """Calculate how many GPUs are needed based on required memory.

    Args:
        required_memory_gb: Required GPU memory in GB
        gpu_memory_groups: Mapping from memory size to GPU indices

    Returns:
        Tuple of (num_gpus_needed, memory_per_gpu):
            - num_gpus_needed: Number of GPUs to allocate
            - memory_per_gpu: Memory capacity of each GPU in the selected group

    Raises:
        ValueError: If no GPU group can satisfy the memory requirement
    """
    if not gpu_memory_groups:
        raise ValueError("No GPUs available")

    # Try to find the smallest GPU type that can fit the requirement
    sorted_groups = sorted(gpu_memory_groups.items(), key=lambda x: x[0])

    for memory_per_gpu, gpu_indices in sorted_groups:
        num_gpus_needed = math.ceil(required_memory_gb / memory_per_gpu)

        # Check if we have enough GPUs of this type
        if num_gpus_needed <= len(gpu_indices):
            return num_gpus_needed, memory_per_gpu

    # If no single GPU type can satisfy, use the largest GPU type available
    largest_memory, largest_gpus = sorted_groups[-1]
    num_gpus_needed = math.ceil(required_memory_gb / largest_memory)

    if num_gpus_needed > len(largest_gpus):
        raise ValueError(
            f"Cannot satisfy memory requirement of {required_memory_gb} GB. "
            f"Largest GPU group has {len(largest_gpus)} GPUs with {largest_memory} GB each "
            f"(total {len(largest_gpus) * largest_memory} GB available, need {required_memory_gb} GB)"
        )

    return num_gpus_needed, largest_memory


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


def execute_single_run(
    run_config: dict,
    shared_hydra_args: list,
    global_resume: bool,
    global_skip_killed: bool,
    global_skip_crashed: bool,
    global_skip_failed: bool,
):
    """Execute a single experiment run in parallel using Ray.

    Note: This function is wrapped with @ray.remote dynamically to support
    per-run GPU resource specifications.
    """
    try:
        # Get GPU IDs assigned by Ray and set CUDA_VISIBLE_DEVICES
        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            logger.info(f"Run assigned GPU(s): {gpu_ids}")
        else:
            logger.info("Run assigned no GPUs (CPU-only execution)")

        assert "command" in run_config, "command is required"
        assert "args" in run_config, "args is required"
        assert "logging" in run_config["args"], "logging is required"
        assert "run_id" in run_config["args"]["logging"], "run_id is required"

        command = run_config["command"]
        resume = run_config["args"]["logging"].get("resume", global_resume)
        skip_killed = run_config["args"].get("skip_killed", global_skip_killed)
        skip_crashed = run_config["args"].get("skip_crashed", global_skip_crashed)
        skip_failed = run_config["args"].get("skip_failed", global_skip_failed)
        run_id = run_config["args"]["logging"]["run_id"]

        # Check run status using WandB API
        exists, state = check_run_status_from_wandb(run_id)

        if exists and state == "running":
            logger.info(f"Run {run_id} is already running, skipping")
            return {"success": True, "skipped": True, "reason": "already running", "run_id": run_id}
        elif state in ["finished", "crashed", "killed", "failed"]:
            if (not resume or resume == "never") and state == "finished":
                logger.info(f"Run {run_id} is finished, skipping")
                return {"success": True, "skipped": True, "reason": "finished", "run_id": run_id}
            if skip_killed and state == "killed":
                logger.info(f"Run {run_id} is killed, skipping")
                return {"success": True, "skipped": True, "reason": "killed", "run_id": run_id}
            if skip_crashed and state == "crashed":
                logger.info(f"Run {run_id} is crashed, skipping")
                return {"success": True, "skipped": True, "reason": "crashed", "run_id": run_id}
            if skip_failed and state == "failed":
                logger.info(f"Run {run_id} is failed, skipping")
                return {"success": True, "skipped": True, "reason": "failed", "run_id": run_id}

        # Convert args dict to Hydra command-line arguments
        hydra_args = dict_to_hydra_args(run_config["args"])
        if run_config.get("custom_args"):
            custom_hydra_args = dict_to_hydra_args(run_config["custom_args"], is_custom_args=True)
            hydra_args.extend(custom_hydra_args)
        if shared_hydra_args:
            all_args = shared_hydra_args.copy()
            all_args.extend(hydra_args)
            hydra_args = all_args

        cmd_parts = build_command(command, hydra_args)

        # Execute the command
        logger.info(f"Executing: {' '.join(cmd_parts)}")
        result = subprocess.run(cmd_parts, check=False)
        if result.returncode != 0:
            logger.error(f"Command failed with return code {result.returncode}")
            return {"success": False, "returncode": result.returncode, "run_id": run_id}
        else:
            logger.info(f"Command executed successfully for run {run_id}")
            return {"success": True, "skipped": False, "run_id": run_id}

    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        run_id = run_config.get("args", {}).get("logging", {}).get("run_id", "unknown")
        return {"success": False, "error": str(e), "run_id": run_id}


# Loads config from `experiments/000-demo-sft.yaml`
@hydra.main(config_path="experiments", version_base=None)
def main(cfg: DictConfig):
    num_workers = cfg.num_workers

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

    shared_custom_args = cfg.get("shared_custom_args", None)
    shared_custom_hydra_args = (
        dict_to_hydra_args(
            OmegaConf.to_container(shared_custom_args, resolve=True),
            is_custom_args=True,
        )
        if shared_custom_args
        else []
    )
    if shared_custom_hydra_args:
        shared_hydra_args = shared_custom_hydra_args + shared_hydra_args

    # Detect available GPUs and their memory capacity
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"Detected {num_gpus} GPU(s) available")

    # Group GPUs by memory capacity
    gpu_memory_groups = detect_gpu_memory_per_device()

    # Get default GPU memory for runs that don't specify memory requirement
    default_gpu_memory = None
    if gpu_memory_groups:
        # Use the smallest GPU memory as default (to avoid over-allocation)
        default_gpu_memory = min(gpu_memory_groups.keys())
        logger.info(f"Default GPU memory requirement set to {default_gpu_memory} GB")

    # Initialize Ray with CPU and GPU resources
    ray.init(num_cpus=num_workers, num_gpus=num_gpus, ignore_reinit_error=True)
    logger.info(f"Initialized Ray with {num_workers} CPU workers and {num_gpus} GPUs")

    try:
        runs = cfg.runs or []

        # Get global resources if specified
        global_resources = (
            OmegaConf.to_container(cfg.get("resources"), resolve=True)
            if cfg.get("resources")
            else None
        )

        # Convert OmegaConf runs to serializable dictionaries
        run_configs = []
        for run in runs:
            # Use run-specific resources if defined, otherwise use global resources
            run_resources = (
                OmegaConf.to_container(run.get("resources"), resolve=True)
                if run.get("resources")
                else global_resources
            )

            run_config = {
                "command": run.command,
                "args": OmegaConf.to_container(run.args, resolve=True),
                "custom_args": (
                    OmegaConf.to_container(run.get("custom_args"), resolve=True)
                    if run.get("custom_args")
                    else None
                ),
                "resources": run_resources,
            }
            run_configs.append(run_config)

        # Submit all runs to Ray for parallel execution
        logger.info(f"Submitting {len(run_configs)} runs for parallel execution")
        futures = []
        for run_config in run_configs:
            # Get resource requirements for this run
            resources = run_config.get("resources") or {}

            # Support both old (num_gpus) and new (gpu_memory_gb) formats
            if "gpu_memory_gb" in resources:
                required_memory_gb = resources["gpu_memory_gb"]
                if num_gpus == 0:
                    raise ValueError(
                        f"Run requires {required_memory_gb} GB GPU memory but no GPUs available"
                    )

                # Calculate how many GPUs are needed
                num_gpus_required, memory_per_gpu = calculate_num_gpus_needed(
                    required_memory_gb, gpu_memory_groups
                )
                logger.info(
                    f"Run requires {required_memory_gb} GB memory: "
                    f"allocating {num_gpus_required} GPU(s) with {memory_per_gpu} GB each"
                )
            elif "num_gpus" in resources:
                # Legacy support: num_gpus directly specifies GPU count
                num_gpus_required = resources["num_gpus"]
                logger.warning(
                    f"Using deprecated 'num_gpus' parameter. Consider using 'gpu_memory_gb' instead."
                )
            else:
                # Default: use single GPU worth of memory if available
                if num_gpus > 0 and default_gpu_memory:
                    num_gpus_required = 1
                    logger.info(
                        f"No resource specification, using default: 1 GPU ({default_gpu_memory} GB)"
                    )
                else:
                    num_gpus_required = 0
                    logger.info("No resource specification and no GPUs available, using CPU")

            # Create a Ray remote function with the specified GPU resources
            remote_fn = ray.remote(num_gpus=num_gpus_required)(execute_single_run)

            run_id = run_config["args"]["logging"]["run_id"]
            logger.info(f"Submitting run {run_id} with {num_gpus_required} GPU(s)")

            # Submit the task
            future = remote_fn.remote(
                run_config,
                shared_hydra_args,
                global_resume,
                global_skip_killed,
                global_skip_crashed,
                global_skip_failed,
            )
            futures.append(future)

        # Wait for all runs to complete and collect results
        results = ray.get(futures)

        # Log summary of results
        logger.info("All runs completed")
        for result in results:
            if result.get("skipped"):
                logger.info(f"Run {result['run_id']}: skipped ({result['reason']})")
            elif result.get("success"):
                logger.info(f"Run {result['run_id']}: completed successfully")
            else:
                error_msg = result.get("error", f"returncode {result.get('returncode')}")
                logger.error(f"Run {result['run_id']}: failed ({error_msg})")

    finally:
        # Cleanup Ray resources
        ray.shutdown()
        logger.info("Ray shutdown complete")


if __name__ == "__main__":
    main()
