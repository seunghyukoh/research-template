import json
import math
import os
import shlex
import signal
import subprocess
import sys
import threading
from datetime import datetime
from typing import Literal, Optional, Tuple

import hydra
import ray
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers.utils.logging import get_logger

import wandb

load_dotenv()

logger = get_logger(__name__)


class GracefulShutdownHandler:
    """Handle graceful shutdown of running processes with timeout."""

    def __init__(self, timeout: int = 60):
        """
        Initialize graceful shutdown handler.

        Args:
            timeout: Maximum time (in seconds) to wait for processes to terminate
        """
        self.timeout = timeout
        self.shutdown_requested = False
        self.force_shutdown = False
        self.processes = []
        self.lock = threading.Lock()

    def register_process(self, proc: subprocess.Popen):
        """Register a process for tracking."""
        with self.lock:
            self.processes.append(proc)

    def unregister_process(self, proc: subprocess.Popen):
        """Unregister a process after it completes."""
        with self.lock:
            if proc in self.processes:
                self.processes.remove(proc)

    def handle_signal(self, signum, frame):
        """Handle SIGINT/SIGTERM signals."""
        if self.force_shutdown:
            logger.error("Force shutdown: Killing all processes immediately")
            with self.lock:
                for proc in self.processes:
                    if proc.poll() is None:
                        proc.kill()
            sys.exit(1)

        if self.shutdown_requested:
            logger.warning("Shutdown already in progress. Press Ctrl+C again to force kill.")
            self.force_shutdown = True
            return

        logger.info(
            f"Graceful shutdown requested (signal {signum}). Sending SIGTERM to running processes..."
        )
        self.shutdown_requested = True

        # Send SIGTERM to all running subprocesses
        with self.lock:
            for proc in self.processes:
                if proc.poll() is None:  # Process still running
                    try:
                        proc.terminate()
                        logger.info(f"Sent SIGTERM to process {proc.pid}")
                    except Exception as e:
                        logger.error(f"Failed to terminate process {proc.pid}: {e}")

        # Start timeout timer
        timer = threading.Timer(self.timeout, self._force_kill)
        timer.daemon = True
        timer.start()

    def _force_kill(self):
        """Force kill all processes after timeout."""
        logger.warning(f"Timeout ({self.timeout}s) reached. Force killing remaining processes...")
        with self.lock:
            for proc in self.processes:
                if proc.poll() is None:
                    try:
                        proc.kill()
                        logger.info(f"Force killed process {proc.pid}")
                    except Exception as e:
                        logger.error(f"Failed to kill process {proc.pid}: {e}")

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_requested


def log_structured(level: str, message: str, **kwargs):
    """Log structured data in JSON format for easier parsing.

    Args:
        level: Log level (info, warning, error, etc.)
        message: Log message
        **kwargs: Additional structured data to log
    """
    log_data = {"timestamp": datetime.now().isoformat(), "message": message, **kwargs}
    log_msg = json.dumps(log_data)

    if level == "info":
        logger.info(log_msg)
    elif level == "warning":
        logger.warning(log_msg)
    elif level == "error":
        logger.error(log_msg)
    elif level == "debug":
        logger.debug(log_msg)
    else:
        logger.info(log_msg)


def ifelse(cond, a, b):
    """OmegaConf resolver for conditional logic.

    Returns 'a' if 'cond' is truthy, otherwise returns 'b'.

    Args:
        cond: Condition to evaluate (truthy or falsy)
        a: Value to return if cond is truthy
        b: Value to return if cond is falsy
    """
    return a if cond else b


OmegaConf.register_new_resolver("ifelse", ifelse)


def _deep_merge_dict(base: dict, update: dict) -> None:
    """Deep merge update dict into base dict (in-place).

    Args:
        base: Base dictionary to merge into
        update: Dictionary with updates to merge
    """
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value


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
        cmd_parts = shlex.split(full_command)
    else:
        # Append args to command
        cmd_parts = shlex.split(command) + hydra_args
    return cmd_parts


def check_run_status(
    task_id: str,
    exp_id: str,
    wandb_entity: str,
    wandb_project: str,
) -> Tuple[bool, Optional[Literal["running", "finished", "crashed", "killed", "failed"]]]:
    """
    Check the status of a run by querying WandB API directly.

    This function queries the WandB API in real-time to get the latest status
    of the most recent run for the given task_id. This ensures that the status
    is always up-to-date, even when multiple agents are running on different machines.

    Args:
        task_id: Task ID to check
        exp_id: Experiment ID
        wandb_entity: WandB entity name
        wandb_project: WandB project name

    Returns:
        Tuple of (exists, status):
            - exists: True if a run exists for this task
            - status: Run state or None if run doesn't exist
    """
    if not wandb_entity or not wandb_project:
        return False, None

    try:
        api = wandb.Api()
        run_path = f"{wandb_entity}/{wandb_project}"
        # Query for runs with matching exp_id and task_id
        # WandB API returns runs sorted by creation time (newest first)
        runs = api.runs(
            run_path,
            filters={"config.exp_id": exp_id, "config.task_id": task_id},
        )

        # Get the most recent run (first in the list)
        for run in runs:
            state = run.state
            logger.info(f"Found run for task {task_id}: state={state}, url={run.url}")
            return True, state

        # No runs found for this task
        return False, None

    except wandb.errors.CommError:
        logger.debug(f"No runs found in WandB for task {task_id}")
        return False, None
    except Exception:
        logger.exception(f"Error fetching WandB run status for task {task_id}")
        return False, None


def execute_single_run(
    run_config: dict,
    shared_args: dict,
    shared_custom_args: dict,
    exp_id: str,
    wandb_entity: str,
    wandb_project: str,
    global_skip_finished: bool,
    global_skip_killed: bool,
    global_skip_crashed: bool,
    global_skip_failed: bool,
):
    """Execute a single experiment run in parallel using Ray.

    Note: This function is wrapped with @ray.remote dynamically to support
    per-run GPU resource specifications.

    Args:
        run_config (dict): Configuration dictionary for the experiment run, including command and arguments.
        shared_args (dict): Shared arguments dict to merge with run args.
        shared_custom_args (dict): Shared custom arguments dict to merge with run custom_args.
        exp_id (str): Unique experiment identifier.
        wandb_entity (str): WandB entity name for status checking.
        wandb_project (str): WandB project name for status checking.
        global_skip_finished (bool): If True, skip runs that are already finished.
        global_skip_killed (bool): If True, skip runs that were killed.
        global_skip_crashed (bool): If True, skip runs that crashed.
        global_skip_failed (bool): If True, skip runs that failed.
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
        assert "task_id" in run_config["args"], "task_id is required"
        assert "logging" in run_config["args"], "logging is required"

        command = run_config["command"]
        skip_finished = run_config.get("skip_finished", global_skip_finished)
        skip_killed = run_config.get("skip_killed", global_skip_killed)
        skip_crashed = run_config.get("skip_crashed", global_skip_crashed)
        skip_failed = run_config.get("skip_failed", global_skip_failed)
        task_id = run_config["args"]["task_id"]

        # Check run status from WandB API (real-time check for latest run)
        exists, state = check_run_status(task_id, exp_id, wandb_entity, wandb_project)

        if exists and state == "running":
            log_structured("info", "Task already running, skipping", task_id=task_id, state=state)
            return {
                "success": True,
                "skipped": True,
                "reason": "already running",
                "task_id": task_id,
            }
        elif state in ["finished", "crashed", "killed", "failed"]:
            if skip_finished and state == "finished":
                log_structured("info", "Task finished, skipping", task_id=task_id, state=state)
                return {"success": True, "skipped": True, "reason": "finished", "task_id": task_id}
            if skip_killed and state == "killed":
                log_structured("info", "Task killed, skipping", task_id=task_id, state=state)
                return {"success": True, "skipped": True, "reason": "killed", "task_id": task_id}
            if skip_crashed and state == "crashed":
                log_structured("info", "Task crashed, skipping", task_id=task_id, state=state)
                return {"success": True, "skipped": True, "reason": "crashed", "task_id": task_id}
            if skip_failed and state == "failed":
                log_structured("info", "Task failed, skipping", task_id=task_id, state=state)
                return {"success": True, "skipped": True, "reason": "failed", "task_id": task_id}

        # Merge shared_args with run args (run args take precedence)
        from copy import deepcopy

        merged_args = deepcopy(shared_args)
        _deep_merge_dict(merged_args, run_config["args"])

        # Merge shared_custom_args with run custom_args
        merged_custom_args = deepcopy(shared_custom_args)
        if run_config.get("custom_args"):
            _deep_merge_dict(merged_custom_args, run_config["custom_args"])

        # Convert merged dicts to Hydra command-line arguments
        hydra_args = dict_to_hydra_args(merged_args)
        if merged_custom_args:
            custom_hydra_args = dict_to_hydra_args(merged_custom_args, is_custom_args=True)
            hydra_args = custom_hydra_args + hydra_args

        cmd_parts = build_command(command, hydra_args)

        # Execute the command
        log_structured("info", "Executing command", task_id=task_id, command=" ".join(cmd_parts))
        result = subprocess.run(cmd_parts, check=False, capture_output=True, text=True)

        if result.returncode != 0:
            log_structured("error", "Command failed", task_id=task_id, returncode=result.returncode)
            if result.stderr:
                logger.error(f"stderr: {result.stderr}")
            if result.stdout:
                logger.info(f"stdout: {result.stdout}")
            return {"success": False, "returncode": result.returncode, "task_id": task_id}
        else:
            log_structured("info", "Command completed successfully", task_id=task_id)
            if result.stdout:
                logger.debug(f"stdout: {result.stdout}")
            return {"success": True, "skipped": False, "task_id": task_id}

    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        task_id = run_config.get("args", {}).get("logging", {}).get("task_id", "unknown")
        return {"success": False, "error": str(e), "task_id": task_id}


@hydra.main(config_path="experiments", version_base=None)
def main(cfg: DictConfig):
    # Setup graceful shutdown handler
    shutdown_handler = GracefulShutdownHandler(timeout=60)
    signal.signal(signal.SIGINT, shutdown_handler.handle_signal)
    signal.signal(signal.SIGTERM, shutdown_handler.handle_signal)
    logger.info("Graceful shutdown handler installed (timeout: 60s)")

    num_workers = cfg.num_workers

    exp_id = cfg.name
    global_skip_finished = cfg.skip_finished
    global_skip_killed = cfg.skip_killed
    global_skip_crashed = cfg.skip_crashed
    global_skip_failed = cfg.skip_failed

    # Get WandB configuration from config or environment variables
    wandb_entity = cfg.get("logging", {}).get("entity") or os.getenv("WANDB_ENTITY")
    wandb_project = cfg.get("logging", {}).get("project") or os.getenv("WANDB_PROJECT")
    if not wandb_entity or not wandb_project:
        logger.warning(
            "WandB entity or project not configured. Run status checking will be disabled."
        )

    # Get shared args as dicts (will be merged at run-time)
    shared_args = (
        OmegaConf.to_container(cfg.get("shared_args"), resolve=True)
        if cfg.get("shared_args")
        else {}
    )
    shared_custom_args = (
        OmegaConf.to_container(cfg.get("shared_custom_args"), resolve=True)
        if cfg.get("shared_custom_args")
        else {}
    )

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

            task_id = run_config["args"]["task_id"]
            logger.info(f"Submitting run {task_id} with {num_gpus_required} GPU(s)")

            # Submit the task
            future = remote_fn.remote(
                run_config,
                shared_args,
                shared_custom_args,
                exp_id,
                wandb_entity,
                wandb_project,
                global_skip_finished,
                global_skip_killed,
                global_skip_crashed,
                global_skip_failed,
            )
            futures.append(future)

        # Wait for all runs to complete and collect results
        # Check for shutdown periodically while waiting
        try:
            results = []
            pending_futures = futures.copy()

            while pending_futures:
                if shutdown_handler.is_shutdown_requested():
                    logger.warning("Shutdown requested. Cancelling remaining tasks...")
                    # Cancel remaining futures
                    for f in pending_futures:
                        ray.cancel(f, force=False)
                    break

                # Wait for any task to complete with short timeout
                ready_futures, remaining_futures = ray.wait(
                    pending_futures, timeout=1.0, num_returns=1
                )

                # Collect completed results
                for future in ready_futures:
                    try:
                        result = ray.get(future, timeout=0)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to get result: {e}")

                pending_futures = remaining_futures

        except KeyboardInterrupt:
            logger.warning("Interrupted while waiting for results")

        # Get any remaining completed results
        if pending_futures:
            ready_futures, _ = ray.wait(
                pending_futures, timeout=0, num_returns=len(pending_futures)
            )
            for future in ready_futures:
                try:
                    results.append(ray.get(future, timeout=0))
                except Exception as e:
                    logger.error(f"Failed to get result: {e}")

        # Log summary of results
        logger.info("All runs completed")
        for result in results:
            if result.get("skipped"):
                log_structured(
                    "info", "Run skipped", task_id=result["task_id"], reason=result["reason"]
                )
            elif result.get("success"):
                log_structured("info", "Run completed successfully", task_id=result["task_id"])
            else:
                error_msg = result.get("error", f"returncode {result.get('returncode')}")
                log_structured("error", "Run failed", task_id=result["task_id"], error=error_msg)

    finally:
        # Cleanup Ray resources with graceful shutdown
        logger.info("Shutting down Ray...")
        try:
            ray.shutdown()
            logger.info("Ray shutdown complete")
        except Exception as e:
            logger.error(f"Error during Ray shutdown: {e}")


if __name__ == "__main__":
    main()
