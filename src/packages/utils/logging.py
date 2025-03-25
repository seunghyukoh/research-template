"""Logging utilities for the project."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
import torch


VALID_LOG_LEVELS = {
    # Support both uppercase and lowercase log levels
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    # Additional levels from training arguments
    "detail": logging.DEBUG,  # map detail to debug
    "passive": logging.WARNING,  # map passive to warning
}

DEFAULT_LOG_LEVEL = "info"  # Changed to lowercase to match training args
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def setup_logging(
    log_file: str,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_file: Path to the log file
        log_level: Logging level (detail, debug, info, warning, error, critical, passive)
        log_format: Custom log format string

    Returns:
        Logger instance

    Raises:
        ValueError: If invalid log level is provided
    """
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT

    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Validate and get log level
    if log_level is None:
        log_level = DEFAULT_LOG_LEVEL

    log_level = str(log_level)  # Convert to string but preserve case
    if log_level not in VALID_LOG_LEVELS:
        # If invalid level, warn and use default
        print(
            f"Warning: Invalid log level '{log_level}'. "
            f"Using default level '{DEFAULT_LOG_LEVEL}'. "
            f"Valid levels are: {', '.join(sorted(set(VALID_LOG_LEVELS.keys())))}"
        )
        log_level = DEFAULT_LOG_LEVEL

    # Configure logging
    logging.basicConfig(
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        level=VALID_LOG_LEVELS[log_level],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized with level: {log_level}")

    return logger


def log_system_info(logger: logging.Logger) -> None:
    """Log system information including GPU, CPU, and memory usage.

    Args:
        logger: Logger instance
    """
    # Log GPU information
    if torch.cuda.is_available():
        logger.info("GPU Information:")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            logger.info(
                f"  GPU {i}: {gpu_props.name} "
                f"(Memory: {gpu_props.total_memory / 1024**3:.1f}GB)"
            )
            if i == torch.cuda.current_device():
                memory = torch.cuda.memory_stats()
                logger.info(
                    f"  Current GPU Memory: "
                    f"Allocated: {memory['allocated_bytes.all.current'] / 1024**3:.1f}GB, "
                    f"Reserved: {memory['reserved_bytes.all.current'] / 1024**3:.1f}GB"
                )
    else:
        logger.info("No GPU available")

    # Log CPU information
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"CPU Count: {cpu_count}, Usage: {cpu_percent}%")

    # Log Memory information
    memory = psutil.virtual_memory()
    logger.info(
        f"Memory: Total={memory.total / 1024**3:.1f}GB, "
        f"Available={memory.available / 1024**3:.1f}GB, "
        f"Used={memory.percent}%"
    )


class PerformanceMonitor:
    """Monitor and log performance metrics during training."""

    def __init__(self, logger: logging.Logger, log_interval: int = 100):
        """Initialize the performance monitor.

        Args:
            logger: Logger instance
            log_interval: How often to log metrics (in steps)
        """
        self.logger = logger
        self.log_interval = log_interval
        self.step = 0
        self.start_time = datetime.now()
        self.last_log_time = self.start_time

    def step_complete(self, loss: float, batch_size: int) -> None:
        """Log performance metrics after a training step.

        Args:
            loss: Training loss
            batch_size: Batch size
        """
        self.step += 1

        if self.step % self.log_interval == 0:
            current_time = datetime.now()
            time_elapsed = current_time - self.last_log_time
            steps_per_second = self.log_interval / time_elapsed.total_seconds()
            samples_per_second = steps_per_second * batch_size

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_stats()
                gpu_memory_used = gpu_memory["allocated_bytes.all.current"] / 1024**3
            else:
                gpu_memory_used = 0

            memory = psutil.virtual_memory()

            self.logger.info(
                f"Step {self.step}: "
                f"Loss={loss:.4f}, "
                f"Steps/sec={steps_per_second:.2f}, "
                f"Samples/sec={samples_per_second:.2f}, "
                f"GPU Memory={gpu_memory_used:.1f}GB, "
                f"RAM Used={memory.percent}%"
            )

            self.last_log_time = current_time
