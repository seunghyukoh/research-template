"""Validation utilities for the project."""

from typing import Any, Dict, List, Optional

import torch

from .exceptions import ConfigurationError, ResourceError, ValidationError


def validate_system_requirements(
    required_memory_gb: float = 0.0,
    required_gpu_memory_gb: float = 0.0,
    required_gpu_count: int = 0,
) -> None:
    """Validate system requirements.

    Args:
        required_memory_gb: Required system memory in GB
        required_gpu_memory_gb: Required GPU memory in GB
        required_gpu_count: Required number of GPUs

    Raises:
        ResourceError: If system requirements are not met
    """
    import psutil

    # Check system memory
    memory = psutil.virtual_memory()
    available_memory_gb = memory.available / (1024**3)
    if required_memory_gb > 0 and available_memory_gb < required_memory_gb:
        raise ResourceError(
            f"Insufficient system memory. "
            f"Required: {required_memory_gb}GB, "
            f"Available: {available_memory_gb:.1f}GB"
        )

    # Check GPU requirements
    if required_gpu_count > 0 or required_gpu_memory_gb > 0:
        if not torch.cuda.is_available():
            raise ResourceError("GPU is required but not available")

        available_gpus = torch.cuda.device_count()
        if available_gpus < required_gpu_count:
            raise ResourceError(
                f"Insufficient GPUs. "
                f"Required: {required_gpu_count}, "
                f"Available: {available_gpus}"
            )

        if required_gpu_memory_gb > 0:
            for i in range(available_gpus):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                if gpu_memory_gb < required_gpu_memory_gb:
                    raise ResourceError(
                        f"Insufficient GPU memory on GPU {i}. "
                        f"Required: {required_gpu_memory_gb}GB, "
                        f"Available: {gpu_memory_gb:.1f}GB"
                    )


def validate_model_config(config: Dict[str, Any]) -> None:
    """Validate model configuration.

    Args:
        config: Model configuration dictionary

    Raises:
        ConfigurationError: If configuration is invalid
    """
    required_fields = ["model_path", "tokenizer_path"]
    for field in required_fields:
        if field not in config:
            raise ConfigurationError(f"Missing required field: {field}")

    if "use_lora" in config and config["use_lora"]:
        lora_fields = ["lora_rank", "lora_alpha", "lora_dropout"]
        for field in lora_fields:
            if field not in config:
                raise ConfigurationError(f"Missing required field for LoRA: {field}")
            if not isinstance(config[field], (int, float)):
                raise ConfigurationError(f"Invalid type for {field}. Expected number")
            if config[field] <= 0:
                raise ConfigurationError(f"Invalid value for {field}. Must be positive")


def validate_training_config(config: Dict[str, Any]) -> None:
    """Validate training configuration.

    Args:
        config: Training configuration dictionary

    Raises:
        ConfigurationError: If configuration is invalid
    """
    required_fields = [
        "output_dir",
        "num_train_epochs",
        "per_device_train_batch_size",
    ]
    for field in required_fields:
        if field not in config:
            raise ConfigurationError(f"Missing required field: {field}")

    if config["num_train_epochs"] <= 0:
        raise ConfigurationError("num_train_epochs must be positive")

    if config["per_device_train_batch_size"] <= 0:
        raise ConfigurationError("per_device_train_batch_size must be positive")


def validate_dataset_config(
    config: Dict[str, Any],
    allowed_datasets: Optional[List[str]] = None,
) -> None:
    """Validate dataset configuration.

    Args:
        config: Dataset configuration dictionary
        allowed_datasets: List of allowed dataset names

    Raises:
        ConfigurationError: If configuration is invalid
    """
    if not config.get("dataset_path"):
        raise ConfigurationError("dataset_path must be specified")

    if not config.get("dataset_name"):
        raise ConfigurationError("dataset_name must be specified")

    if config["dataset_name"] not in allowed_datasets:
        raise ConfigurationError(
            f"dataset_name must be one of {allowed_datasets}, got {config['dataset_name']}"
        )

    max_train = config.get("max_train_samples", -1)
    if max_train != -1 and max_train <= 0:
        raise ConfigurationError(
            "max_train_samples must be positive or -1 (for all samples)"
        )

    max_val = config.get("max_validation_samples", -1)
    if max_val != -1 and max_val <= 0:
        raise ConfigurationError(
            "max_validation_samples must be positive or -1 (for all samples)"
        )

    if not config.get("shuffle_seed"):
        raise ConfigurationError("shuffle_seed must be specified")
